import random
import numpy as np
import ipdb
import math

import torch
import torch.nn as nn
from model.lamaml_base import *
from model.meta.learner import BinaryLayer
from torch.nn import functional as F

class Net(BaseNet):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,           
                 args, n_item_outputs = 0):
        super(Net, self).__init__(n_inputs,
                                 n_outputs,
                                 n_tasks,           
                                 args, n_item_outputs = n_item_outputs)

        self.nc_per_task = n_outputs


    def calculate_recon_loss(self, x, x_recon, average=False):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]'''
        # print(x)
        #x = nn.Sigmoid(x)
        batch_size = x.size(0)
        reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1),
                                        reduction='none')
        reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)

        return reconL

    def forward(self, x, t, record_hidden=False, lesion_large=False, lesion_small=False):
        if self.args.num_of_item_labels > 0:
            if record_hidden:
                category_output, item_output, hidden_dict = self.net.forward(x, record_hidden=record_hidden, task_index=t, lesion_large=lesion_large, lesion_small=lesion_small)
                return category_output, item_output, hidden_dict
            else:
                category_output, item_output = self.net.forward(x, lesion_large=lesion_large, lesion_small=lesion_small)
                return category_output, item_output
        else:
            if record_hidden:
                output, hidden_dict = self.net.forward(x, record_hidden=record_hidden, task_index=t, lesion_large=lesion_large, lesion_small=lesion_small)
                return output, hidden_dict
            else:
                output = self.net.forward(x, lesion_large=lesion_large, lesion_small=lesion_small)
                return output


    def compute_category_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)

    def compute_item_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.ni_per_task
        offset2 = (task + 1) * self.ni_per_task
        return int(offset1), int(offset2)

    def take_category_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_category_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y-offset1)

        return loss

    def take_item_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_item_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y-offset1)

        return loss

    def take_multitask_loss(self, bt, t, logits, y):
        # compute loss on data from a multiple tasks
        # separate from take_loss() since the output positions for each task's
        # logit vector are different and we nly want to compute loss on the relevant positions
        # since this is a task incremental setting

        loss = 0.0

        for i, ti in enumerate(bt):
            if self.args.num_of_item_labels > 0: 
                a = 0
            else:
                offset1, offset2 = self.compute_category_offsets(ti)
                loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)

    def meta_loss(self, x, fast_weights, y, t, bt=None, z=None, record_sparsity=False, only_category_loss=False, only_item_loss=False):
        """
        differentiate the loss through the network updates wrt alpha
        """
        if self.args.num_of_item_labels > 0: 

            if record_sparsity:
                category_logits, item_logits, proportion_active = self.net.forward(x, fast_weights, record_proportion_active=True)
            else:
                category_logits, item_logits = self.net.forward(x, fast_weights)

            loss_q_category = self.loss(category_logits.squeeze(1), y)
            loss_q_item = self.loss(item_logits.squeeze(1), z)

            # # # for disabling loss_q_category
            if only_item_loss:
                loss_q_category = loss_q_category - loss_q_category

            # # # # for disabling loss_q_item
            if only_category_loss:
                loss_q_item = loss_q_item - loss_q_item

            # # synchronous update for categories and items
            loss_q = loss_q_category + loss_q_item
        
            if record_sparsity:
                return loss_q, category_logits, item_logits, proportion_active
            else:
                return loss_q, category_logits, item_logits
        else:
            if record_sparsity:
                logits, proportion_active = self.net.forward(x, fast_weights, record_proportion_active=True)
                if self.args.cuda:
                    loss_q = self.loss(logits.squeeze(1).cuda(device=1), y.cuda(device=1))
                else:
                    loss_q = self.loss(logits.squeeze(1), y)

                return loss_q, logits, proportion_active
            else:
                logits = self.net.forward(x, fast_weights)
                loss_q = self.loss(logits.squeeze(1), y)

                return loss_q, logits

    def inner_update(self, x, fast_weights, y, t, z=None):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        if self.args.num_of_item_labels == 0:
            logits = self.net.forward(x, fast_weights)
            if self.args.cuda:
                loss = self.loss(logits.cuda(device=1), y.cuda(device=1))  
            else:
                loss = self.loss(logits, y) 
        else: 

            category_logits, item_logits = self.net.forward(x, fast_weights)
            category_loss = self.loss(category_logits, y)  
            item_loss = self.loss(item_logits, z)

            loss = category_loss + item_loss 

        if fast_weights is None:
            fast_weights = self.net.parameters() 
    
        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.args.second_order
        grads = torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required)

        for i in range(len(grads)):
            torch.clamp(grads[i], min = -self.args.grad_clip_norm, max = self.args.grad_clip_norm)

        if self.args.sparse_MAML == 1.0:
            fast_weights = list(
                map(lambda p: p[1][0] -  p[0] * 0.15 *  nn.functional.relu(p[1][1]), zip(grads, zip(fast_weights, self.net.alpha_lr))))
        else:
            if self.args.no_relu_lr:
                fast_weights = list(
                    map(lambda p: p[1][0] - p[0] * p[1][1], zip(grads, zip(fast_weights, self.net.alpha_lr))))
            else:
                fast_weights = list(
                    map(lambda p: p[1][0] - p[0] * nn.functional.relu(p[1][1]), zip(grads, zip(fast_weights, self.net.alpha_lr))))


        return fast_weights

    def observe(self, x, y, t, z=None, record_variables=False):

        sparsity_batch = {}

        self.net.train() 

        for pass_itr in range(self.glances):
            self.pass_itr = pass_itr

            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            if self.args.num_of_item_labels != 0:
                z = z[perm]
            
            self.epoch += 1
            self.zero_grads()

            if t != self.current_task:
                self.M = self.M_new
                self.current_task = t

            batch_sz = x.shape[0]
            meta_losses = [0 for _ in range(batch_sz)] 

            if self.args.num_of_item_labels != 0:
                bx, by, bt, bz = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t, z=z.cpu().numpy())
            else:
                bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)

            fast_weights = None

            for i in range(0, batch_sz):

                if self.args.num_of_item_labels != 0:
                    batch_x = x[i].unsqueeze(0)
                    batch_y = y[i].unsqueeze(0)
                    batch_z = z[i].unsqueeze(0)

                    fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t, z=batch_z)
                    
                    if(self.real_epoch == 0):
                        self.push_to_mem(batch_x, batch_y, torch.tensor(t), batch_z=batch_z)

                    if record_variables:

                        meta_loss, logits_category, logits_item, sparsity = self.meta_loss(bx, fast_weights, by, t, bt=bt, z=bz, record_sparsity=True)
                        meta_losses[i] += meta_loss
                        for key in sparsity:
                            if key not in sparsity_batch:
                                sparsity_batch[key] = [sparsity[key]]
                            else:
                                sparsity_batch[key].append(sparsity[key])
                    else:
                        meta_loss, logits_category, logits_item = self.meta_loss(bx, fast_weights, by, t, bt=bt, z=bz)
                        meta_losses[i] += meta_loss

                else:

                    batch_x = x[i].unsqueeze(0)
                    batch_y = y[i].unsqueeze(0)

                    fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)

                    if(self.real_epoch == 0):
                        self.push_to_mem(batch_x, batch_y, torch.tensor(t))

                    if record_variables:

                        meta_loss, logits, sparsity = self.meta_loss(bx, fast_weights, by, t, bt=bt, record_sparsity=True) 
                        meta_losses[i] += meta_loss
                        for key in sparsity:
                            if key not in sparsity_batch:
                                sparsity_batch[key] = [sparsity[key]]
                            else:
                                sparsity_batch[key].append(sparsity[key])
                    else:
                        if self.args.cuda:
                            bx = bx.cuda(device=1)
                            by = by.cuda(device=1)
                            bt = bt.cuda(device=1)
                        meta_loss, logits = self.meta_loss(bx, fast_weights, by, t, bt=bt) 
                        meta_losses[i] += meta_loss
    
            # Taking the meta gradient step (will update the learning rates)
            self.zero_grads()

            meta_loss = sum(meta_losses)/len(meta_losses)
            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.net.alpha_lr.parameters(), self.args.grad_clip_norm)

            if self.args.learn_inhibition_multiplier:
                torch.nn.utils.clip_grad_norm_(self.net.inhibition_multiplier.parameters(), self.args.grad_clip_norm)
            if self.args.learn_lr:
                self.opt_lr.step()

            if self.args.learn_inhibition_multiplier:
                self.opt_multiplier.step()

            if(self.args.sync_update):
                self.opt_wt.step()
            else:  
                for i,p in enumerate(self.net.parameters()):
                    if self.args.sparse_MAML == 1.0:
                        p.data = p.data - p.grad * 0.15 * nn.functional.relu(self.net.alpha_lr[i])
                    else:
                        if self.args.no_relu_lr:
                            p.data = p.data - p.grad * self.net.alpha_lr[i]
                        else:
                            p.data = p.data - p.grad * nn.functional.relu(self.net.alpha_lr[i])
         
            self.net.zero_grad()
            self.net.alpha_lr.zero_grad()
            if self.args.learn_inhibition_multiplier:
                self.net.inhibition_multiplier.zero_grad()

        if record_variables:
            return meta_loss.item(), sparsity_batch
        else:
            return meta_loss.item()
