import random
from random import shuffle
import numpy as np
import ipdb
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import model.meta.learner as Learner
import model.meta.modelfactory as mf
from scipy.stats import pearsonr
import datetime

def calculate_recon_loss(x, x_recon, average=False):
    '''Calculate reconstruction loss for each element in the batch.

    INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
            - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
            - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

    OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]'''

    batch_size = x.size(0)
    reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1),
                                    reduction='none')
    reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)

    return reconL

class BaseNet(torch.nn.Module):



    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,          
                 args, n_item_outputs = 0):
        super(BaseNet, self).__init__()


        self.args = args
        nl, nh = args.n_layers, args.n_hiddens

        config = mf.ModelFactory.get_model(model_type = args.arch, sizes = [n_inputs] + [nh] * nl + [n_outputs],
                                                dataset = args.dataset, args=args, n_item_outputs=n_item_outputs)

        self.net = Learner.Learner(config, args)

        # define the lr params
        self.net.define_task_lr_params(alpha_init = args.alpha_init)

        inhibition_initialization = 0.01

        if self.args.learn_inhibition_multiplier:
            self.net.define_inhibition_multiplier(multiplier_init = inhibition_initialization)
            self.opt_multiplier = torch.optim.SGD(list(self.net.inhibition_multiplier.parameters()), lr=args.multiplier_lr) 
            
        self.opt_wt = torch.optim.SGD(list(self.net.parameters()), lr=args.opt_wt) 

        if args.sparse_MAML == 1.0:
            mask_lr = 0.0075
            momentum_init = 0.9
            self.opt_lr = torch.optim.SGD(list(self.net.alpha_lr.parameters()), lr=mask_lr, 
                                                momentum=momentum_init,
                                                nesterov=(momentum_init > 0.0))
        else: 
            self.opt_lr = torch.optim.SGD(list(self.net.alpha_lr.parameters()), lr=args.opt_lr) 

        self.epoch = 0
        # allocate buffer
        self.M = []        
        self.M_new = []
        self.age = 0

        self.loss = torch.nn.CrossEntropyLoss()
        self.is_cifar = ((args.dataset == 'cifar100') or (args.dataset == 'tinyimagenet'))
        self.glances = args.glances
        self.pass_itr = 0
        self.real_epoch = 0

        self.current_task = 0
        self.memories = args.memories
        self.batchSize = int(args.replay_batch_size)

        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda(device=1)

        self.n_outputs = n_outputs

    

    def push_to_mem(self, batch_x, batch_y, t, batch_z=None):
        """
        Reservoir sampling to push subsampled stream
        of data points to replay/memory buffer
        """

        if(self.real_epoch > 0 or self.pass_itr>0):
            return
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()  
        if self.args.num_of_item_labels != 0:
            batch_z = batch_z.cpu()            
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M_new) < self.memories:
                if self.args.num_of_item_labels != 0:
                    self.M_new.append([batch_x[i], batch_y[i], t, batch_z[i]])
                else:
                    self.M_new.append([batch_x[i], batch_y[i], t])
            else:
                p = random.randint(0,self.age)  
                if p < self.memories:
                    if self.args.num_of_item_labels != 0:
                        self.M_new[p] = [batch_x[i], batch_y[i], t, batch_z[i]]
                    else:
                        self.M_new[p] = [batch_x[i], batch_y[i], t]


    def getBatch(self, x, y, t, z=None, batch_size=None):
        """
        Given the new data points, create a batch of old + new data, 
        where old data is sampled from the memory buffer
        """

        if(x is not None):
            mxi = np.array(x)
            myi = np.array(y)
            mti = np.ones(x.shape[0], dtype=int)*t   
            if self.args.num_of_item_labels != 0:
                mzi = np.array(z)     
        else:
            mxi = np.empty( shape=(0, 0) )
            myi = np.empty( shape=(0, 0) )
            mti = np.empty( shape=(0, 0) )
            if self.args.num_of_item_labels != 0:
                mzi = np.empty( shape=(0, 0) )    

        bxs = []
        bys = []
        bts = []
        if self.args.num_of_item_labels != 0:
            bzs = []

        if self.args.use_old_task_memory and t>0:
            MEM = self.M
        else:
            MEM = self.M_new

        batch_size = self.batchSize if batch_size is None else batch_size

        if len(MEM) > 0:
            order = [i for i in range(0,len(MEM))]
            osize = min(batch_size,len(MEM))
            for j in range(0,osize):
                shuffle(order)
                k = order[j]

                if self.args.num_of_item_labels != 0:
                    x,y,t,z = MEM[k]

                    xi = np.array(x)
                    yi = np.array(y)
                    ti = np.array(t)
                    zi = np.array(z)

                    bxs.append(xi)
                    bys.append(yi)
                    bts.append(ti)
                    bzs.append(zi)
                else:
                    x,y,t = MEM[k]

                    xi = np.array(x)
                    yi = np.array(y)
                    ti = np.array(t)
                    bxs.append(xi)
                    bys.append(yi)
                    bts.append(ti)

        for j in range(len(myi)):
            if self.args.num_of_item_labels != 0:
                bxs.append(mxi[j])
                bys.append(myi[j])
                bts.append(mti[j])
                bzs.append(mzi[j])
            else:
                bxs.append(mxi[j])
                bys.append(myi[j])
                bts.append(mti[j])

        bxs = Variable(torch.from_numpy(np.array(bxs))).float() 
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)
        if self.args.num_of_item_labels != 0:
            bzs = Variable(torch.from_numpy(np.array(bzs))).long().view(-1)
        
        # handle gpus if specified
        if self.cuda:
            bxs = bxs.cuda(device=1)
            bys = bys.cuda(device=1)
            bts = bts.cuda(device=1)
            if self.args.num_of_item_labels != 0:
                bzs = bzs.cuda()

        if self.args.num_of_item_labels != 0:
            return bxs,bys,bts,bzs
        else:
            return bxs,bys,bts

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.nc_per_task
        offset2 = (task + 1) * self.nc_per_task
        return int(offset1), int(offset2)

    def zero_grads(self):
        if self.args.learn_lr:
            self.opt_lr.zero_grad()
        self.opt_wt.zero_grad()
        self.net.zero_grad()
        self.net.alpha_lr.zero_grad()
        if self.args.learn_inhibition_multiplier:
            self.net.inhibition_multiplier.zero_grad()

            
