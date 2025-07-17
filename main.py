import importlib
import datetime
import argparse
import time
import os
import ipdb
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import parser as file_parser
from metrics.metrics import confusion_matrix
from utils import misc_utils
from main_multi_task import life_experience_iid, eval_iid_tasks
from model.meta.learner import BinaryLayer
from torch.nn import functional as F


def calculate_recon_loss(x, x_recon, average=False):
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

def eval_class_tasks(model, tasks, args):

    model.eval()
    result = []
    for t, task_loader in enumerate(tasks):
        rt = 0

        for (i, (x, y)) in enumerate(task_loader):

            if args.cuda:
                x = x.cuda(device=1)
            _, p = torch.max(model(x, t).data.cpu(), 1, keepdim=False)
            rt += (p == y).float().sum()
            
        result.append(rt / len(task_loader.dataset))

    return result

def eval_tasks(model, tasks, args, lesion_large=False, lesion_small=False):

    model.eval()

    if args.num_of_item_labels > 0:
        category_result = []
        item_result = []
    else:
        result = []

    if args.num_of_item_labels > 0:

        for i, task in enumerate(tasks):
            # task[0] 
            t = i
            x = task[1]
            y = task[2]
            z = task[3]
            category_rt = 0
            item_rt = 0
            
            eval_bs = x.size(0)

            for b_from in range(0, x.size(0), eval_bs):
                # print(b_from)
                b_to = min(b_from + eval_bs, x.size(0) - 1)
                if b_from == b_to:
                    xb = x[b_from].view(1, -1)
                    yb = torch.LongTensor([y[b_to]]).view(1, -1)
                    zb = torch.LongTensor([z[b_to]]).view(1, -1)
                else:
                    xb = x[b_from:b_to]
                    yb = y[b_from:b_to]
                    zb = z[b_from:b_to]
                if args.cuda:
                    xb = xb.cuda(device=1)
                    yb = yb.cuda(device=1)
                    zb = zb.cuda(device=1)
                
                if args.lesion_large or lesion_large:
                    category_output, item_output = model(xb, t, lesion_large=True)
                elif args.lesion_small or lesion_small:
                    category_output, item_output = model(xb, t, lesion_small=True)
                else:
                    category_output, item_output = model(xb, t)
                
                _, category_pb = torch.max(category_output.data.cpu(), 1, keepdim=False)
                category_rt += (category_pb == yb).float().sum()

                _, item_pb = torch.max(item_output.data.cpu(), 1, keepdim=False)
                item_rt += (item_pb == zb).float().sum()

            category_result.append(category_rt / x.size(0))
            item_result.append(item_rt / x.size(0))

        return category_result, item_result

    else:

        for i, task in enumerate(tasks):

            t = i
            x = task[1]
            y = task[2]
            rt = 0
            
            eval_bs = x.size(0)

            for b_from in range(0, x.size(0), eval_bs):
                b_to = min(b_from + eval_bs, x.size(0) - 1)
                if b_from == b_to:
                    xb = x[b_from].view(1, -1)
                    yb = torch.LongTensor([y[b_to]]).view(1, -1)
                else:
                    xb = x[b_from:b_to]
                    yb = y[b_from:b_to]
                if args.cuda:
                    xb = xb.cuda(device=1)
                    yb = yb.cuda(device=1)
                
                if args.lesion_large or lesion_large:
                    _, pb = torch.max(model(xb, t, lesion_large=True).data.cpu(), 1, keepdim=False)
                elif args.lesion_small or lesion_small:
                    _, pb = torch.max(model(xb, t, lesion_small=True).data.cpu(), 1, keepdim=False)
                else:
                    _, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
                if args.cuda:
                    pb = pb.cuda(device=1)
                rt += (pb == yb).float().sum()


            result.append(rt / x.size(0))

        return result


def save_hidden(model, tasks, args):

    model.eval()

    hidden_act = {}

    if args.num_of_item_labels > 0:

        for i, task in enumerate(tasks):
            t = i
            x = task[1]
            y = task[2]
            z = task[3]
            category_rt = 0
            item_rt = 0
            
            eval_bs = x.size(0)

            for b_from in range(0, x.size(0), eval_bs):
                # print(b_from)
                b_to = min(b_from + eval_bs, x.size(0) - 1)
                if b_from == b_to:
                    xb = x[b_from].view(1, -1)
                    yb = torch.LongTensor([y[b_to]]).view(1, -1)
                    zb = torch.LongTensor([z[b_to]]).view(1, -1)
                else:
                    xb = x[b_from:b_to]
                    yb = y[b_from:b_to]
                    zb = z[b_from:b_to]
                if args.cuda:
                    xb = xb.cuda(device=1)
                    yb = yb.cuda(device=1)
                    zb = zb.cuda(device=1)
                category_output, item_output, hidden_dict = model(xb, t, record_hidden=True)
            hidden_act[i] = hidden_dict[i]

    elif args.model == "lamaml_cifar":

        for t, task_loader in enumerate(tasks):
            rt = 0

            task_hidden_info = [{}, [], []]

            for (i, (x, y)) in enumerate(task_loader):
                if args.cuda:
                    x = x.cuda(device=1)
                    y = y.cuda(device=1)

                output, hidden_dict = model(x, t, record_hidden=True)

                class_lower = torch.min(y)
                class_upper = torch.max(y)

                for imagenet_num in range(class_lower, class_upper+1):
                    imagenet_num_indices = (y == imagenet_num).nonzero(as_tuple=True)[0][:1].tolist()

                    if args.cuda:
                        imagenet_num_indices = torch.LongTensor(imagenet_num_indices).cuda(device=1)
                    else:
                        imagenet_num_indices = torch.LongTensor(imagenet_num_indices)

                    for layer in hidden_dict:
                        if args.cuda:
                            hidden_tensor = torch.tensor(hidden_dict[layer]).cuda(device=1)
                        else:
                            hidden_tensor = torch.tensor(hidden_dict[layer])
                        if layer not in task_hidden_info[0]:
                            task_hidden_info[0][layer] = torch.index_select(hidden_tensor, 0, imagenet_num_indices)
                        else:
                            task_hidden_info[0][layer] = torch.cat((task_hidden_info[0][layer], torch.index_select(hidden_tensor, 0, imagenet_num_indices)))
            
                    if len(task_hidden_info[1]) == 0:
                        task_hidden_info[1] = torch.index_select(output, 0, imagenet_num_indices)
                        task_hidden_info[2] = torch.index_select(y, 0, imagenet_num_indices)
                    else:
                        task_hidden_info[1] = torch.cat((task_hidden_info[1], torch.index_select(output, 0, imagenet_num_indices)))
                        task_hidden_info[2] = torch.cat((task_hidden_info[2], torch.index_select(y, 0, imagenet_num_indices)))
            hidden_act[t] = task_hidden_info

    elif args.model != "lamaml_cifar":

        for i, task in enumerate(tasks):
            t = i
            x = task[1]
            y = task[2]

            rt = 0
            
            eval_bs = x.size(0)

            for b_from in range(0, x.size(0), eval_bs):
                b_to = min(b_from + eval_bs, x.size(0) - 1)
                if b_from == b_to:
                    xb = x[b_from].view(1, -1)
                    yb = torch.LongTensor([y[b_to]]).view(1, -1)
                else:
                    xb = x[b_from:b_to]
                    yb = y[b_from:b_to]
                if args.cuda:
                    xb = xb.cuda(device=1)
                    yb = yb.cuda(device=1)

                output, hidden_dict = model(xb, t, record_hidden=True)

                task_hidden_info = [{}, [], []]

                for mnist_num in range(10):
                    mnist_num_indices = (yb == mnist_num).nonzero(as_tuple=True)[0][:10].tolist()
                    if args.cuda:
                        mnist_num_indices = torch.LongTensor(mnist_num_indices).cuda(device=1)
                    else:
                        mnist_num_indices = torch.LongTensor(mnist_num_indices)

                    # hidden_dict structure: {current task:layer index:hidden data}
                    for layer in hidden_dict[i]:
                        if args.cuda:
                            hidden_tensor = hidden_dict[i][layer].cuda(device=1)
                        else:
                            hidden_tensor = hidden_dict[i][layer]
                        if mnist_num == 0:
                            task_hidden_info[0][layer] = torch.index_select(hidden_tensor, 0, mnist_num_indices)
                        else:
                            task_hidden_info[0][layer] = torch.cat((task_hidden_info[0][layer], torch.index_select(hidden_tensor, 0, mnist_num_indices)))
            
                    if mnist_num == 0:
                        task_hidden_info[1] = torch.index_select(output, 0, mnist_num_indices)
                        task_hidden_info[2] = torch.index_select(yb, 0, mnist_num_indices)
                    else:
                        task_hidden_info[1] = torch.cat((task_hidden_info[1], torch.index_select(output, 0, mnist_num_indices)))
                        task_hidden_info[2] = torch.cat((task_hidden_info[2], torch.index_select(yb, 0, mnist_num_indices)))

            hidden_act[i] = task_hidden_info

    return hidden_act

def life_experience(model, inc_loader, args):

    if args.num_of_item_labels > 0:
        result_val_a_category = []
        result_test_a_category = []

        result_val_a_item = []
        result_test_a_item = []

        result_val_t = []
        result_test_t = []
    else:
        result_val_a = []
        result_test_a = []

        result_val_t = []
        result_test_t = []

    time_start = time.time()
    test_tasks = inc_loader.get_tasks("test")
    val_tasks = inc_loader.get_tasks("val")

    sparsity_tasks = []
    multiplier_tasks = []
    learning_rate_tasks = []
    
    evaluator = eval_tasks
    if args.loader == "class_incremental_loader":
        evaluator = eval_class_tasks

    hidden_info_before = save_hidden(model, val_tasks, args)

    if args.num_of_item_labels == 0:
        task_label = "single_task_"
    else:
        task_label = "dual_task_"

    if args.hsplit_idx > -1:
        architecture = "two_pathway_"
    else:
        architecture = "single_pathway_"

    if args.learn_layer_lr:
        meta_lr = "lr_"
    else:
        meta_lr = ""

    if args.learn_inhibition_multiplier:
        meta_multiplier = "multiplier_"
    else:
        meta_multiplier = ""

    if args.alpha_init:
        init_lr = str(args.alpha_init) + "_"

    if args.learn_layer_lr:
        lr_meta_lr = str(args.opt_lr) + "_"
    else:
        lr_meta_lr = ""

    layer_size = str(args.n_hiddens) + "_"
    layer_count = str(args.n_layers) + "_"

    if args.model == "lamaml_cifar":
        torch.save(hidden_info_before, "./cnn_npy/cnn_"  + task_label + architecture + meta_lr + meta_multiplier + init_lr + lr_meta_lr + str(args.seed) + '_activation_before.pth')   
    elif args.num_of_item_labels > 0:
        torch.save(hidden_info_before, "./dual_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_activation_before.pth')
    elif args.model != "lamaml_cifar" and args.dataset == "fashion_mnist":
        torch.save(hidden_info_before, "./fashion_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_activation_before.pth') 
    elif args.model != "lamaml_cifar" and args.dataset != "fashion_mnist" and architecture == "two_pathway":
        torch.save(hidden_info_before, "./rotated_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_activation_before.pth')

    sparsity_initial = []
    learning_rate_initial = []


    for task_i in range(inc_loader.n_tasks):

        print("training task: {}",format(task_i))

        task_info, train_loader, _, _ = inc_loader.new_task()

        sparsity_task = []
        learning_rate_task = []

        for ep in range(args.n_epochs):

            model.real_epoch = ep

            prog_bar = tqdm(train_loader)

            if args.num_of_item_labels == 0:

                if task_i == 0 and ep == 0:

                    if args.learn_layer_lr:
                        learning_rate_task = []
                        for p in model.net.alpha_lr.parameters():
                            p_copy = p.clone().detach()
                            learning_rate_task.append(p.item())
                        learning_rate_tasks.append(learning_rate_task)

                    sparsity_initial = {}
                    for (i, (x, y)) in enumerate(prog_bar):
                        if args.cuda:
                            x = x.cuda(device=1)
                            y = y.cuda(device=1)
                        logits, sparsity = model.net.forward(x, record_proportion_active=True)
                        for layer_i in sparsity:
                            if layer_i not in sparsity_initial:
                                sparsity_initial[layer_i] = sparsity[layer_i]
                            else:
                                sparsity_initial[layer_i] = sparsity_initial[layer_i] + sparsity[layer_i]
                    
                    sparsity_tasks.append([sparsity_initial])

                for (i, (x, y)) in enumerate(prog_bar):

                    if((i % args.log_every) == 0):
                        result_val_a.append(evaluator(model, val_tasks, args))
                        result_val_t.append(task_info["task"])

                    v_x = x
                    v_y = y

                    if args.arch == 'linear':
                        v_x = x.view(x.size(0), -1)
                    if args.cuda:
                        v_x = v_x.cuda(device=1)
                        v_y = v_y.cuda(device=1)

                    model.train()

                    # For not recording sparsity all the time
                    if i%(1000*args.n_epochs) == 9:
                        loss, sparsity = model.observe(Variable(v_x), Variable(v_y), task_info["task"], record_variables=True)
                        sparsity_task.append(sparsity)

                    else:
                        if args.model == "lamaml_cifar":
                            loss, sparsity = model.observe(Variable(v_x), Variable(v_y), task_info["task"], record_variables=True)
                            sparsity_task.append(sparsity)
                            #print(sparsity)
                        else:
                            loss = model.observe(Variable(v_x), Variable(v_y), task_info["task"])

                    prog_bar.set_description(
                        "Task: {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                            task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                            round(sum(result_val_a[-1]).item()/len(result_val_a[-1]), 5), round(result_val_a[-1][task_info["task"]].item(), 5)
                        )
                    )
            else: 
                for (i, (x, y, z)) in enumerate(prog_bar):

                    if((i % args.log_every) == 0):
                        category_result, item_result = evaluator(model, val_tasks, args)
                        result_val_a_category.append(category_result)
                        result_val_a_item.append(item_result)
                        result_val_t.append(task_info["task"])

                    v_x = x
                    v_y = y
                    v_z = z

                    if args.arch == 'linear':
                        v_x = x.view(x.size(0), -1)
                    if args.cuda:
                        v_x = v_x.cuda(device=1)
                        v_y = v_y.cuda(device=1)
                        v_z = v_z.cuda(device=1)

                    model.train()

                    if i%(1000*args.n_epochs) == 4:
                        loss, sparsity = model.observe(Variable(v_x), Variable(v_y), task_info["task"], z=Variable(v_z), record_variables=True)
                        sparsity_task.append(sparsity)

                    else:
                        loss = model.observe(Variable(v_x), Variable(v_y), task_info["task"], z=Variable(v_z))


                    prog_bar.set_description(
                        "Task (category): {} | Epoch: {}/{} | Iter: {} | Loss: {} | Acc: Total: {} Current Task: {} ".format(
                            task_info["task"], ep+1, args.n_epochs, i%(1000*args.n_epochs), round(loss, 3),
                            round(sum(result_val_a_category[-1]).item()/len(result_val_a_category[-1]), 5), round(result_val_a_category[-1][task_info["task"]].item(), 5)
                        )
                    )

        sparsity_tasks.append(sparsity_task)

        if args.learn_inhibition_multiplier:
            multiplier_task = []
            for multiplier in model.net.inhibition_multiplier.parameters():
                multiplier_task.append(multiplier.item())
            print(multiplier_task)
            multiplier_tasks.append(multiplier_task)

        if args.learn_layer_lr:
            learning_rate_task = []
            for p in model.net.alpha_lr.parameters():
                p_copy = p.clone().detach()
                learning_rate_task.append(p.item())
                print(p_copy)
            learning_rate_tasks.append(learning_rate_task)

        if args.model == "lamaml_cifar":
            if args.num_of_item_labels == 0:
                result_val_a.append(evaluator(model, val_tasks, args))
                result_val_t.append(task_info["task"])

                if args.calc_test_accuracy:
                    result_test_a.append(evaluator(model, test_tasks, args))
                    result_test_t.append(task_info["task"])
            else:
                category_result, item_result = evaluator(model, val_tasks, args)
                result_val_a_category.append(category_result)
                result_val_a_item.append(item_result)
                result_val_t.append(task_info["task"])

                if args.calc_test_accuracy:
                    category_test_result, item_test_result = evaluator(model, test_tasks, args)
                    result_test_a_category.append(category_test_result)
                    result_test_a_item.append(item_test_result)
                    result_test_t.append(task_info["task"])

        if args.model != "lamaml_cifar":
            if args.num_of_item_labels == 0:
                result_val_a.append(evaluator(model, val_tasks, args))
                result_val_t.append(task_info["task"])

                if args.calc_test_accuracy:
                    result_test_a.append(evaluator(model, test_tasks, args))
                    result_test_t.append(task_info["task"])
            else:
                category_result, item_result = evaluator(model, val_tasks, args)
                result_val_a_category.append(category_result)
                result_val_a_item.append(item_result)
                result_val_t.append(task_info["task"])

                if args.calc_test_accuracy:
                    category_test_result, item_test_result = evaluator(model, test_tasks, args)
                    result_test_a_category.append(category_test_result)
                    result_test_a_item.append(item_test_result)
                    result_test_t.append(task_info["task"])


    hidden_info_after = save_hidden(model, val_tasks, args)
    
    print(sparsity_tasks)
    print(learning_rate_tasks)


    if args.num_of_item_labels == 0:
        task_label = "single_task_"
    else:
        task_label = "dual_task_"

    if args.learn_layer_lr:
        meta_lr = "lr_"
    else:
        meta_lr = ""

    if args.learn_inhibition_multiplier:
        meta_multiplier = "multiplier_"
    else:
        meta_multiplier = ""

    if args.hsplit_idx > -1:
        architecture = "two_pathway_"
    else:
        architecture = "single_pathway_"
    
    if args.alpha_init:
        init_lr = str(args.alpha_init) + "_"

    if args.learn_layer_lr:
        lr_meta_lr = str(args.opt_lr) + "_"
    else:
        lr_meta_lr = ""

    layer_size = str(args.n_hiddens) + "_"
    layer_count = str(args.n_layers) + "_"

    if args.model == "lamaml_cifar":
        torch.save(hidden_info_after, "./cnn_npy/cnn_" + task_label + architecture + meta_lr + meta_multiplier + init_lr + lr_meta_lr + str(args.seed) + '_activation_after.pth')
    elif args.num_of_item_labels > 0:
        torch.save(hidden_info_after, "./dual_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_activation_after.pth')
    elif args.model != "lamaml_cifar" and args.dataset == "fashion_mnist":
        torch.save(hidden_info_after, "./fashion_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_activation_after.pth')
    elif args.model != "lamaml_cifar" and args.dataset != "fashion_mnist":
        torch.save(hidden_info_after, "./rotated_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_activation_after.pth')
    

    if args.model == "lamaml_cifar":
        torch.save(sparsity_tasks, "./cnn_npy/cnn_" + task_label + architecture + meta_lr + meta_multiplier + init_lr + lr_meta_lr + str(args.seed) + '_sparsity_after.pth')
        torch.save(multiplier_tasks, "./cnn_npy/cnn_" + task_label + architecture + meta_lr + meta_multiplier + init_lr + lr_meta_lr + str(args.seed) + '_multiplier_after.pth')
        torch.save(learning_rate_tasks, "./cnn_npy/cnn_" + task_label + architecture + meta_lr + meta_multiplier + init_lr + lr_meta_lr + str(args.seed) + '_learning_rate_after.pth')
    
    elif args.num_of_item_labels > 0:

        torch.save(sparsity_tasks, "./dual_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_sparsity_after.pth')
        torch.save(multiplier_tasks, "./dual_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_multiplier_after.pth')
        torch.save(learning_rate_tasks, "./dual_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_learning_rate_after.pth')
        
    elif args.model != "lamaml_cifar" and args.dataset == "fashion_mnist":


        torch.save(sparsity_tasks, "./fashion_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_sparsity_after.pth')
        torch.save(multiplier_tasks, "./fashion_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_multiplier_after.pth')
        torch.save(learning_rate_tasks, "./fashion_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_learning_rate_after.pth')
        
    elif args.model != "lamaml_cifar" and args.dataset != "fashion_mnist":

        torch.save(sparsity_tasks, "./rotated_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_sparsity_after.pth')
        torch.save(multiplier_tasks, "./rotated_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_multiplier_after.pth')
        torch.save(learning_rate_tasks, "./rotated_npy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_learning_rate_after.pth')

    if args.num_of_item_labels == 0:

        print("####Final Validation Accuracy####")
        print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a[-1])/len(result_val_a[-1]), result_val_a[-1]))

        if args.model == "lamaml_cifar":
            torch.save(result_val_a, "./cnn_accuracy/cnn_" + task_label + architecture + meta_lr + meta_multiplier + init_lr + lr_meta_lr + str(args.seed) + '_validation_accuracy.pth')
        elif args.two_pathway:
            torch.save(result_val_a, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_validation_accuracy.pth')
        elif args.model != "lamaml_cifar" and args.dataset == "fashion_mnist":
            torch.save(result_val_a, "./fashion_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_validation_accuracy.pth')
        elif args.model != "lamaml_cifar" and args.dataset != "fashion_mnist":
            torch.save(result_val_a, "./rotated_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_validation_accuracy.pth')

        if args.calc_test_accuracy:
            print("####Final Test Accuracy####")
            print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_test_a[-1])/len(result_test_a[-1]), result_test_a[-1]))

        if args.model == "lamaml_cifar":
            torch.save(result_test_a, "./cnn_accuracy/cnn_" + task_label + architecture + meta_lr + meta_multiplier + init_lr + lr_meta_lr + str(args.seed) + '_test_accuracy.pth')
        elif args.model != "lamaml_cifar" and args.dataset == "fashion_mnist":
            torch.save(result_test_a, "./fashion_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_test_accuracy.pth')
        elif args.model != "lamaml_cifar" and args.dataset != "fashion_mnist":
            torch.save(result_test_a, "./rotated_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_test_accuracy.pth')

        time_end = time.time()
        time_spent = time_end - time_start
        return torch.Tensor(result_val_t), torch.Tensor(result_val_a), torch.Tensor(result_test_t), torch.Tensor(result_test_a), time_spent

    else:
        print("####Final Validation Accuracy####")
        print("Final Results (category):- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a_category[-1])/len(result_val_a_category[-1]), result_val_a_category[-1]))
        print("Final Results (item):- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_val_a_item[-1])/len(result_val_a_item[-1]), result_val_a_item[-1]))

        if args.calc_test_accuracy:

            print("####Final Test Accuracy####")
            print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_test_a_category[-1])/len(result_test_a_category[-1]), result_test_a_category[-1]))
            print("Final Results:- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(result_test_a_item[-1])/len(result_test_a_item[-1]), result_test_a_item[-1]))

        torch.save(result_val_a_category, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_validation_category_accuracy.pth')
        torch.save(result_val_a_item, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_validation_item_accuracy.pth')
        torch.save(result_test_a_category, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_test_category_accuracy.pth')
        torch.save(result_test_a_item, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_test_item_accuracy.pth')
        
        if args.hsplit_idx > -1:

            no_large_category_result, no_large_item_result = evaluator(model, val_tasks, args, lesion_large=True)

            print("####Lesion large: Final Validation Accuracy####")
            print("Final Results (category):- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(no_large_category_result)/len(no_large_category_result), no_large_category_result))
            print("Final Results (item):- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(no_large_item_result)/len(no_large_item_result), no_large_item_result))

            no_small_category_result, no_small_item_result = evaluator(model, val_tasks, args, lesion_small=True)

            print("####Lesion small: Final Validation Accuracy####")
            print("Final Results (category):- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(no_small_category_result)/len(no_small_category_result), no_small_category_result))
            print("Final Results (item):- \n Total Accuracy: {} \n Individual Accuracy: {}".format(sum(no_small_item_result)/len(no_small_item_result), no_small_item_result))

            torch.save(no_large_category_result, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_no_large_category_accuracy.pth')
            torch.save(no_large_item_result, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_no_large_item_accuracy.pth')
            torch.save(no_small_category_result, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_no_small_category_accuracy.pth')
            torch.save(no_small_item_result, "./dual_accuracy/" + task_label + architecture + meta_lr + meta_multiplier + layer_size + layer_count + init_lr + lr_meta_lr + str(args.seed) + '_no_small_item_accuracy.pth')

        time_end = time.time()
        time_spent = time_end - time_start
        return torch.Tensor(result_val_t), [torch.Tensor(result_val_a_category), torch.Tensor(result_val_a_item)], torch.Tensor(result_test_t), [torch.Tensor(result_test_a_category), torch.Tensor(result_test_a_item)], time_spent


def save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time):
    fname = os.path.join(args.log_dir, 'results')

    # save confusion matrix and print one line of stats
    val_stats = confusion_matrix(result_val_t, result_val_a, args.log_dir, 'results.txt')
    
    one_liner = str(vars(args)) + ' # val: '
    one_liner += ' '.join(["%.3f" % stat for stat in val_stats])

    test_stats = 0
    if args.calc_test_accuracy:
        test_stats = confusion_matrix(result_test_t, result_test_a, args.log_dir, 'results.txt')
        one_liner += ' # test: ' +  ' '.join(["%.3f" % stat for stat in test_stats])

    # save all results in binary file
    torch.save((result_val_t, result_val_a, model.state_dict(),
                val_stats, one_liner, args), fname + '.pt')
    return val_stats, test_stats

def main():
    parser = file_parser.get_parser()

    args = parser.parse_args()

    # initialize seeds
    misc_utils.init_seed(args.seed)

    Loader = importlib.import_module('dataloaders.' + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks = loader.get_dataset_info()

    # setup logging
    timestamp = misc_utils.get_date_time()
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp)

    # load model
    Model = importlib.import_module('model.' + args.model)

    # Model.Net calls learner.py to instantiate the model
    if args.num_of_item_labels != 0:
        if args.item_option == "per_item":
            n_item_outputs = n_tasks * args.samples_per_task
        else:       
            n_item_outputs = args.samples_per_task
        model = Model.Net(n_inputs, n_outputs, n_tasks, args, n_item_outputs=n_item_outputs)
    else:
        model = Model.Net(n_inputs, n_outputs, n_tasks, args)

    if args.cuda:
        try:
            model.net.cuda(device=1)            
        except:
            pass 

    # run model on loader
    if args.model == "iid2":
        # oracle baseline with all task data shown at same time
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience_iid(
            model, loader, args)
    else:
        # for all the CL baselines
        result_val_t, result_val_a, result_test_t, result_test_a, spent_time = life_experience(
            model, loader, args)

        # save results in files or print on terminal
        if args.num_of_item_labels == 0:
            save_results(args, result_val_t, result_val_a, result_test_t, result_test_a, model, spent_time)


if __name__ == "__main__":
    main()
