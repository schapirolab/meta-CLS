import math
import os
import sys
import traceback
import numpy as np
import ipdb

import torch
from torch import nn
from torch.nn import functional as F


class BinaryLayer(torch.autograd.Function):
    def __init__(self):
        super(BinaryLayer, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Learner(nn.Module):

    def __init__(self, config, args = None):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config
        print(config)
        self.tf_counter = 0
        self.args = args

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        self.names = []

        self.outlayer = nn.Sigmoid()

        split_index = self.args.hsplit_idx

        for i, (name, param, extra_name) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]                
                if(self.args.xav_init):
                    w = nn.Parameter(torch.ones(*param[:4]))
                    b = nn.Parameter(torch.zeros(param[0]))
                    torch.nn.init.xavier_normal_(w.data)
                    b.data.normal_(0, math.sqrt(2)/math.sqrt(1+9*b.data.shape[0]))
                    self.vars.append(w)
                    if not self.args.remove_bias:
                        self.vars.append(b)
                else:
                    w = nn.Parameter(torch.ones(*param[:4]))
                    # gain=1 according to cbfin's implementation
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    # [ch_out]
                    if not self.args.remove_bias:
                        self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                if not self.args.remove_bias:
                    self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':

                if isinstance(param[0] , int):

                    if(self.args.xav_init):
                        w_param = nn.Parameter(torch.ones(*param))
                        torch.nn.init.xavier_normal_(w_param.data)
                        self.vars.append(w_param)
                    else:     
                        # [ch_out, ch_in]
                        w_param = nn.Parameter(torch.ones(*param))
                        # gain=1 according to cbfinn's implementation
                        torch.nn.init.kaiming_normal_(w_param)
                        self.vars.append(w_param)
                    # [ch_out]

                    # for bias
                    if not self.args.remove_bias:
                        self.vars.append(nn.Parameter(torch.zeros(param[0])))

                else: 

                    for split_param in param:

                        # for weight
                        if(self.args.xav_init):
                            w_split_param = nn.Parameter(torch.ones(*split_param))
                            torch.nn.init.xavier_normal_(w_split_param.data)
                            self.vars.append(w_split_param)
                        else:     
                            # [ch_out, ch_in]
                            w_split_param = nn.Parameter(torch.ones(*split_param))
                            # gain=1 according to cbfinn's implementation
                            torch.nn.init.kaiming_normal_(w_split_param)
                            self.vars.append(w_split_param)
                        # [ch_out]

                        # for bias
                        if not self.args.remove_bias:
                            self.vars.append(nn.Parameter(torch.zeros(split_param[0])))

            elif name is 'cat':
                pass
            elif name is 'cat_start':
                pass
            elif name is "rep":
                pass
            elif name in ["residual3", "residual5", "in"]:
                pass
            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):

        info = ''

        for name, param, extra_name in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name is 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name is 'rep':
                tmp = 'rep'
                info += tmp + "\n"


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def apply_inhibition(self, x, k_ratio, multiplier_idx, k_dim=1, reshape_back=False, x_size=None):
        """
        Applies inhibitory subtraction to `x` based on top-k values along a given dimension.

        Args:
            x (Tensor): Input tensor (e.g., reshaped).
            k_ratio (float): Fraction of elements to consider for top-k.
            multiplier_idx (int): Index into self.inhibition_multiplier.
            k_dim (int): Dimension along which to compute top-k (default: 1).
            reshape_back (bool): Whether to reshape x after inhibition.
            x_size (tuple, optional): Original size to reshape back to.

        Returns:
            Tensor: Inhibited (and optionally reshaped) tensor.
        """
        curr_k = int(x.size(k_dim) * k_ratio)
        vals, _ = torch.topk(x, curr_k, dim=k_dim, sorted=False)
        inhib_amount, _ = torch.min(vals.detach(), dim=k_dim, keepdim=True)
        x = x - (self.inhibition_multiplier[multiplier_idx] * inhib_amount)

        if reshape_back:
            if x_size is None:
                raise ValueError("x_size must be provided when reshape_back is True.")
            x = torch.reshape(x, x_size)

        return x

    def forward(self, x, vars=None, bn_training=False, feature=False, record_proportion_active=False, record_hidden=False, task_index=None, lesion_large=False, lesion_small=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        # what is cat?
        cat_var = False
        cat_list = []

        x_item = None
        reached_dual_output = False

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        proportion_active = {}
        hidden_activation = {}

        k_dim = 1
        k_ratio = 0.01


        try:

            for (name, param, extra_name) in self.config:

                # assert(name == "conv2d")
                if name == 'conv2d':
                    if self.args.normalize_hidden:
                        x = F.normalize(x)
                    if not self.args.remove_bias:
                        w, b = vars[idx], vars[idx + 1]
                        x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                        idx += 2
                    else:
                        w = vars[idx]
                        x = F.conv2d(x, w, stride=param[4], padding=param[5])
                        idx += 1

                    x_size = x.size()
                    x_reshape = torch.reshape(x, (x_size[0], x_size[1]*x_size[2]*x_size[3]))

                    if self.args.learn_inhibition_multiplier:

                        multiplier_idx = int(extra_name.split("_")[1])

                        x = self.apply_inhibition(
                            x=x_reshape,
                            k_ratio=k_ratio,
                            multiplier_idx=multiplier_idx,
                            k_dim=1,
                            reshape_back=True,
                            x_size=x_size
                        )

                    # print(name, param, '\tout:', x.shape)
                elif name == 'convt2d':
                    if self.args.normalize_hidden:
                        x = F.normalize(x)
                    if not self.args.remove_bias:
                        w, b = vars[idx], vars[idx + 1]
                        x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                        idx += 2
                    else:
                        w = vars[idx]
                        x = F.conv_transpose2d(x, w, stride=param[4], padding=param[5])
                        idx += 1


                elif name == 'linear':

                    if extra_name == "non_merged_dual_output":
                        # for dual task

                        reached_dual_output = True
                        # ipdb.set_trace()

                        if self.args.normalize_hidden:
                            x1 = F.normalize(x1)

                            x2 = F.normalize(x2)

                        if extra_name == 'cosine':

                            w1_category = F.normalize(vars[idx])
                            x1_category = F.linear(x1, w1_category)
                            idx += 1

                            w2_category = F.normalize(vars[idx])
                            x2_category = F.linear(x2, w2_category)
                            idx += 1

                            w1_item = F.normalize(vars[idx])
                            x1_item = F.linear(x1, w1_item)
                            idx += 1

                            w2_item = F.normalize(vars[idx])
                            x2_item = F.linear(x2, w2_item)
                            idx += 1

                            x_category = torch.add(x1_category, x2_category)
                            x_item = torch.add(x1_item, x2_item)

                        else:
                            if not self.args.remove_bias:

                                w1_category, b1_category = vars[idx], vars[idx + 1]
                                x1_category = F.linear(x1, w1_category, b1_category)
                                idx += 2

                                w2_category, b2_category = vars[idx], vars[idx + 1]
                                x2_category = F.linear(x2, w2_category, b2_category)
                                idx += 2

                                w1_item, b1_item = vars[idx], vars[idx + 1]
                                x1_item = F.linear(x1, w1_item, b1_item)
                                idx += 2

                                w2_item, b2_item = vars[idx], vars[idx + 1]
                                x2_item = F.linear(x2, w2_item, b2_item)
                                idx += 2

                                x_category = torch.add(x1_category, x2_category)
                                x_item = torch.add(x1_item, x2_item)
                            else:

                                w1_category = vars[idx]
                                x1_category = F.linear(x1, w1_category)
                                idx += 1

                                w2_category = vars[idx]
                                x2_category = F.linear(x2, w2_category)
                                idx += 1

                                w1_item = vars[idx]
                                x1_item = F.linear(x1, w1_item)
                                idx += 1

                                w2_item = vars[idx]
                                x2_item = F.linear(x2, w2_item)
                                idx += 1

                            if lesion_large:
                                x_category = x2_category
                                x_item = x2_item
                            elif lesion_small:
                                x_category = x1_category
                                x_item = x1_item
                            else:
                                x_category = torch.add(x1_category, x2_category)
                                x_item = torch.add(x1_item, x2_item)


                        if cat_var:
                            cat_list.append(x1_category)
                            cat_list.append(x2_category)
                            cat_list.append(x1_item)
                            cat_list.append(x2_item)

                    elif extra_name == "merged_dual_output":
                        reached_dual_output = True
                        # ipdb.set_trace()

                        if self.args.normalize_hidden:
                            x = F.normalize(x)

                        if extra_name == 'cosine':

                            w_category = F.normalize(vars[idx])
                            x_category = F.linear(x, w_category)
                            idx += 1

                            w_item = F.normalize(vars[idx])
                            x_item = F.linear(x, w_item)
                            idx += 1
                        else:
                            if not self.args.remove_bias:

                                w_category, b_category = vars[idx], vars[idx + 1]
                                x_category = F.linear(x, w_category, b_category)
                                idx += 2

                                w_item, b_item = vars[idx], vars[idx + 1]
                                x_item = F.linear(x, w_item, b_item)
                                idx += 2

                            else:
                                w_category = vars[idx]
                                x_category = F.linear(x, w_category)
                                idx += 1

                                w_item = vars[idx]
                                x_item = F.linear(x, w_item)
                                idx += 1

                        if cat_var:
                            cat_list.append(x_category)
                            cat_list.append(x_item)

                    elif extra_name == "non_merged_output":

                        if self.args.normalize_hidden:

                            x1 = F.normalize(x1)
                            x2 = F.normalize(x2)
                        
                        w = vars[idx]
                        x1_output = F.linear(x1, w)
                        idx += 1

                        w = vars[idx]
                        x2_output = F.linear(x2, w)
                        idx += 1

                        x = x1_output + x2_output

                    elif "hidden_early" in extra_name:

                        multiplier_idx = int(extra_name.split("_")[3])

                        if "split" in extra_name:
                            # split layer
                            if "integration" in extra_name:

                                if self.args.normalize_hidden:
                                    x1 = F.normalize(x1)
                                    x2 = F.normalize(x2)
                                w1 = vars[idx]
                                x1 = F.linear(x1, w1)
                                idx += 1

                                w2 = vars[idx]
                                x2 = F.linear(x2, w2)
                                idx += 1


                                if "integration" in extra_name:
                                    if lesion_large:
                                        x = x2
                                    elif lesion_small:
                                        x = x1
                                    else:
                                        x = x1 + x2

                                    if self.args.learn_inhibition_multiplier:
                                        x = self.apply_inhibition(x, k_ratio, multiplier_idx)

                                else:

                                    if self.args.learn_inhibition_multiplier:
                                        x1 = self.apply_inhibition(x1, k_ratio, multiplier_idx)
                                        x2 = self.apply_inhibition(x2, k_ratio, multiplier_idx+1)
                            else:
                                if self.args.normalize_hidden:
                                    x = F.normalize(x)
                                w1 = vars[idx]
                                x1 = F.linear(x, w1)
                                idx += 1

                                w2 = vars[idx]
                                x2 = F.linear(x, w2)
                                idx += 1

                                if self.args.learn_inhibition_multiplier:
                                    x1 = self.apply_inhibition(x1, k_ratio, multiplier_idx)
                                    x2 = self.apply_inhibition(x2, k_ratio, multiplier_idx+1)

                        else:
                            # regular layer, only one input
                            if self.args.normalize_hidden:
                                x = F.normalize(x)
                            w = vars[idx]
                            x = F.linear(x, w)
                            idx += 1

                            #print("Got here: {}".format(extra_name))

                            if self.args.learn_inhibition_multiplier:
                                x = self.apply_inhibition(x, k_ratio, multiplier_idx)

                    else:
                        # ipdb.set_trace()
                        if extra_name == 'cosine':
                            w = F.normalize(vars[idx])
                            if self.args.normalize_hidden:
                                x = F.normalize(x)
                            x = F.linear(x, w)
                            idx += 1
                        else:
                            if not self.args.remove_bias:
                                if self.args.normalize_hidden:
                                    x = F.normalize(x)
                                w, b = vars[idx], vars[idx + 1]
                                x = F.linear(x, w, b)
                                idx += 2
                            else:
                                if self.args.normalize_hidden:
                                    x = F.normalize(x)
                                w = vars[idx]
                                x = F.linear(x, w)
                                idx += 1

                        if self.args.model == "lamaml_cifar":
                            multiplier_idx = int(extra_name.split("_")[1])
                            if self.args.dataset == "cifar100" and multiplier_idx < 5:
                                if self.args.learn_inhibition_multiplier:
                                    x = self.apply_inhibition(x, k_ratio, multiplier_idx)
                            else:
                                x = F.relu(x, inplace=param[0])

                        
                        if cat_var:
                            cat_list.append(x)


                elif name == 'rep':

                    if self.args.model != "lamaml_cifar":

                        if feature:
                            if self.args.num_of_item_labels > 0 and reached_dual_output:
                                return x_category, x_item
                            else:
                                return x

                elif name == "cat_start":
                    cat_var = True
                    cat_list = []

                elif name == "cat":
                    cat_var = False
                    x = torch.cat(cat_list, dim=1)

                elif name == 'bn':
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    idx += 2
                    bn_idx += 2

                elif name == 'flatten':

                    x = x.view(x.size(0), -1)

                elif name == 'reshape':

                    # [b, 8] => [b, 2, 2, 2]
                    x = x.view(x.size(0), *param)

                elif name == 'relu':

                    layer_identifier = extra_name.split("_")[1]

                    if "split_integration" in extra_name or "skip_integration" in extra_name or "regular" in extra_name:

                        x = F.relu(x, inplace=param[0])
                        x_copy = x.clone().detach()
                        n_units = x_copy.size()[0]* x_copy.size()[1]
                        n_active = torch.count_nonzero(x_copy).item()

                        if record_proportion_active:
                            if layer_identifier not in proportion_active:
                                proportion_active[layer_identifier] = [n_active/n_units]
                            else:
                                proportion_active[layer_identifier].append(n_active/n_units)

                        if record_hidden:
                            layer_act = x.detach().cpu()

                            if task_index not in hidden_activation:
                                hidden_activation[task_index] = {}
                                hidden_activation[task_index][layer_identifier] = layer_act
                            else:
                                hidden_activation[task_index][layer_identifier] = layer_act

                    elif "skip" in extra_name and "skip_integration" not in extra_name and "skip_successive" not in extra_name:
                        # if a regular skip layer

                        x1 = F.relu(x1, inplace=param[0])

                        x1_copy = x1.clone().detach()
                        n1_units = x1_copy.size()[0]* x1_copy.size()[1]
                        n1_active = torch.count_nonzero(x1_copy).item()

                        if record_proportion_active:
                            if layer_identifier not in proportion_active:
                                proportion_active[layer_identifier] = [n1_active/n1_units]
                            else:
                                proportion_active[layer_identifier].append(n1_active/n1_units)

                        if record_hidden:
                            layer_act = x1_copy.detach().cpu().numpy()
                            if task_index not in hidden_activation:
                                hidden_activation[task_index] = {}
                                hidden_activation[task_index][layer_identifier] = layer_act
                            else:
                                hidden_activation[task_index][layer_identifier] = layer_act

                    else:

                        if self.args.model == "lamaml_cifar":
                            
                            x = F.relu(x, inplace=param[0])

                            x_copy = x.clone().detach()
                            x_copy = torch.flatten(x_copy)
                            n_units = x_copy.size()[0]
                            n_active = torch.count_nonzero(x_copy).item()
                            
                            if record_proportion_active:
                                if layer_identifier not in proportion_active:
                                    proportion_active[layer_identifier] = [n_active/n_units]
                                else:
                                    proportion_active[layer_identifier].append(n_active/n_units)

                            if record_hidden:
                                layer_act = x.clone().detach().cpu().numpy()
                                hidden_activation[layer_identifier] = layer_act

                        else:

                            # there should be two outputs

                            x1 = F.relu(x1, inplace=param[0])
                            x2 = F.relu(x2, inplace=param[0])

                            x1_copy = x1.clone().detach()
                            n1_units = x1_copy.size()[0]* x1_copy.size()[1]
                            n1_active = torch.count_nonzero(x1_copy).item()

                            x2_copy = x2.clone().detach()
                            n2_units = x2_copy.size()[0]* x2_copy.size()[1]
                            n2_active = torch.count_nonzero(x2_copy).item()

                            layer_identifier1 = layer_identifier + "left"
                            layer_identifier2 = layer_identifier + "right"

                            if record_proportion_active:

                                if layer_identifier1 not in proportion_active:
                                    proportion_active[layer_identifier1] = [n1_active/n1_units]
                                else:
                                    proportion_active[layer_identifier1].append(n1_active/n1_units)

                                if layer_identifier2 not in proportion_active:
                                    proportion_active[layer_identifier2] = [n2_active/n2_units]
                                else:
                                    proportion_active[layer_identifier2].append(n2_active/n2_units)

                            if record_hidden:
                                layer_act1 = x1_copy.detach().cpu().numpy()
                                if task_index not in hidden_activation:
                                    hidden_activation[task_index] = {}
                                    hidden_activation[task_index][layer_identifier1] = layer_act1
                                else:
                                    hidden_activation[task_index][layer_identifier1] = layer_act1

                                layer_act2 = x2_copy.detach().cpu().numpy()
                                if task_index not in hidden_activation:
                                    hidden_activation[task_index] = {}
                                    hidden_activation[task_index][layer_identifier2] = layer_act2
                                else:
                                    hidden_activation[task_index][layer_identifier2] = layer_act2

                elif name == 'leakyrelu':
                    x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    x = F.tanh(x)
                elif name == 'sigmoid':
                    x = torch.sigmoid(x)
                elif name == 'upsample':
                    x = F.upsample_nearest(x, scale_factor=param[0])
                elif name == 'max_pool2d':
                    x = F.max_pool2d(x, param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    x = F.avg_pool2d(x, param[0], param[1], param[2])

                else:
                    print(name)
                    raise NotImplementedError

        except:
            traceback.print_exc(file=sys.stdout)
            ipdb.set_trace()

        # make sure variable is used properly
        if self.args.num_of_item_labels > 0:
            assert idx == len(vars)
            assert bn_idx == len(self.vars_bn)
        else:
            assert idx == len(vars)
            assert bn_idx == len(self.vars_bn)


        if self.args.num_of_item_labels > 0 and reached_dual_output:
            if record_proportion_active:
                return x_category, x_item, proportion_active
            else:
                if record_hidden:
                    return x_category, x_item, hidden_activation
                else:
                    return x_category, x_item
        else:
            if record_proportion_active:
                return x, proportion_active
            else:
                if record_hidden:
                    return x, hidden_activation
                else:
                    return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def define_task_lr_params(self, alpha_init=1e-3): 
        # Setup learning parameters
        self.alpha_lr = nn.ParameterList([])

        self.lr_name = []
        for n, p in self.named_parameters():
            self.lr_name.append(n)

        for p in self.parameters():

            if self.args.sparse_MAML == 1.0:
                alpha = nn.Parameter(torch.zeros(p.shape, requires_grad=True))

                if len(p.shape) > 1:
                    nn.init.kaiming_uniform_(alpha)
                    
                else:
                    nn.init.uniform_(alpha, a=-0.5, b=0.5)  
                    # control the mean / sparsity init explicitly
                self.alpha_lr.append(alpha)  

            elif self.args.learn_layer_lr:
                self.alpha_lr.append(nn.Parameter(alpha_init * torch.ones([1], requires_grad=True)))

            else:
                self.alpha_lr.append(nn.Parameter(alpha_init * torch.ones(p.shape, requires_grad=True)))


    def define_inhibition_multiplier(self, multiplier_init=0.00001): 
        # Setup learning parameters
        self.inhibition_multiplier = nn.ParameterList([])

        if self.args.model != "lamaml_cifar":
            num_of_multipliers = -1
            for layer_info in self.config:
                layer_name = layer_info[-1]
                if "hidden" in layer_name:
                    multiplier_idx = int(layer_name.split("_")[3])
                    if (multiplier_idx + 1) > num_of_multipliers:
                        num_of_multipliers = multiplier_idx + 1
                    if "successive" in layer_name or ("split" in layer_name and "integration" not in layer_name):
                        num_of_multipliers += 1

        if self.args.model == "lamaml_cifar":


            for idx, p in enumerate(self.parameters()):
                if idx == len(self.parameters()) - 1:
                    continue

                if len(p.shape) > 1:
                    self.inhibition_multiplier.append(nn.Parameter(multiplier_init * torch.ones([1], requires_grad=True)))  
        else:
            for i in range(num_of_multipliers):
                self.inhibition_multiplier.append(nn.Parameter(multiplier_init * torch.ones([1], requires_grad=True)))



    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


