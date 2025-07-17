import ipdb

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, sizes, dataset='mnist', args=None, n_item_outputs=0):

        net_list = []
        hidden_count = 0
        multiplier_tag = 0

        lower_split_ratio = int(5.0)

        if args.n_layers == args.hsplit_idx + 1:
            non_merge_output = True
        else:
            non_merge_output = False

        if "mnist" in dataset:
            if model_type=="linear":

                for i in range(0, len(sizes) - 1):

                    # hidden layer
                    if i != (len(sizes) - 2):

                        if hidden_count == args.hsplit_idx:
                            # split layer
                            net_list.append(('linear', [[sizes[i+1]*lower_split_ratio, sizes[i]], [sizes[i+1], sizes[i]]], 'hidden_early_' + str(hidden_count) + "_" + str(multiplier_tag) + "_split"))
                            multiplier_tag += 2 
                        elif hidden_count == (args.hsplit_idx+1):
                            net_list.append(('linear', [[sizes[i+1], sizes[i]*lower_split_ratio], [sizes[i+1], sizes[i]]], 'hidden_early_'+str(hidden_count) + "_" + str(multiplier_tag) + "_split_integration" ))
                            multiplier_tag += 1
                        else:
                            # regular layer
                            net_list.append(('linear', [sizes[i+1], sizes[i]], 'hidden_early_'+str(hidden_count) + "_" + str(multiplier_tag) + "_regular"))
                            multiplier_tag += 1

                    # output layer
                    elif i == (len(sizes) - 2):
                        print("n_item outputs : {}".format(n_item_outputs))
                        if non_merge_output:
                            if n_item_outputs > 0:
                                net_list.append(('linear', [[sizes[i+1], sizes[i]*lower_split_ratio], [sizes[i+1], sizes[i]], [n_item_outputs, sizes[i]*lower_split_ratio], [n_item_outputs, sizes[i]]], 'non_merged_dual_output'))
                            else: 
                                net_list.append(('linear', [[sizes[i+1], sizes[i]*lower_split_ratio], [sizes[i+1], sizes[i]]], 'non_merged_output'))
                        else:
                            if n_item_outputs > 0:
                                net_list.append(('linear', [[sizes[i+1], sizes[i]], [n_item_outputs, sizes[i]]], 'merged_dual_output'))
                            else:
                                net_list.append(('linear', [sizes[i+1], sizes[i]], 'output'))

                    if i < (len(sizes) - 2):
                        if hidden_count == args.hsplit_idx:
                            net_list.append(('relu', [True], 'relu_' + str(hidden_count) + "_split"))
                        elif hidden_count == (args.hsplit_idx+1):
                            net_list.append(('relu', [True], 'relu_' + str(hidden_count) + "_split_integration"))
                        else:
                            net_list.append(('relu', [True], 'relu_' + str(hidden_count) + "_regular"))
                        

                    if i == (len(sizes) - 2):
                        net_list.append(('rep', [], ''))

                    hidden_count += 1

                return net_list

        elif dataset == "tinyimagenet":

            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], 'cnn_0'),
                    ('relu', [True], 'relu_0'),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'cnn_1'),
                    ('relu', [True], 'relu_1'),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'cnn_2'),
                    ('relu', [True], 'relu_2'),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'cnn_3'),
                    ('relu', [True], 'relu_3'),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 16 * channels], 'cnn_4'),
                    ('relu', [True], 'relu_4'),

                    ('linear', [640, 640], 'cnn_5'),
                    ('relu', [True], 'relu_5'),
                    ('linear', [sizes[-1], 640], 'cnn_6')
                ]

        elif dataset == "cifar100":


            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], 'cnn_0'),
                    ('relu', [True], 'relu_0'),
                    
                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'cnn_1'),
                    ('relu', [True], 'relu_1'),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], 'cnn_2'),
                    ('relu', [True], 'relu_2'),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 16 * channels], 'cnn_3'),
                    ('relu', [True], 'relu_3'),

                    ('linear', [640, 640], 'cnn_4'),
                    ('relu', [True], 'relu_4'),
                    ('linear', [sizes[-1], 640], 'cnn_5')
                ]

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)



 