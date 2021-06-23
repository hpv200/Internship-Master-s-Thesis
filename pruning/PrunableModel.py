import math

import torch
import torch.nn as nn

import re

import numpy as np
import pandas as pd


class PrunableModel(nn.Module):

    def __init__(self):
        super(PrunableModel, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

    def _post_init(self):
        """ initializes all pruning components """

        # define a mask for each parameter vector
        self.mask_dict = {
            name + ".weight": torch.ones_like(module.weight.data, device=self.device)
            for name, module in self.named_modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        }

        # define vectors for each output shape. this shape is shared with the input shape from the next layer
        self.structured_vectors = {name: torch.ones(tens.shape[0], device=self.device) for name, tens in
                                   [x for x in self.mask][:-1]}

    def magnitude_prune_unstructured(self, percentage=0.0):
        """
        sets mask based on percentage (of remaining weights) and magnitude of weights
        does global magnitude pruning
        """

        percentage = percentage + self.unstructured_sparsity

        # get threshold
        all_weights = torch.cat(
            [torch.flatten(x) for name, x in self.named_parameters() if name in self.mask_dict]
        )
        count = len(all_weights)
        amount = int(count * percentage)
        limit = torch.topk(all_weights.abs(), amount, largest=False).values[-1]

        # prune
        for (name, weights) in self.named_parameters():
            if name in self.mask_dict:
                # prune on l1
                mask = weights.abs() > limit
                self.mask_dict[name] = mask
        self.apply_mask()

    def magnitude_prune_structured(self, percentage = 0.0):
        """
        sets structured mask based on percentage (of remaining nodes) and magnitude of rows and columns
        does layer-wise magnitude pruning
        """
        percentage = percentage + self.structured_sparsity

        # determine how many nodes we are gonna prune per layer
        prunable_nodes = {
            name: math.ceil(percentage * x.shape[0]) for name, x in self.structured_vectors.items()
        }

        self.prunable_nodes = prunable_nodes

        to_prune = sum([y for x, y in prunable_nodes.items()])
        if (self.get_num_nodes * percentage) < to_prune:
            diff = int(torch.round(to_prune - self.get_num_nodes * percentage))
            prunable_nodes[list(prunable_nodes.keys())[0]] -= diff

        # count magnitudes
        # weight_counts = {}
        # last_name = None
        # for name, param in self.named_parameters():
        #     if 'weight' in name and not ("Norm" in str(param.__class__)):
        #         magnitude_output = param.abs().sum(dim=1)
        #         magnitude_input = param.abs().sum(dim=0)
        #         # input shape belongs to the magnitude of the previous layer
        #         if last_name is not None and last_name in weight_counts:
        #             weight_counts[last_name] += magnitude_input
        #         if name in self.structured_vectors:
        #             weight_counts[name] = magnitude_output
        #             last_name = name

        weight_counts = {}
        weight_counts['features'] = {}
        weight_counts['classifier'] = {}
        last_name = None
        for name, layer in self.named_modules():
            if (isinstance(layer, nn.Conv2d)):
                magnitude_output = layer.weight.abs().sum([1, 2, -1])  # 32 ; 64
                magnitude_input = layer.weight.abs().sum([0, 2, -1])  # 1 ;  32

                layer_num = int(re.findall('\.(.*)', name)[0])

                if last_name is not None and last_name in weight_counts['features']:
                    weight_counts['features'][last_name] += magnitude_input

                if (name + '.weight') in self.structured_vectors.keys():
                    weight_counts['features'][layer_num] = magnitude_output
                    last_name = layer_num

            elif (isinstance(layer, nn.Linear)):

                if len(weight_counts['classifier']) > 0:
                    magnitude_output = layer.weight.abs().sum(dim=1)
                    magnitude_input = layer.weight.abs().sum(dim=0)

                    layer_num = int(re.findall('\.(.*)', name)[0])

                    if last_name is not None and last_name in weight_counts['classifier']:
                        weight_counts['classifier'][last_name] += magnitude_input
                    if (name + '.weight') in self.structured_vectors:
                        weight_counts['classifier'][layer_num] = magnitude_output
                        last_name = layer_num

                elif len(weight_counts['classifier']) == 0:
                    layer_num = int(re.findall('\.(.*)', name)[0])
                    last_name = layer_num


                    #adds to last conv layer
                    reshaped = torch.reshape(layer.weight, (layer.weight.shape[0],magnitude_output.shape[0], int(layer.weight.shape[1]/magnitude_output.shape[0])))
                    magnitude_input = reshaped.abs().sum([0,2])

                    weight_counts['features'][list(weight_counts['features'].keys())[-1]] += magnitude_input

                    #adds to the first fc layer
                    magnitude_output = layer.weight.abs().sum(dim=1)
                    weight_counts['classifier'][layer_num] = magnitude_output

        if len(weight_counts['features']) == 0:
            for name, counts in weight_counts.items():
                limit = torch.topk(counts.abs(), prunable_nodes[name], largest=False).values[-1]
                self.structured_vectors[name] = (counts > limit).float()
        else:
            for layer_type in weight_counts:
                for name, counts in weight_counts[layer_type].items():
                    limit = torch.topk(counts.abs(), prunable_nodes[layer_type+'.'+str(name)+'.weight'], largest=False).indices
                    self.structured_vectors[layer_type+'.'+str(name)+'.weight'][limit] = 0

        self.apply_structured_vectors()
        self.apply_mask()
    def apply_structured_vectors(self):

        """ applies structured vectors to masks """
        last_name = None
        for name, mask_param in self.mask:
            # apply vector to input dimension

            if last_name is not None and last_name in self.structured_vectors:
                masker = self.structured_vectors[last_name]
                if len(mask_param.data.shape) == 4:
                    shapes_4d = mask_param.data.shape
                    true_masker = torch.ones(shapes_4d, device=self.device)
                    indices = np.where(masker.cpu().numpy() == 0)[0]

                    for x in indices:
                        true_masker[:,x,:,:] = 0

                    mask_param.data *= true_masker
                else:
                    if mask_param.data.shape[-1] == masker.shape[0]:
                        mask_param.data = mask_param.data * masker

                    else:
                        mask_param.data *= masker.repeat_interleave(int(mask_param.shape[-1]/masker.shape[0]))
                        a = 0


            # apply vector to output dimension
            if name in self.structured_vectors:
                masker = self.structured_vectors[name]

                if len(mask_param.data.shape) == 4:
                    shapes_4d = mask_param.data.shape
                    temp = torch.reshape(mask_param.data, (shapes_4d[0], shapes_4d[1] * shapes_4d[2] * shapes_4d[3] ))
                    temp = (temp.t() * masker).t()
                    mask_param.data = torch.reshape(temp, shapes_4d)
                else:
                    mask_param.data = (mask_param.data.t() * masker).t()

                last_name = name

    @property
    def mask(self):
        """ iterator as property """
        return self.mask_dict.items()

    @property
    def get_num_nodes(self):
        """ counts number of nodes in network """
        counter = 0
        last_addition = 0
        for i, (name, module) in enumerate(self.named_modules()):
            if (hasattr(module, "weight")) and not ("Norm" in str(module.__class__)):
                last_addition = module.weight.shape[0]  # add output shape each time
                counter += last_addition
        a = 0

        return counter - last_addition  # dont prune the output layer so remove last addition

    @property
    def get_num_nodes_unpruned(self):
        """ counts number of nodes in network which are still active """
        counter = 0
        last_addition = 0
        for i, (name, module) in enumerate(self.named_modules()):
            if hasattr(module, "in_features"):
                last_addition = (module.weight.abs().sum(dim=1) > 0).sum()  # add output shape each time of all nonzero
                counter += last_addition

            elif (hasattr(module, "in_channels")):
                for x in range(module.weight.shape[0]):
                    last_addition = (module.weight[x].abs().sum() > 0).sum()  # add output shape each time of all nonzero
                    counter += last_addition
        return counter - last_addition  # dont prune the output layer so remove last addition

    @property
    def get_num_weights(self):
        """ count number of weights in network """
        counter = 0
        for i, (name, module) in enumerate(self.named_modules()):
            if (hasattr(module, "weight")) and not ("Norm" in str(module.__class__)):
                counter += module.weight.shape[1] * module.weight.shape[0]
        return counter

    @property
    def get_num_weights_unpruned(self):
        """ count number of weights in network which are still active """

        counter = 0
        for i, mask_tensor in self.mask:
            counter += (mask_tensor > 0).sum()
        return counter

    @property
    def structured_sparsity(self):
        """ calculate structured sparsity """
        return (1 - (self.get_num_nodes_unpruned) / (self.get_num_nodes))

    @property
    def unstructured_sparsity(self):
        """ calculate unstructured sparsity """
        return (1.0 - ((self.get_num_weights_unpruned) / (self.get_num_weights))).item()

    def apply_mask(self):
        """ applies mask to both grads and weights """
        self._apply_grad_mask()
        self._apply_weight_mask()

    def _apply_weight_mask(self):
        """ applies mask to weights """

        with torch.no_grad():
            last_vector = 0
            for name, tensor in self.named_parameters():
                if name in self.mask_dict:
                    tensor.data *= self.mask_dict[name]

                if ('bias' in name) & (name[:-4] + 'weight' in self.structured_vectors):
                    tensor.data *= self.structured_vectors[name[:-4] + 'weight']


    def _apply_grad_mask(self):
        """ applies mask to grads """

        with torch.no_grad():
            for name, tensor in self.named_parameters():
                if name in self.mask_dict and tensor.grad is not None:
                    tensor.grad.data *= self.mask_dict[name]
