import torch
from .functional import bin_op
from .difflogic import LogicLayer, GroupSum


class CompiledPython(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Sequential,
            device='cpu',
            verbose=False,
    ):
        super(CompiledPython, self).__init__()
        self.model = model
        self.device = device
        if self.model is not None:
            layers = []

            self.num_inputs = None
            #Model Structure Validation. 
            assert isinstance(self.model[-1], GroupSum), 'The last layer of the model must be GroupSum, but it is {} / {}' \
                                                         ' instead.'.format(type(self.model[-1]), self.model[-1])
            self.num_classes = self.model[-1].k
            #if self.model[-1] is a GroupSum object, then it has a variable k=number of intended real valued outputs, e.g., number of classes
            #Extracting Logic Layers
            first = True
            for layer in self.model:
                #compiles only logic layers
                if isinstance(layer, LogicLayer):
                    if first:
                        self.num_inputs = layer.in_dim
                        first = False
                    self.num_out_per_class = layer.out_dim // self.num_classes
                    layers.append((layer.indices[0], layer.indices[1], layer.weights.argmax(1)))
                    #Each LogicLayer defines: indices[0]: index of input A, indices[1]: index of input B and
                    #weights.argmax(1): chosen Boolean operation 
                    # In LogicLayer, weights has shape:(num_neurons, 16). The argmax(1) means:
                    # for each neuron, find the index (gate) of the largest weight across the 16 operations.                
                elif isinstance(layer, torch.nn.Flatten):
                    if verbose:
                        print('Skipping torch.nn.Flatten layer ({}).'.format(type(layer)))
                elif isinstance(layer, GroupSum):
                    if verbose:
                        print('Skipping GroupSum layer ({}).'.format(type(layer)))
                else:
                    assert False, 'Error: layer {} / {} unknown.'.format(type(layer), layer)

            self.layers = layers

            if verbose:
                print('`layers` created and has {} layers.'.format(len(layers)))

    def forward(self, x):
        """
        x: torch.BoolTensor of shape (batch_size, num_inputs)
        returns: torch.IntTensor of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        prev_vals = x.bool().int()  # convert bool → int for bitwise ops

        for layer_a, layer_b, layer_op in self.layers:
            num_neurons = len(layer_a)
            layer_vals = torch.zeros(batch_size, num_neurons, dtype=torch.int32)
            for i in range(num_neurons):
                a_idx = layer_a[i]
                b_idx = layer_b[i]
                op = layer_op[i]
                layer_vals[:, i] = bin_op(prev_vals[:, a_idx], prev_vals[:, b_idx], op)
            prev_vals = layer_vals

        # GroupSum: convert last layer outputs to num_classes
        neurons_per_class = prev_vals.shape[1] // self.num_classes
        outputs = torch.zeros(batch_size, self.num_classes, dtype=torch.int32)
        for c in range(self.num_classes):
            outputs[:, c] = prev_vals[:, c*neurons_per_class:(c+1)*neurons_per_class].sum(dim=1)

        return outputs

