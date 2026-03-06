import torch
from .functional import bin_op
from .difflogic import LogicLayer, GroupSum

BITS_TO_DTYPE = {
    8:  torch.int8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}
#Int cause vals are 0 1 or 2

class CompiledTernaryPython(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Sequential,
            device='cpu',
            num_bits=64,
            verbose=False,
    ):
        """
        :param model:      GNN model that user wants to test
        :param device:      device (options: 'cuda' / 'cpu')
        :param num_bits: bits accuracy (options: 64,32,16,8)
        :param verbose: if true prints steps
        """
        super(CompiledTernaryPython, self).__init__()
        self.model = model
        self.device = device
        self.num_bits = num_bits
        assert num_bits in [8, 16, 32, 64]
        self.dtype = BITS_TO_DTYPE[self.num_bits]
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
                    #Each LogicLayer defines: indices[0]: Tensor of indices for input A of each neuron, indices[1]: Tensor of indices for input B of each neuron and
                    #weights.argmax(1): chosen Boolean operation 
                    # In LogicLayer, weights has shape:(num_neurons, num_of_gates). The argmax(1) means:
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
        x: torch.IntTensor of shape (batch_size, num_inputs)
        returns: torch.IntTensor of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        prev_vals = x.to(self.dtype)

        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        one = torch.tensor(1, device=self.device, dtype=self.dtype)
        two = torch.tensor(2, device=self.device, dtype=self.dtype)

        allowed = (
            (prev_vals == zero) |
            (prev_vals == one) |
            (prev_vals == two)
        )
        #allowed values for prev_vals


        if not torch.all(allowed):
            raise ValueError("Input x must contain only 0, 1, or 2.")
        #Raise an error if any value is not 0, 1, or 2 

        for layer_a, layer_b, layer_op in self.layers:
            #number of neurons in this layer 
            num_neurons = len(layer_a)
            #creates the output tensor of this layer, zero initialized 
            layer_vals = torch.zeros(batch_size, num_neurons, dtype=self.dtype,device=self.device)
            #computes the output of each neuron
            for i in range(num_neurons):
                #index of first input
                a_idx = layer_a[i]
                #indec of second input
                b_idx = layer_b[i]
                #operation of the neuron
                op = layer_op[i]
                #applies the operation 
                layer_vals[:, i] = bin_op(prev_vals[:, a_idx], prev_vals[:, b_idx], op)
            #sets layer output to prev_vals
            prev_vals = layer_vals

        # GroupSum: convert last layer outputs to num_classes
        #Computes how many neurons contribute to each class
        neurons_per_class = prev_vals.shape[1] // self.num_classes
        #creates the output tensor of groupsum layer, zero initialized 
        outputs = torch.zeros(batch_size, self.num_classes, dtype=self.dtype,device=self.device)
        for c in range(self.num_classes):
            #Sums neurons belonging to class c
            outputs[:, c] = prev_vals[:, c*neurons_per_class:(c+1)*neurons_per_class].sum(dim=1)

        return outputs

