import torch
from .functional import tern_op
from .difflogic import LogicLayer, GroupSum

BITS_TO_DTYPE = {
    8:  torch.int8,
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}
#Int cause vals are 0 -1 or 1

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
                    #weights.argmax(1): chosen Ternary operation 
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
        oneplus = torch.tensor(1, device=self.device, dtype=self.dtype)
        oneminus = torch.tensor(-1, device=self.device, dtype=self.dtype)

        allowed = (
            (prev_vals == zero) |
            (prev_vals == oneplus) |
            (prev_vals == oneminus)
        )
        #allowed values for prev_vals


        if not torch.all(allowed):
            raise ValueError("Input x must contain only 0, -1, or 1.")
        #Raise an error if any value is not 0, -1, or 1 

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
                layer_vals[:, i] = tern_op(prev_vals[:, a_idx], prev_vals[:, b_idx], op)
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

    def gate_statistics(self):
        from collections import Counter

        total_stats = Counter()
        layer_stats = []

        for layer_id, (_, _, layer_op) in enumerate(self.layers):
            # count gates in this layer
            layer_counter = Counter(layer_op.tolist())

            layer_stats.append(layer_counter)

            # update total statistics
            total_stats.update(layer_counter)

        return layer_stats, total_stats
    
    def network_area_2inputs(self,gates_used,total_stats):
        area={'0':0 ,'1':1.482 ,'2':1.482 ,'3':1.482 ,'4':1.482 ,'5':23.012 ,'6':23.012 ,'7':0 ,'8':3.592 ,'9':3.592 ,'10':3.592 ,'11':3.592 , 
              '12':0 ,'13':3.250 ,'14':3.250 ,'15':3.250 ,'16':3.250 ,'17':7.506 ,'18':7.506 ,'19':7.506 ,'20':7.506 ,'21':8.267 ,'22':8.267 ,
              '23':18.433 ,'24':18.433 ,'25':18.433 ,'26':18.433 ,'27':4.675 ,'28':4.675 ,'29':4.675 ,'30':4.675 ,'31':8.609 ,'32':16.780 ,'33':16.780 ,
              '34':11.440 ,'35':11.440 ,'36':18.547 ,'37':18.547 ,'38':17.863 ,'39':17.863 ,'40':20.960 ,'41':20.960 ,'42':26.300 ,'43':26.300 ,'44':26.300 ,
              '45':26.300 }
        total_area_2inputs=0

        for gate_type, count in total_stats.items():
            total_area_2inputs=total_area_2inputs + count*area.get(gates_used[gate_type],0)

        return  total_area_2inputs
    
    def network_area_mul_inputs(self,gates_used,total_stats):
        area={'0':0 ,'1':1.482 ,'2':1.482 ,'3':1.482 ,'4':1.482 ,'5':19.763 ,'6':19.763 ,'7':0 ,'8':3.592 ,'9':3.592 ,'10':3.592 ,'11':3.592 , 
              '12':0 ,'13':3.250 ,'14':3.250 ,'15':3.250 ,'16':3.250 ,'17':6.423 ,'18':6.423 ,'19':6.423 ,'20':6.423 ,'21':6.100 ,'22':6.100 ,
              '23':15.582 ,'24':15.582 ,'25':14.898 ,'26':14.898 ,'27':3.592 ,'28':3.592 ,'29':3.592 ,'30':3.592 ,'31':5.416 ,'32':11.364 ,'33':11.364 ,
              '34':11.440 ,'35':11.440 ,'36':13.530 ,'37':13.530 ,'38':13.530 ,'39':13.530 ,'40':14.917 ,'41':14.917 ,'42':19.516 ,'43':19.516 ,'44':19.516 ,
              '45':19.516 }
        total_area_mul_inputs=0

        for gate_type, count in total_stats.items():
            total_area_mul_inputs=total_area_mul_inputs + count*area.get(gates_used[gate_type],0)

        return  total_area_mul_inputs
      
