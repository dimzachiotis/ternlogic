import torch
import math
from .difflogic import LogicLayer, GroupSum
import tempfile
import subprocess
import shutil
import ctypes
import numpy as np
import numpy.typing
import time
from typing import Union

# Bit-width mappings for C types, NumPy dtypes, ctypes, and literals
BITS_TO_DTYPE        = {8: "char",       16: "short",       32: "int",        64: "long long"}
BITS_TO_ZERO_LITERAL = {8: "(char) 0",   16: "(short) 0",   32: "0",          64: "0LL"}
BITS_TO_ONE_LITERAL  = {8: "(char) 1",   16: "(short) 1",   32: "1",          64: "1LL"}
BITS_TO_ALL_LITERAL  = {8: "(char) -1",  16: "(short) -1",  32: "-1",         64: "-1LL"}
# BITS_TO_ALL_LITERAL gives all-ones bitmask (~0) for each bit width
BITS_TO_C_DTYPE      = {8: ctypes.c_int8, 16: ctypes.c_int16, 32: ctypes.c_int32, 64: ctypes.c_int64}
BITS_TO_NP_DTYPE     = {8: np.int8,       16: np.int16,        32: np.int32,        64: np.int64}


class CompiledTernaryBinaryNet2Inputs(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Sequential,
            device='cpu',
            num_bits=64,
            cpu_compiler='gcc',
            verbose=False,
            gates_used=[],
    ):
        """
        Compiles a trained ternary logic gate network into C code using
        dual-polarity binary encoding for ternary values {-1, 0, 1}.

        Each ternary value is encoded as two binary wires:
            +1 → (high=1, low=0)
            -1 → (high=0, low=1)
             0 → (high=0, low=0)
        invalid → (high=1, low=1) → masked out during GroupSum

        :param model:        trained Sequential model with LogicLayers and GroupSum
        :param device:       'cpu' only
        :param num_bits:     bit-packing width (8, 16, 32, 64)
        :param cpu_compiler: 'gcc' or 'clang'
        :param verbose:      print debug info
        """
        super(CompiledTernaryBinaryNet2Inputs, self).__init__()
        self.model        = model
        self.device       = device
        self.num_bits     = num_bits
        self.cpu_compiler = cpu_compiler
        self.gates_used   = gates_used

        assert cpu_compiler in ["clang", "gcc"]
        assert num_bits in [8, 16, 32, 64]

        if self.model is not None:
            layers = []
            self.num_inputs = None

            # Validate model ends with GroupSum
            assert isinstance(self.model[-1], GroupSum), \
                'Last layer must be GroupSum.'
            self.num_classes = self.model[-1].k
            # num_classes: number of output classes (e.g. 10 for MNIST)

            first = True
            for layer in self.model:
                if isinstance(layer, LogicLayer):
                    if first:
                        self.num_inputs = layer.in_dim
                        # num_inputs: number of input features (e.g. 784 for MNIST)
                        first = False
                    self.num_out_per_class = layer.out_dim // self.num_classes
                    # Store (input_indices_A, input_indices_B, chosen_gate_per_neuron)
                    # weights.argmax(1): for each neuron picks the gate with highest weight
                    layers.append((
                        layer.indices[0],        # [num_neurons] indices for input A
                        layer.indices[1],        # [num_neurons] indices for input B
                        layer.weights.argmax(1)  # [num_neurons] chosen gate index 0-12
                    ))
                elif isinstance(layer, torch.nn.Flatten):
                    if verbose: print('Skipping Flatten.')
                elif isinstance(layer, GroupSum):
                    if verbose: print('Skipping GroupSum.')
                else:
                    assert False, f'Unknown layer: {type(layer)}'

            self.layers = layers
            # self.layers: list of (layer_a, layer_b, layer_op) per LogicLayer

        self.lib_fn = None
        # lib_fn: compiled C function, set after compile()

    def get_gate_code(self, a_high, a_low, b_high, b_low, gate_id):
        """
        Returns (res_high, res_low) C expressions for a ternary gate
        given dual-polarity input wire names.

        a_high, a_low: C variable names for input A's positive and negative wires
        b_high, b_low: C variable names for input B's positive and negative wires
        gate_op: integer gate index 0-12
        """
        z = BITS_TO_ZERO_LITERAL[self.num_bits]   # "0" or "0LL" etc.
        o = BITS_TO_ALL_LITERAL[self.num_bits]     # "-1" (all ones bitmask)

        if gate_id == 0:
            # Constant -1: always output -1 → (high=0, low=all_ones)
            res_high = z
            res_low  = o

        elif gate_id == 1:
            # Output = A: pass A wires through unchanged
            res_high = a_high
            res_low  = a_low

        elif gate_id == 2:
            # Output = B: pass B wires through unchanged
            res_high = b_high
            res_low  = b_low

        elif gate_id == 3:
            # Output = -A: negate A by swapping high and low wires
            res_high = a_low
            res_low  = a_high

        elif gate_id == 4:
            # Output = -B: negate B by swapping high and low wires
            res_high = b_low
            res_low  = b_high

        elif gate_id == 5:
            # Output = A*B:
            n2=f"~({a_high} ^ {b_high})"
            n5=f"~({b_high} ^ {a_low})"
            n0=f"~({b_high} ^ {b_low})"
            n1=f"~({n0})"
            n3=f"~(({n0} | {n2}))"
            n4=f"~(({n1} & {n2}))"
            res_high = f"~(({n5} | {n4}))"
            res_low=f"(({n5} & {n3}))"
            
        elif gate_id == 6:
            # Output = -A*B = -(A*B): negate gate 5 by swapping high/low
            n2=f"~({a_high} ^ {b_high})"
            n5=f"~({b_high} ^ {a_low})"
            n0=f"~({b_high} ^ {b_low})"
            n1=f"~({n0})"
            n3=f"~(({n0} | {n2}))"
            n4=f"~(({n1} & {n2}))"
            res_high=f"(({n5} & {n3}))"
            res_low = f"~(({n5} | {n4}))"

        elif gate_id == 7:
            # Constant 0: always output 0 → (high=0, low=0)
            res_high = z
            res_low  = z

        elif gate_id == 8:
            res_high = f"({a_high} | {a_low})"
            res_low  = z

        elif gate_id == 9:
            res_high = f"({b_high} | {b_low})"
            res_low  = z

        elif gate_id == 10:
            res_high = z
            res_low  = f"({a_high} | {a_low})"

        elif gate_id == 11:
            res_high = z
            res_low  = f"({b_high} | {b_low})"

        elif gate_id == 12:
            # Gate 12: constant 1 → (high=all_ones, low=0)
            res_high = o
            res_low  = z

        elif gate_id == 13:
            res_high = z
            res_low  = f"~({b_high} | {b_low})"

        elif gate_id == 14:
            res_high = z
            res_low  = f"~({a_high} | {a_low})"

        elif gate_id == 15:
            res_high = f"~({b_high} | {b_low})"
            res_low  = z

        elif gate_id == 16:
            res_high = f"~({a_high} | {a_low})"
            res_low  = z

        elif gate_id == 17:
            n0=f"~({a_high})"
            res_high = f"~(({a_low} | {n0}))"
            res_low  = f"(({a_low} & {n0}))"

        elif gate_id == 18:
            n0=f"~({b_high})"
            res_high = f"~(({b_low} | {n0}))"
            res_low  = f"(({b_low} & {n0}))"

        elif gate_id == 19:
            n0=f"~({a_low})"
            res_high = f"~(({a_high} | {n0}))"
            res_low  = f"(({a_high} & {n0}))"

        elif gate_id == 20:
            n0=f"~({b_low})"
            res_high = f"~(({b_high} | {n0}))"
            res_low  = f"(({b_high} & {n0}))"

        elif gate_id == 21:
            n0=f"~({a_high} | {a_low})"
            n1=f"~({b_high} | {b_low})"
            res_high = f"~(({n0} | {n1}))"
            res_low  = z

        elif gate_id == 22:
            n0=f"~({a_high} | {a_low})"
            n1=f"~({b_high} | {b_low})"
            res_high = z
            res_low  = f"~(({n0} | {n1}))"

        elif gate_id == 23:
            n0=f"~({b_high} ^ {b_low})"
            n1=f"~({a_high} ^ {a_low})"
            n2=f"(({b_high} | {n1}))"
            n3=f"(({a_high} | {n0}))"
            res_high = f"~(({n2} | {b_low}))"
            res_low  = f"~(({n3} | {a_low}))"

        elif gate_id == 24:
            n0=f"~({a_high} ^ {a_low})"
            n1=f"~({b_high} ^ {b_low})"
            n2=f"(({a_high} | {n1}))"
            n3=f"(({b_high} | {n0}))"
            res_high = f"~(({n2} | {a_low}))"
            res_low  = f"~(({n3} | {b_low}))"
        
        elif gate_id == 25:
            n0=f"({b_high} | {a_low})"
            n1=f"~({b_high} ^ {b_low})"
            n2=f"~({a_high} ^ {a_low})"
            n3=f"(({b_low} | {n0}))"
            res_high = f"~(({n2} | {n1}))"
            res_low  = f"~(({n3} | {a_high}))"

        elif gate_id == 26:
            n0=f"({b_high} | {a_low})"
            n1=f"~({b_high} ^ {b_low})"
            n2=f"~({a_high} ^ {a_low})"
            n3=f"(({b_low} | {n0}))"
            res_high = f"~(({n3} | {a_high}))"
            res_low  = f"~(({n2} | {n1}))"
        
        elif gate_id == 27:
            n0=f"~({a_high})"
            res_high = f"~(({a_low} | {n0}))"
            res_low  = z

        elif gate_id == 28:
            n0=f"~({b_high})"
            res_high = f"~(({b_low} | {n0}))"
            res_low  = z

        elif gate_id == 29:
            n0=f"~({a_low})"
            res_high = z
            res_low  = f"~(({a_high} | {n0}))"

        elif gate_id == 30:
            n0=f"~({b_low})"
            res_high = z
            res_low  = f"~(({b_high} | {n0}))"

        elif gate_id == 31:
            n0=f"~({b_high} | {b_low})"
            n1=f"~({a_high} | {a_low})"
            res_high = f"(({n0} & {n1}))"
            res_low  = z

        elif gate_id == 32:
            n0=f"~({a_high})"
            n1=f"~(({n0} & {a_low}))"
            n2=f"~(({n0} | {a_low}))"
            n3=f"~(({b_high} & {n2}))"
            n4=f"~(({b_high} | {n1}))"
            res_high = f"~(({b_low} | {n3}))"
            res_low  = f"(({b_low} & {n4}))"
        
        elif gate_id == 33:
            n0=f"~({a_high})"
            n1=f"~(({n0} | {a_low}))"
            n2=f"~(({n0} & {a_low}))"
            n3=f"~(({b_high} | {n2}))"
            n4=f"~(({b_high} & {n1}))"
            res_high = f"(({b_low} & {n3}))"
            res_low  = f"~(({b_low} | {n4}))"
        
        elif gate_id == 34:
            n0=f"({b_low} | {a_low})"
            n1=f"~({b_high} | {a_high})"
            res_high = f"~(({n1} | {n0}))"
            res_low  = f"(({n1} & {n0}))"
        
        elif gate_id == 35:
            n0=f"({b_high} | {a_high})"
            n1=f"~({b_low} | {a_low})"
            res_high = f"~(({n1} | {n0}))"
            res_low  = f"(({n1} & {n0}))"

        elif gate_id == 36:
            n0=f"~({b_low})"
            n2=f"~({a_high})"
            n1=f"~(({n0} & {b_high}))"
            n3=f"~(({a_low} & {n2}))"
            n4=f"(({a_low} | {n2}))"
            n6=f"~(({n3} | {b_high}))"
            res_high = f"~(({n1} & {n4}))"
            res_low  = f"(({b_low} & {n6}))"
        
        elif gate_id == 37:
            n0=f"~({b_low})"
            n2=f"~({a_high})"
            n1=f"~(({n0} & {b_high}))"
            n3=f"~(({a_low} & {n2}))"
            n4=f"(({a_low} | {n2}))"
            n6=f"~(({n3} | {b_high}))"
            res_high = f"(({b_low} & {n6}))"
            res_low  = f"~(({n1} & {n4}))"

        elif gate_id == 38:
            n0=f"~({b_high})"
            n2=f"~({a_high})"
            n1=f"~(({n0} & {b_low}))"
            n3=f"~(({a_low} | {n2}))"
            n4=f"~(({a_low} & {n2}))"
            n6=f"~(({n3} & {b_high}))"
            res_high = f"~(({b_low} | {n6}))"
            res_low  = f"~(({n1} & {n4}))"
        
        elif gate_id == 39:
            n0=f"~({b_high})"
            n2=f"~({a_high})"
            n1=f"~(({n0} & {b_low}))"
            n3=f"~(({a_low} | {n2}))"
            n4=f"~(({a_low} & {n2}))"
            n6=f"~(({n3} & {b_high}))"
            res_high = f"~(({n1} & {n4}))"
            res_low  = f"~(({b_low} | {n6}))"

        elif gate_id == 40:
            n0=f"~({a_low} & {b_low})"
            n1=f"({a_high} ^ {b_high})"
            n2=f"(({n0} | {a_high}))"
            n3=f"(({a_low} | {n1}))"
            n4=f"~(({b_high} | {n2}))"
            n5=f"~(({n3} | {b_low}))"
            res_low  = f"~(({n4} | {n5}))"
            res_high = f"~(({res_low}))"
        
        elif gate_id == 41:
            n0=f"~({a_low} & {b_low})"
            n1=f"({a_high} ^ {b_high})"
            n2=f"(({n0} | {a_high}))"
            n3=f"(({a_low} | {n1}))"
            n4=f"~(({b_high} | {n2}))"
            n5=f"~(({n3} | {b_low}))"
            res_high  = f"~(({n4} | {n5}))"
            res_low = f"~(({res_high}))"

        elif gate_id == 42:
            n0=f"~({b_low})"
            n2=f"~(({b_high} | {n0}))"
            n1=f"~({a_low} ^ {a_high})"
            n3=f"~(({n2}))"
            n4=f"~(({a_high} & {n2}))"
            n5=f"~(({b_high} & {n1}))"
            n6=f"~(({a_high} | {n3}))"
            n8=f"(({b_low} | {n5}))"
            n9=f"~(({a_low} & {n6}))"
            res_high  = f"~(({n8} & {n9}))"
            res_low = f"~(({n4} | {a_low}))"
        
        elif gate_id == 43:
            n0=f"~({b_low})"
            n2=f"~(({b_high} | {n0}))"
            n1=f"~({a_low} ^ {a_high})"
            n3=f"~(({n2}))"
            n4=f"~(({a_high} & {n2}))"
            n5=f"~(({b_high} & {n1}))"
            n6=f"~(({a_high} | {n3}))"
            n8=f"(({b_low} | {n5}))"
            n9=f"~(({a_low} & {n6}))"
            res_high  =  f"~(({n4} | {a_low}))"
            res_low = f"~(({n8} & {n9}))"

        elif gate_id == 44:
            n0=f"~({b_high})"
            n2=f"~(({b_low} | {n0}))"
            n1=f"~({a_low} ^ {a_high})"
            n3=f"~(({n2}))"
            n4=f"~(({a_high} & {n2}))"
            n5=f"~(({b_high} | {n1}))"
            n6=f"~(({a_high} | {n3}))"
            n7=f"(({a_low} | {n4}))"
            n8=f"~(({b_low} & {n5}))"
            res_high  = f"~(({n8} & {n7}))"
            res_low = f"(({n6} & {a_low}))"
        
        elif gate_id == 45:
            n0=f"~({b_high})"
            n2=f"~(({b_low} | {n0}))"
            n1=f"~({a_low} ^ {a_high})"
            n3=f"~(({n2}))"
            n4=f"~(({a_high} & {n2}))"
            n5=f"~(({b_high} | {n1}))"
            n6=f"~(({a_high} | {n3}))"
            n7=f"(({a_low} | {n4}))"
            n8=f"~(({b_low} & {n5}))"
            res_high  = f"(({n6} & {a_low}))"
            res_low = f"~(({n8} & {n7}))"
        
        else:
            raise ValueError(f"Unknown gate_id: {gate_id}")


        # Cast to correct type to prevent C integer promotion bugs
        if self.num_bits == 8:
            res_high = f"(char) ({res_high})"
            res_low  = f"(char) ({res_low})"
        elif self.num_bits == 16:
            res_high = f"(short) ({res_high})"
            res_low  = f"(short) ({res_low})"

        return res_high, res_low

    def get_layer_code(self, layer_a, layer_b, layer_op, layer_id, prefix_sums):
        """
        Generates C code for one LogicLayer.
        Each neuron i produces 2 output wires (high, low).
        Input wires are indexed as 2*neuron_idx and 2*neuron_idx+1.
        """
        code = []
        for var_id, (gate_a, gate_b, gate_op) in enumerate(zip(layer_a, layer_b, layer_op)):

            # Compute wire names for inputs A and B
            # Each neuron index maps to 2 consecutive wires: high=2*idx, low=2*idx+1
            if layer_id == 0:
                # First layer reads directly from packed input array
                a_high = f"inp[{2 * gate_a}]"
                a_low  = f"inp[{2 * gate_a + 1}]"
                b_high = f"inp[{2 * gate_b}]"
                b_low  = f"inp[{2 * gate_b + 1}]"
            else:
                # Hidden layers read from previous layer's output variables
                # prefix_sums[layer_id-1] is the offset of previous layer's variables
                # Multiply by 2 because each neuron has 2 wires
                base = prefix_sums[layer_id - 1]
                a_high = f"v{base + 2 * gate_a}"
                a_low  = f"v{base + 2 * gate_a + 1}"
                b_high = f"v{base + 2 * gate_b}"
                b_low  = f"v{base + 2 * gate_b + 1}"

            # Get C expressions for this gate's output wires
            gate_id = int(self.gates_used[gate_op])
            res_high, res_low = self.get_gate_code(a_high, a_low, b_high, b_low, gate_id)

            if self.device == 'cpu' and layer_id == len(prefix_sums) - 1:
                # Last layer writes directly to output array
                code.append(f"\tout[{2 * var_id}]     = {res_high};")
                code.append(f"\tout[{2 * var_id + 1}] = {res_low};")
            else:
                # Hidden layers declare local const variables
                # Each neuron produces 2 variables: v{offset} (high) and v{offset+1} (low)
                base_out = prefix_sums[layer_id]
                code.append(f"\tconst {BITS_TO_DTYPE[self.num_bits]} v{base_out + 2*var_id}     = {res_high};")
                code.append(f"\tconst {BITS_TO_DTYPE[self.num_bits]} v{base_out + 2*var_id + 1} = {res_low};")

        return code

    def get_c_code(self):
        """
        Generates the full C source code for the compiled ternary network.
        """
        # Build prefix sums for variable index offsets
        # Each layer contributes 2 * num_neurons variables (high + low per neuron)
        prefix_sums = [0]
        cur_count = 0
        for layer_a, _, _ in self.layers[:-1]:
            cur_count += 2 * len(layer_a)
            # 2* because each neuron has 2 wires
            prefix_sums.append(cur_count)

        code = [
            "#include <stddef.h>",
            "#include <stdlib.h>",
            "#include <stdbool.h>",
            "",
            # Main logic function: takes packed input, writes packed output
            # Input:  inp[num_inputs] packed ternary (2 wires per feature)
            # Output: out[num_neurons_last_layer * 2] packed ternary
            f"void logic_gate_net({BITS_TO_DTYPE[self.num_bits]} const *inp, {BITS_TO_DTYPE[self.num_bits]} *out) {{",
        ]

        # Generate code for each layer
        for layer_id, (layer_a, layer_b, layer_op) in enumerate(self.layers):
            code.extend(self.get_layer_code(layer_a, layer_b, layer_op, layer_id, prefix_sums))

        code.append("}")

        # Last layer has num_neurons outputs, each with 2 wires = 2*num_neurons total
        num_neurons_ll = self.layers[-1][0].shape[0]
        # neurons_per_class: how many neurons each class sums over
        neurons_per_class = num_neurons_ll // self.num_classes
        # Bits needed to represent the count of positive or negative votes
        # +1 to handle overflow when all neurons vote the same way
        log2_bits = math.ceil(math.log2(neurons_per_class + 1))

        code.append(fr"""
void apply_logic_gate_net(bool const *inp, {BITS_TO_DTYPE[32]} *out, size_t len) {{

    // inp_temp: packed input with 2 wires per feature
    // 2 * num_inputs entries, each holding num_bits packed samples
    {BITS_TO_DTYPE[self.num_bits]} *inp_temp = malloc(2 * {self.num_inputs} * sizeof({BITS_TO_DTYPE[self.num_bits]}));

    // out_temp: packed last layer output with 2 wires per neuron
    {BITS_TO_DTYPE[self.num_bits]} *out_temp = malloc(2 * {num_neurons_ll} * sizeof({BITS_TO_DTYPE[self.num_bits]}));

    // pos_accum, neg_accum: ripple-carry accumulators for positive and negative votes
    {BITS_TO_DTYPE[self.num_bits]} *pos_accum = malloc({log2_bits} * sizeof({BITS_TO_DTYPE[self.num_bits]}));
    {BITS_TO_DTYPE[self.num_bits]} *neg_accum = malloc({log2_bits} * sizeof({BITS_TO_DTYPE[self.num_bits]}));

    // Process one batch of num_bits samples at a time
    for(size_t i = 0; i < len; ++i) {{

        // Pack ternary inputs into dual-polarity binary format
        // For each input feature d, pack num_bits samples into two words:
        //   inp_temp[2*d]:   high wire (sample is +1)
        //   inp_temp[2*d+1]: low wire  (sample is -1)
        for(size_t d = 0; d < {self.num_inputs}; ++d) {{
            {BITS_TO_DTYPE[self.num_bits]} res_high = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            {BITS_TO_DTYPE[self.num_bits]} res_low  = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                // sample b in batch i, feature d
                // dual input layout: [sample, 2*num_inputs]
                // high wire at position 2*d, low wire at 2*d+1
                size_t sample_idx = i * {self.num_bits} + ({self.num_bits} - b - 1);
                size_t high_idx   = sample_idx * 2 * {self.num_inputs} + 2 * d;
                size_t low_idx    = high_idx + 1;
                res_high = (res_high << 1) | !!(inp[high_idx]);
                res_low  = (res_low  << 1) | !!(inp[low_idx]);
            }}
            inp_temp[2 * d]     = res_high;
            inp_temp[2 * d + 1] = res_low;
        }}

        // Run all logic layers
        logic_gate_net(inp_temp, out_temp);

        // GroupSum: for each class c, sum ternary votes from its neurons
        for(size_t c = 0; c < {self.num_classes}; ++c) {{

            // Clear accumulators
            for(size_t d = 0; d < {log2_bits}; ++d) {{
                pos_accum[d] = {BITS_TO_ZERO_LITERAL[self.num_bits]};
                neg_accum[d] = {BITS_TO_ZERO_LITERAL[self.num_bits]};
            }}

            // Sum over neurons belonging to class c
            // Each neuron has 2 wires: high (+1 votes) and low (-1 votes)
            for(size_t a = 0; a < {neurons_per_class}; ++a) {{
                size_t base = (c * {neurons_per_class} + a) * 2;
                // p_wire: bits where this neuron voted +1
                {BITS_TO_DTYPE[self.num_bits]} p_wire = out_temp[base];
                // n_wire: bits where this neuron voted -1
                {BITS_TO_DTYPE[self.num_bits]} n_wire = out_temp[base + 1];

                // Mask out invalid states where both wires are 1
                // valid_mask has 0 where both p and n are 1 (impossible in valid ternary)
                {BITS_TO_DTYPE[self.num_bits]} valid_mask = ~(p_wire & n_wire);
                p_wire &= valid_mask;
                n_wire &= valid_mask;

                // Ripple-carry addition for positive votes
                // Adds p_wire into pos_accum bit-serially
                {BITS_TO_DTYPE[self.num_bits]} carry = p_wire;
                for(int d = {log2_bits} - 1; d >= 0; --d) {{
                    {BITS_TO_DTYPE[self.num_bits]} tmp = pos_accum[d];
                    pos_accum[d] = carry ^ tmp;
                    carry        = carry & tmp;
                }}

                // Ripple-carry addition for negative votes
                carry = n_wire;
                for(int d = {log2_bits} - 1; d >= 0; --d) {{
                    {BITS_TO_DTYPE[self.num_bits]} tmp = neg_accum[d];
                    neg_accum[d] = carry ^ tmp;
                    carry        = carry & tmp;
                }}
            }}

            // Unpack accumulated counts back to per-sample integers
            // For each sample b in this batch, extract its positive and negative vote counts
            for(size_t b = 0; b < {self.num_bits}; ++b) {{
                const {BITS_TO_DTYPE[self.num_bits]} bit_mask = {BITS_TO_ONE_LITERAL[self.num_bits]} << b;
                {BITS_TO_DTYPE[32]} total_pos = 0;
                {BITS_TO_DTYPE[32]} total_neg = 0;

                // Reconstruct binary number from accumulator bits
                for(size_t d = 0; d < {log2_bits}; ++d) {{
                    total_pos = (total_pos << 1) + !!(pos_accum[d] & bit_mask);
                    total_neg = (total_neg << 1) + !!(neg_accum[d] & bit_mask);
                }}

                // Net score = positive votes - negative votes
                // This is equivalent to GroupSum over ternary values
                out[(i * {self.num_bits} + b) * {self.num_classes} + c] = total_pos - total_neg;
            }}
        }}
    }}

    free(inp_temp);
    free(out_temp);
    free(pos_accum);
    free(neg_accum);
}}
""")
        return "\n".join(code)

    def compile(self, opt_level=1, save_lib_path=None, verbose=False):
        """Compiles the C code to a shared library."""
        with tempfile.NamedTemporaryFile(suffix=".so") as lib_file:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".c") as c_file:
                code = self.get_c_code()

                if verbose and len(code.split('\n')) <= 200:
                    print("\n\n" + code + "\n\n")

                c_file.write(code)
                c_file.flush()

                if verbose:
                    print(f'C code: {len(code.split(chr(10)))} lines')

                t_s = time.time()
                compiler_out = subprocess.run([
                    self.cpu_compiler, "-shared", "-fPIC",
                    f"-O{opt_level}", "-o", lib_file.name, c_file.name
                ])

                if compiler_out.returncode != 0:
                    raise RuntimeError(f'Compilation failed with code {compiler_out.returncode}')

                print(f'Compiled in {time.time()-t_s:.3f}s')

            if save_lib_path is not None:
                shutil.copy(lib_file.name, save_lib_path)

            lib    = ctypes.cdll.LoadLibrary(lib_file.name)
            lib_fn = lib.apply_logic_gate_net
            lib_fn.restype  = None
            # Input:  bool array (dual-polarity packed ternary)
            # Output: int32 array (class scores)
            # Size:   number of batches
            lib_fn.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_bool,  flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(BITS_TO_C_DTYPE[32], flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
            ]

        self.lib_fn = lib_fn

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        x: float tensor of shape (batch_size, num_inputs) with values in {-1, 0, 1}
            returns: int tensor of shape (batch_size, num_classes)
        """
        x_np = x.numpy()
        batch_size = x_np.shape[0]

        # high: True where value is +1
        high = (x_np > 0.5).astype(bool)
        # low:  True where value is -1
        low  = (x_np < -0.5).astype(bool)

        # Interleave high and low per feature
        # Layout: [batch_size, 2 * num_inputs]
        # Position 2*d   → high wire for feature d
        # Position 2*d+1 → low  wire for feature d
        dual = np.empty((batch_size, 2 * x_np.shape[1]), dtype=bool)
        dual[:, 0::2] = high  # even: high wires
        dual[:, 1::2] = low   # odd:  low wires

        # Pad batch to multiple of num_bits
        batch_size_div_bits = math.ceil(batch_size / self.num_bits)
        pad_len = batch_size_div_bits * self.num_bits - batch_size
        if pad_len > 0:
            dual = np.concatenate([
                dual,
                np.zeros((pad_len, dual.shape[1]), dtype=bool)
            ])

        if verbose:
            print('dual.shape', dual.shape)

        # Flatten to 1D: C code reads as [sample * 2 * num_inputs + 2*d] for high
        #                                 [sample * 2 * num_inputs + 2*d + 1] for low
        dual_flat = dual.reshape(-1)

        out = np.zeros(
            batch_size_div_bits * self.num_bits * self.num_classes,
            dtype=BITS_TO_NP_DTYPE[32]
        )

        self.lib_fn(dual_flat, out, batch_size_div_bits)

        out = torch.tensor(out).view(
            batch_size_div_bits * self.num_bits,
            self.num_classes
        )
        if pad_len > 0:
            out = out[:-pad_len]

        if verbose:
            print('out.shape', out.shape)

        return out