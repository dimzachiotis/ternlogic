import os
import math
import torch
from .compiled_ternary_model_python import CompiledTernaryPython
from .difflogic import LogicLayer, GroupSum


class VHDLGenerator:
    def __init__(self, model, device='cpu', num_bits=64, verbose=False, gates_used=[]):
        """
        :param model: trained torch.nn.Sequential model with LogicLayers and GroupSum
        :param device: 'cpu' or 'cuda'
        :param num_bits: bit-packing width
        :param verbose: print debug info
        """
        # Wrap model in CompiledTernaryPython to extract layer structure
        self.compiled = CompiledTernaryPython(
            model=model,
            device=device,
            num_bits=num_bits,
            verbose=verbose
        )

        self.num_inputs  = self.compiled.num_inputs
        # num_inputs: number of input features e.g. 784 for MNIST

        self.num_classes = self.compiled.num_classes
        # num_classes: number of output classes e.g. 10 for MNIST

        self.layers = self.compiled.layers
        # layers: list of (layer_a, layer_b, layer_op) per LogicLayer

        self.last_layer_neurons = len(self.layers[-1][0])
        self.neurons_per_class  = self.last_layer_neurons // self.num_classes
        # neurons_per_class: how many neurons contribute to each class score

        # sum_width: bits needed to represent signed sum in range
        # [-neurons_per_class, +neurons_per_class]
        # +2 for sign bit and overflow guard
        self.sum_width = max(8, math.ceil(math.log2(max(1, self.neurons_per_class) + 1)) + 2)
        self.gates_used = gates_used

    def get_required_gate_files(self):
        """
        Returns list of gate VHDL filenames needed for this network.
        Gate files are assumed to already exist as gate_id_0.vhd, gate_id_1.vhd etc.
        """
        unique_gates = set()
        for _, _, layer_op in self.layers:
            for op in layer_op:
                unique_gates.add(int(self.gates_used[int(op.item())]))

        files = [f"gate_id_{gate_id}.vhd" for gate_id in sorted(unique_gates)]
        print("Required gate files:")
        for f in files:
            print(f"  {f}")
        return files

    def generate_network_vhdl(self, output_path="ternary_network.vhd"):
        """
        Generates top-level VHDL structural file connecting all layers and GroupSum.

        Dual-polarity encoding:
            +1 → '10' (high=1, low=0)
            -1 → '01' (high=0, low=1)
             0 → '00' (high=0, low=0)

        Each input/output signal uses 2 bits per ternary value.
        Gate entities are referenced from your existing gate_id_X.vhd files.
        """
        lines = [
            "library IEEE;",
            "use IEEE.STD_LOGIC_1164.ALL;",
            "use IEEE.NUMERIC_STD.ALL;",
            "",
            "entity ternary_to_binary_network is",
            "    port (",
            # 2 bits per input for dual-polarity encoding
            f"        inputs  : in  std_logic_vector({self.num_inputs * 2 - 1} downto 0);",
            # sum_width bits per class for signed output score
            f"        outputs : out std_logic_vector({self.num_classes * self.sum_width - 1} downto 0)",
            "    );",
            "end ternary_to_binary_network;",
            "",
            "architecture Structural of ternary_to_binary_network is",
            "",
        ]

        # Declare intermediate signal vectors for each layer output
        # Each neuron produces 2 bits (dual-polarity high and low wires)
        for idx, (layer_a, _, _) in enumerate(self.layers):
            num_neurons = len(layer_a)
            lines.append(
                f"    signal layer_{idx}_out : std_logic_vector({num_neurons * 2 - 1} downto 0);"
            )

        lines.extend(["", "begin", ""])

        # Instantiate gate entities layer by layer
        for idx, (layer_a, layer_b, layer_op) in enumerate(self.layers):
            lines.append(f"    -- ===== LAYER {idx} =====")
            num_neurons = len(layer_a)

            for i in range(num_neurons):
                op    = int(self.gates_used[layer_op[i].item()])
                a_idx = int(layer_a[i].item())
                b_idx = int(layer_b[i].item())

                # Source signals for inputs A and B
                # Each neuron index maps to 2 consecutive bits:
                # high wire = bit 2*idx+1, low wire = bit 2*idx
                if idx == 0:
                    # First layer reads from top-level inputs port
                    src_a = f"inputs({a_idx * 2 + 1} downto {a_idx * 2})"
                    src_b = f"inputs({b_idx * 2 + 1} downto {b_idx * 2})"
                else:
                    # Hidden layers read from previous layer's output signal
                    src_a = f"layer_{idx-1}_out({a_idx * 2 + 1} downto {a_idx * 2})"
                    src_b = f"layer_{idx-1}_out({b_idx * 2 + 1} downto {b_idx * 2})"

                # Destination signal slice for this neuron's output
                dst = f"layer_{idx}_out({i * 2 + 1} downto {i * 2})"

                # Instantiate gate entity from your existing gate_id_X.vhd files
                lines.append(f"    gate_l{idx}_n{i} : entity work.gate_id_{op}")
                lines.append(f"        port map (A => {src_a}, B => {src_b}, Y => {dst});")

            lines.append("")

        # GroupSum process
        # For each class c, sums ternary values from its neurons
        # and writes the signed result to the outputs port
        last_idx = len(self.layers) - 1
        lines.extend([
            "    -- ===== GROUPSUM =====",
            f"    process(layer_{last_idx}_out)",
            # Declarative region: Variables must be declared before 'begin'
            f"        variable class_accum : signed({self.sum_width - 1} downto 0);",
            "        variable raw_bits    : std_logic_vector(1 downto 0);",
            f"        variable item_val    : signed({self.sum_width - 1} downto 0);",
            "    begin",
            f"        for c in 0 to {self.num_classes - 1} loop",
            "            class_accum := (others => '0');",
            f"            for n in 0 to {self.neurons_per_class - 1} loop",
            # Extract 2-bit dual-polarity slice for neuron n of class c
            f"                raw_bits := layer_{last_idx}_out(((c * {self.neurons_per_class} + n) * 2 + 1) downto ((c * {self.neurons_per_class} + n) * 2));",
            "                -- Decode dual-polarity encoding:",
            "                -- '10' = +1, '01' = -1, '00' = 0",
            '                if raw_bits = "10" then',
            f"                    item_val := to_signed(1, {self.sum_width});",
            '                elsif raw_bits = "01" then',
            f"                    item_val := to_signed(-1, {self.sum_width});",
            "                else",
            "                    item_val := (others => '0');",
            "                end if;",
            # Accumulate ternary value into class sum
            "                class_accum := class_accum + item_val;",
            "            end loop;",
            # Write signed sum for class c to outputs slice
            f"            outputs((c + 1) * {self.sum_width} - 1 downto c * {self.sum_width}) <= std_logic_vector(class_accum);",
            "        end loop;",
            "    end process;",
            "",
            "end Structural;",
        ])

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Generated: {output_path}")

    def generate_network_no_adder_vhdl(self, output_path="ternary_network.vhd"):
        """
        Generates top-level VHDL structural file connecting all layers.
        GroupSum has been removed; outputs are raw 2-bit dual-polarity signals.
        """
        lines = [
            "library IEEE;",
            "use IEEE.STD_LOGIC_1164.ALL;",
            "",
            "entity ternary_to_binary_network_no_adder is",
            "    port (",
            # 2 bits per input for dual-polarity encoding
            f"        inputs  : in  std_logic_vector({self.num_inputs * 2 - 1} downto 0);",
            # 2 bits per final layer neuron (no accumulation)
            f"        outputs : out std_logic_vector({self.last_layer_neurons * 2 - 1} downto 0)",
            "    );",
            "end ternary_to_binary_network_no_adder;",
            "",
            "architecture Structural of ternary_to_binary_network_no_adder is",
            "",
        ]

        # Declare intermediate signal vectors for each layer output
        for idx, (layer_a, _, _) in enumerate(self.layers):
            num_neurons = len(layer_a)
            lines.append(
                f"    signal layer_{idx}_out : std_logic_vector({num_neurons * 2 - 1} downto 0);"
            )

        lines.extend(["", "begin", ""])

        # Instantiate gate entities layer by layer
        for idx, (layer_a, layer_b, layer_op) in enumerate(self.layers):
            lines.append(f"    -- ===== LAYER {idx} =====")
            num_neurons = len(layer_a)

            for i in range(num_neurons):
                op    = int(self.gates_used[layer_op[i].item()])
                a_idx = int(layer_a[i].item())
                b_idx = int(layer_b[i].item())

                if idx == 0:
                    src_a = f"inputs({a_idx * 2 + 1} downto {a_idx * 2})"
                    src_b = f"inputs({b_idx * 2 + 1} downto {b_idx * 2})"
                else:
                    src_a = f"layer_{idx-1}_out({a_idx * 2 + 1} downto {a_idx * 2})"
                    src_b = f"layer_{idx-1}_out({b_idx * 2 + 1} downto {b_idx * 2})"

                dst = f"layer_{idx}_out({i * 2 + 1} downto {i * 2})"

                lines.append(f"    gate_l{idx}_n{i} : entity work.gate_id_{op}")
                lines.append(f"        port map (A => {src_a}, B => {src_b}, Y => {dst});")

            lines.append("")

        # Directly route the final layer to the outputs port
        last_idx = len(self.layers) - 1
        lines.extend([
            "    -- ===== OUTPUT ASSIGNMENT =====",
            f"    outputs <= layer_{last_idx}_out;",
            "",
            "end Structural;",
        ])

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Generated: {output_path}")

    def generate_adder_vhdl(self, output_path="ternary_adder.vhd"):
        """
        Generates a standalone GroupSum adder module.
        Takes raw 2-bit dual-polarity signals and outputs packed signed binary scores.
        """
        lines = [
            "library IEEE;",
            "use IEEE.STD_LOGIC_1164.ALL;",
            "use IEEE.NUMERIC_STD.ALL;",
            "",
            "entity ternary_groupsum_adder is",
            "    port (",
            f"        ternary_inputs : in  std_logic_vector({self.last_layer_neurons * 2 - 1} downto 0);",
            f"        binary_outputs : out std_logic_vector({self.num_classes * self.sum_width - 1} downto 0)",
            "    );",
            "end ternary_groupsum_adder;",
            "",
            "architecture Behavioral of ternary_groupsum_adder is",
            "begin",
            "",
            "    process(ternary_inputs)",
            f"        variable class_accum : signed({self.sum_width - 1} downto 0);",
            "        variable raw_bits    : std_logic_vector(1 downto 0);",
            f"        variable item_val    : signed({self.sum_width - 1} downto 0);",
            "    begin",
            f"        for c in 0 to {self.num_classes - 1} loop",
            "            class_accum := (others => '0');",
            f"            for n in 0 to {self.neurons_per_class - 1} loop",
            # Extract 2-bit dual-polarity slice for neuron n of class c
            f"                raw_bits := ternary_inputs(((c * {self.neurons_per_class} + n) * 2 + 1) downto ((c * {self.neurons_per_class} + n) * 2));",
            "                ",
            "                -- Decode dual-polarity encoding:",
            '                if raw_bits = "10" then',
            f"                    item_val := to_signed(1, {self.sum_width});",
            '                elsif raw_bits = "01" then',
            f"                    item_val := to_signed(-1, {self.sum_width});",
            "                else",
            "                    item_val := (others => '0');",
            "                end if;",
            "                ",
            "                class_accum := class_accum + item_val;",
            "            end loop;",
            "            ",
            # Write out accumulated signed value
            f"            binary_outputs((c + 1) * {self.sum_width} - 1 downto c * {self.sum_width}) <= std_logic_vector(class_accum);",
            "        end loop;",
            "    end process;",
            "",
            "end Behavioral;",
        ]

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Generated Standalone Adder: {output_path}")

    def generate_all(self, output_dir=".",file_name="ternary_network"):
        """
        Generates the top-level network VHDL file and prints
        the list of required gate files that must already exist.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Print required gate files
        self.get_required_gate_files()

        # Generate top-level network
        self.generate_network_vhdl(
            os.path.join(output_dir, f"network_{file_name}.vhd")
        )
        self.generate_network_no_adder_vhdl(
            os.path.join(output_dir, f"network_no_adder_{file_name}.vhd")
        )
        self.generate_adder_vhdl(
            os.path.join(output_dir, f"network_adder_{file_name}.vhd")
        )