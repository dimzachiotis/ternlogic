#This script loads tabular or image datasets, converts data into binary logic features, 
#trains a differentiable logic gate network, evaluates in multiple modes and optionally 
#compiles the trained model to optimized C code

import argparse
import math
import random
import os
import sys
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import json

from results_json import ResultsJSON
#Store experiment results in JSON files

#Custom dataset loaders
import mnist_dataset
import uci_datasets
#Custom logic-based neural network components

top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if top_level_dir not in sys.path:
    sys.path.insert(0, top_level_dir)
#it doesnt see difflogic.py file, so this fixes it

from difflogic.difflogic import LogicLayer, GroupSum
from difflogic.packbitstensor import PackBitsTensor
from difflogic.compiled_model import CompiledLogicNet
from difflogic.compiled_ternary_model_python import CompiledTernaryPython
from difflogic.compiled_binary_model_python import CompiledBinaryPython

device ='cpu' if  not torch.cuda.is_available() else 'cuda'
#if no cuda available, then use cpu

#Forces PyTorch to use one CPU thread
torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}
#Maps bit precision → PyTorch dtype. Used to train with different numerical precision

#Dataset loading function.
#Loads datasets and returns:train_loader, validation_loader and test_loader
def load_dataset(args):
    validation_loader = None
    #Default: no validation set unless explicitly created

    #Adult dataset
    if args.dataset == 'adult':
        train_set = uci_datasets.AdultDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.AdultDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    #Breast cancer dataset
    elif args.dataset == 'breast_cancer':
        train_set = uci_datasets.BreastCancerDataset('./data-uci', split='train', download=True, with_val=False)
        test_set = uci_datasets.BreastCancerDataset('./data-uci', split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    #MONK datasets
    elif args.dataset.startswith('monk'):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset('./data-uci', style, split='train', download=True, with_val=False)
        test_set = uci_datasets.MONKsDataset('./data-uci', style, split='test', with_val=False)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(1e6), shuffle=False)
    #MNIST datasets
    elif args.dataset in ['mnist', 'mnist20x20']:
        train_set = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=args.dataset == 'mnist20x20')
        test_set = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=args.dataset == 'mnist20x20')
        #removes border pixels if args.dataset == 'mnist20x20', else it keeps the origina 28x28
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        #Computes training set size after validation split
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
        #Randomly splits dataset

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        #Drops last incomplete batch, ensuring consistent batch size

    #CIFAR-10 datasets
    elif 'cifar-10' in args.dataset:
        transform = {
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0),
        }[args.dataset]
        #if args.dataset=cifar-10-3-thresholds, creates 3 binary masks: x > 0.25, x > 0.50, x > 0.75 so one float pixel becomes a tensor of 3 binary values -> shape change
        # original: 3 × 32 × 32, after: (3×3) × 32 × 32 = 9 × 32 × 32
        #if args.dataset=cifar-10-31-thresholds, Same idea but 31 masks, much higher dimensional input

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(transform),
        ])
        train_set = torchvision.datasets.CIFAR10('./data-cifar', train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.CIFAR10('./data-cifar', train=False, transform=transforms)

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    else:
        raise NotImplementedError(f'The data set {args.dataset} is not supported!')

    return train_loader, validation_loader, test_loader

#Yields exactly n batches even if dataset is smaller (Recycles data loader if needed)
def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break

#Returns input feature size (flattened tensor)
def input_dim_of_dataset(dataset):
    return {
        'adult': 116,
        'breast_cancer': 51,
        'monk1': 17,
        'monk2': 17,
        'monk3': 17,
        'mnist': 784,
        'mnist20x20': 400,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }[dataset]

#Returns number of output classes
def num_classes_of_dataset(dataset):
    return {
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }[dataset]

#Model creation. Builds model, loss function, optimizer.
def get_model(args):
    llkw = dict(grad_factor=args.grad_factor, connections=args.connections)
    #Common keyword arguments for logic layers

    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)
    #Determines input/output sizes

    logic_layers = []
    #Stores layers before final classifier

    arch = args.architecture
    k = args.num_neurons
    #number of neurons/gates per layer
    l = args.num_layers
    #Model hyperparameters

    ####################################################################################################################

    if arch == 'randomly_connected':
        logic_layers.append(torch.nn.Flatten())
        #Flattens input and appends it to list
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        #First logic layer

        #Adds hidden layers
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

        #Wraps layers into a single model
        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )
        #GroupSum is the output layer with class_count classes and t=args.tau

    ####################################################################################################################

    else:
        raise NotImplementedError(arch)

    ####################################################################################################################
    #Counts logic neurons (excluding Flatten & GroupSum)
    total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
    print(f'total_num_neurons={total_num_neurons}')
    #Counts logic connections
    total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
    print(f'total_num_weights={total_num_weights}')
    if args.experiment_id is not None:
        results.store_results({
            'total_num_neurons': total_num_neurons,
            'total_num_weights': total_num_weights,
        })

    model = model.to(device)
    #Moves model to device

    print(model)
    if args.experiment_id is not None:
        results.store_results({'model_str': str(model)})

    loss_fn = torch.nn.CrossEntropyLoss()
    #Cross entropy Loss function

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #Adam optimizer

    return model, loss_fn, optimizer

#Training Function
def train(model, x, y, loss_fn, optimizer):
    x = model(x)
    #Forward pass
    loss = loss_fn(x, y)
    #Compute loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #Backpropagation and parameter update

    return loss.item()
    #returns a scalar

def eval(model, loader, mode):
    orig_mode = model.training
    #Stores original train/eval mode
    with torch.no_grad():
        model.train(mode=mode)
        #Allows evaluating in training mode or eval mode
        res = np.mean(
            [
                (model(x.to(device).round()).argmax(-1) == y.to(device)).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        #Mean accuracy across batches
        model.train(mode=orig_mode)
        #Restores original mode
    return res.item()


def packbits_eval(model, loader):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (model(PackBitsTensor(x.to(device).reshape(x.shape[0], -1).round().bool())).argmax(-1) == y.to(
                    device)).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()

#Main script - Example usage
#IMPORTANT
#When Python starts executing a file, it automatically creates several special variables, including:
#__name__, __file__,__package__, __loader__ .
#Python assigns __name__ before any line of your code runs.
#if you run the file directly python main.py
#Python sets: __name__ = "__main__"
if __name__ == '__main__':

    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--dataset', type=str, choices=[
        'adult', 'breast_cancer',
        'monk1', 'monk2', 'monk3',
        'mnist', 'mnist20x20',
        'cifar-10-3-thresholds',
        'cifar-10-31-thresholds',
    ], required=True, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                                     'additional eval step.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    parser.add_argument('--architecture', '-a', type=str, default='randomly_connected')
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)

    parser.add_argument('--grad-factor', type=float, default=1.)

    args = parser.parse_args()
    #creates arg object
    ####################################################################################################################

    print(vars(args))

    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        #Creates experiment log file
        results.store_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    #keeping same seed number, it ensures reproducibility
    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args)
    #Load data & model
    ####################################################################################################################

    best_acc = 0
    #Iterates exactly num_iterations batches
    for i, (x, y) in tqdm(
            enumerate(load_n(train_loader, args.num_iterations)),
            desc='iteration',
            total=args.num_iterations,
    ):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(device)
        y = y.to(device)
        #Moves data to device

        loss = train(model, x, y, loss_fn, optim)
        #Performs one optimization step

        #Evaluates: Train accuracy, Validation accuracy (optional), Test accuracy. Both train mode and eval mode
        if (i+1) % args.eval_freq == 0:
            if args.extensive_eval:
                train_accuracy_train_mode = eval(model, train_loader, mode=True)
                valid_accuracy_eval_mode = eval(model, validation_loader, mode=False)
                valid_accuracy_train_mode = eval(model, validation_loader, mode=True)
            else:
                train_accuracy_train_mode = -1
                valid_accuracy_eval_mode = -1
                valid_accuracy_train_mode = -1
            train_accuracy_eval_mode = eval(model, train_loader, mode=False)
            test_accuracy_eval_mode = eval(model, test_loader, mode=False)
            test_accuracy_train_mode = eval(model, test_loader, mode=True)

            r = {
                'train_acc_eval_mode': train_accuracy_eval_mode,
                'train_acc_train_mode': train_accuracy_train_mode,
                'valid_acc_eval_mode': valid_accuracy_eval_mode,
                'valid_acc_train_mode': valid_accuracy_train_mode,
                'test_acc_eval_mode': test_accuracy_eval_mode,
                'test_acc_train_mode': test_accuracy_train_mode,
            }

            if args.packbits_eval:
                r['train_acc_eval'] = packbits_eval(model, train_loader)
                r['valid_acc_eval'] = packbits_eval(model, train_loader)
                r['test_acc_eval'] = packbits_eval(model, test_loader)

            if args.experiment_id is not None:
                results.store_results(r)
            else:
                print(r)

            #Tracks best validation performance
            if valid_accuracy_eval_mode > best_acc:
                best_acc = valid_accuracy_eval_mode
                #Saves best model statistics
                if args.experiment_id is not None:
                    results.store_final_results(r)
                else:
                    print('IS THE BEST UNTIL NOW.')
            #Writes JSON file to disk
            if args.experiment_id is not None:
                results.save()
####################################################################################################################
    #Store Weights
    import json

    if args.experiment_id is not None:
    # Ensure results folder exists
        os.makedirs('./results', exist_ok=True)

        layer_weights = []
        for i, layer in enumerate(model):
            if isinstance(layer, LogicLayer):
                # Access the neuron weights (or connections, depending on your LogicLayer implementation)
                weights = layer.weights.detach().cpu()
                layer_weights.append(weights)
                print(f"Layer {i}: weights shape={weights.shape}, mean={weights.mean():.4f}, std={weights.std():.4f}")

        # Save as JSON file (human-readable, can open in any text editor)
        json_filename = f"./results/trained_weights_{args.experiment_id}.json"
        weights_json = {f"layer_{i}": w.tolist() for i, w in enumerate(layer_weights)}
        with open(json_filename, "w") as f:
            json.dump(weights_json, f)
        print(f"Neuron weights saved as JSON: {json_filename}")

    ####################################################################################################################
    
    #Ternary Model Python Compilation (Optional)
    if args.compile_model:
        print('\n' + '='*80)
        print(' Compiling model with Python...')
        print('='*80)
        for num_bits in [
                # 16,
                # 32,
                64
            ]:
                #Creates a CompiledPython object
                compiled_model = CompiledTernaryPython(
                model=model,
                verbose=False,
                num_bits=num_bits
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        #flattens the input tensor to 1D per sample and converts the data to boolean values (0 or 1). shape[batch size,product of dimesions of data]
                        data = torch.nn.Flatten()(data).bool()
                        #Returns predictions as outputs shape[batch size,number of classes]
                        output = compiled_model.forward(data)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]
                #Accuracy of compiled model
                python_acc = correct / total
                print('COMPILED PYTHON MODEL', num_bits , python_acc)

                #Store Accuracy of python compilation
                if args.experiment_id is not None:
                    # Ensure results folder exists
                    os.makedirs('./results', exist_ok=True)

                    # Prepare filename
                    json_filename = f"./results/{args.experiment_id}_{num_bits}_python.json"

                    # If python_acc is a tensor, convert it to float
                    if isinstance(python_acc, torch.Tensor):
                        python_acc = python_acc.item()  # gets the scalar value as a float

                    # Wrap in dict for JSON
                    acc_data = {'accuracy': python_acc}

                    # Save to JSON
                    with open(json_filename, "w") as f:
                        json.dump(acc_data, f, indent=4)

                    print(f"python_acc saved as JSON: {json_filename}") 
            #Model Compilation (Optional)
####################################################################################################################
    
    #Model C Compilation (Optional)
    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)

        for opt_level in range(4):

            for num_bits in [
                # 8,
                # 16,
                # 32,
                64
            ]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = 'lib/{:08d}_{}.so'.format(
                    args.experiment_id if args.experiment_id is not None else 0, num_bits
                )
                #Converts logic network into pure C code
                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()
                        #Executes compiled C model
                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]
                #Accuracy of compiled model
                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)

    #Store Accuracy of c compilation
    if args.experiment_id is not None:
        # Ensure results folder exists
        os.makedirs('./results', exist_ok=True)

        # Prepare filename
        json_filename = f"./results/{args.experiment_id}_c.json"

        # If acc3 is a tensor, convert it to float
        if isinstance(acc3, torch.Tensor):
            acc3 = acc3.item()  # gets the scalar value as a float

        # Wrap in dict for JSON
        acc_data = {'accuracy': acc3}


        # Save to JSON
        with open(json_filename, "w") as f:
            json.dump(acc_data, f, indent=4)

        print(f"Accuracy saved as JSON: {json_filename}")        
