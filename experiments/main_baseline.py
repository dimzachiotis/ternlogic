#This .py file implements a fully configurable PyTorch training pipeline that:
#loads multiple datasets, builds a fully connected neural network, trains it on GPU,
#evaluates performance periodically,and logs all results reproducibly.
#both cuda and cpu
import argparse
import math
import random

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from results_json import ResultsJSON

import mnist_dataset
import uci_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#if no cuda available, then use cpu

torch.set_num_threads(1)
#Limits PyTorch to 1 CPU thread

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
            'cifar-10-real-input': lambda x: x,
            'cifar-10-3-thresholds': lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0),
            'cifar-10-31-thresholds': lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0),
        }[args.dataset]
        #if args.dataset=cifar-10-real-input, Keeps original pixel values, Input shape: 3 × 32 × 32, Pixel values in [0, 1]
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
        'cifar-10-real-input': 3 * 32 * 32,
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
        'cifar-10-real-input': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }[dataset]

#Model creation. Builds model, loss function, optimizer.
def get_model(args):
    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)
    #Determines input/output sizes

    layers = []
    #Will store PyTorch layers

    arch = args.architecture
    k = args.num_neurons
    #number of neurons per layer
    l = args.num_layers
    #Model hyperparameters

    total_num_neurons = 0

    ####################################################################################################################
    #Fully connected network
    if arch == 'fully_connected':
        layers.append(torch.nn.Flatten())
        #Flattens input and appends it to list
        layers.append(torch.nn.Linear(in_dim, k, dtype=BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]))
        #First dense layer
        layers.append(torch.nn.ReLU())
        #Non-linearity
        total_num_neurons += k

        #Adds hidden layers
        for _ in range(l - 2):
            layers.append(torch.nn.Linear(k, k, dtype=BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]))
            layers.append(torch.nn.ReLU())
            total_num_neurons += k

        #Final classification layer
        layers.append(torch.nn.Linear(k, class_count, dtype=BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]))
        total_num_neurons += class_count
        #???  reviouly it said total_num_neurons += 10
        model = torch.nn.Sequential(*layers)
        #Wraps layers into a single model

    ####################################################################################################################

    else:
        raise NotImplementedError(arch)

    ####################################################################################################################
    #Counts trainable weights
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'total_num_neurons={total_num_neurons}')
    total_num_weights = count_parameters(model)
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

#Evaluation function
def eval(model, loader, mode):
    orig_mode = model.training
    #Stores original train/eval mode
    with torch.no_grad():
        model.train(mode=mode)
        #Allows evaluating in training mode or eval mode
        res = np.mean(
            [
                (model(
                    x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(device)
                ).argmax(-1) == y.to(device)
                ).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        #Mean accuracy across batches
        model.train(mode=orig_mode)
        #Restores original mode
    return res.item()

#Main script - Example usage
if __name__ == '__main__':

    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--dataset', type=str, choices=[
        'adult', 'breast_cancer',
        'monk1', 'monk2', 'monk3',
        'mnist', 'mnist20x20',
        'cifar-10-real-input',
        'cifar-10-3-thresholds',
        'cifar-10-31-thresholds',
    ], required=True, help='the dataset to use')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument('--architecture', '-a', type=str, default='fully_connected')
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)

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

