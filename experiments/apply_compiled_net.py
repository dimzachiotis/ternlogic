import torch
import torchvision

import mnist_dataset
from difflogic import CompiledLogicNet
#Imports CompiledLogicNet from the difflogic package. This is not a standard neural network. 
# It represents a logic-based model that has been compiled into native code (C/C++ → .so file).

torch.set_num_threads(1)
#Forces PyTorch to use only one CPU thread. Makes timing reproducible and avoids overhead from multithreading.

dataset = 'mnist20x20'
batch_size = 1_000
#Sets the number of samples processed at once (1,000 images per batch).

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: torch.round(x*2)/2),
])
#Creates a pipeline of image transformations. 
# Converts a PIL image or NumPy array into a PyTorch tensor and also scales pixel values from [0, 255] → [0.0, 1.0].
#Applies x.round() to the tensor.This binarizes the image, rounds every pixel to either 0.0, 0.5 or 1.0 
#This is important because Logic networks operate on Boolean inputs, not real numbers

#creates test set
test_set = mnist_dataset.MNIST('./data-mnist', train=False, transform=transforms, remove_border=True)
#test_set behaves like a list: test_set[i] → (image, label)
# Each image is a 20×20 binary tensor with values of 0 or 1

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
#Wraps the dataset to load data in batches, handle shuffling and optimize memory usage.
#test_set=Dataset from were to load the data set, sets batch size to batch_size, keeps test data in fixed order(Important for 
#reproducibility - shuffling isn’t needed during evaluation), pin_memory=True (Helps speed up CPU → GPU transfer. 
# Even on CPU, it often improves performance), drop_last=True drops the last batch if it’s smaller than batch_size.

#only logic bit-widths 64 is active. Tradeoff between speed and expressiveness
for num_bits in [
    # 8,
    # 16,
    # 32,
    64
]:
    save_lib_path = 'lib/{:08d}_{}.so'.format(0, num_bits)
    #Builds the path to a compiled shared library.
    compiled_model = CompiledLogicNet.load(save_lib_path, 10, num_bits)
    #Loads the compiled shared library. path,10=Number of output classes, num_bits = Bit width used internally.

    #evaluation loop
    correct, total = 0, 0
    for (data, labels) in test_loader:
        data = torch.nn.Flatten()(data).bool().numpy()
        #flattens each image (batch_size, 1, 20, 20)→(batch_size, 400)

        output = compiled_model.forward(data)

        correct += (output.argmax(-1) == labels).float().sum()
        #Finds the predicted class index and compares predictions to true labels, producing a Boolean tensor. 
        # Converts the to floats and counts correct predictions in the batch.
        total += output.shape[0]
        #Adds number of samples in the batch.

    acc3 = correct / total
    #computes accuracy
    print('COMPILED MODEL', num_bits, acc3)
