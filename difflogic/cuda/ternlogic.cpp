//This file is the bridge between Python (PyTorch) and CUDA/C++ kernels.
//It uses PyBind11 and PyTorch to expose CUDA functions so they can be called directly from Python.
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>


namespace py = pybind11;
//Function Declarations
//These tell the compiler that the implementations exist somewhere else (likely .cu files).
torch::Tensor logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);
torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
);
torch::Tensor logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
);
torch::Tensor logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);
std::tuple<torch::Tensor, int> tensor_packtern_cuda(
    torch::Tensor t
);
torch::Tensor groupternsum(
    torch::Tensor t,   // packed tensor from tensor_packtern_cuda
    int pad_len,       // padding returned by tensor_packtern_cuda
    int k,             // number of neuron groups
    int neurons,       // original number of neurons
    int batch          // original batch size
);

//Module Creation
//This creates a Python module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def( 
        "forward",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return logic_layer_cuda_forward(x, a, b, w);
        },
        "logic layer forward (CUDA)");
    m.def(
        "backward_w", [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor grad_y) {
            return logic_layer_cuda_backward_w(x, a, b, grad_y);
        },
        "logic layer backward w (CUDA)");
    m.def(
        "backward_x",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor grad_y, torch::Tensor given_x_indices_of_y_start, torch::Tensor given_x_indices_of_y) {
            return logic_layer_cuda_backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y);
        },
        "logic layer backward x (CUDA)");
    m.def(
        "eval",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return logic_layer_cuda_eval(x, a, b, w);
        },
        "logic layer eval (CUDA)");
    m.def(
        "tensor_packtern_cuda",
        [](torch::Tensor t) {
            return tensor_packtern_cuda(t); // returns tuple (packed_tensor, pad_len)
        },
        "Pack ternary tensor into 2-bit representation (CUDA)"
    );

    m.def(
        "groupternsum",
        [](torch::Tensor t, int pad_len, int k, int neurons, int batch) {
            return groupternsum(t, pad_len, k, neurons, batch);
        },
        "Sum ternary packed tensor over neuron groups (CUDA)"
    );
}
