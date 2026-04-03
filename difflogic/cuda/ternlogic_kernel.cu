#include <torch/extension.h>
//Includes PyTorch C++ API for building extensions

#include <c10/util/Half.h>
//Provides PyTorch half-precision utilities.
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//CUDA driver/runtime APIs for GPU programming.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <sm_32_atomic_functions.h>
#include <device_atomic_functions.h>
//Standard C++ utilities

//On NVIDIA GPUs, threads are executed in warps of 32
#define BACKWARD_W_BATCH_THREADS 32
//defines a constant named BACKWARD_W_BATCH_THREADS

///////////////////////////////
//////defines functions////////
///////////////////////////////

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
//Ensures tensor is on GPU

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
//Ensures memory layout is contiguous (CUDA kernels assume this)

//Used at the start of every exposed function
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

// adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
//CUDA error checking
#define gpuErrchk(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(const cudaError_t code, const char *const file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
//if a CUDA call fails -> GPUassert: <error> <file> <line> -> then exits. (Critical for debugging kernels)

//ceiling division
template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }


/**********************************************************************************************************************/


template <typename T> struct AtomicFPOp;

//Atomic func: special operation used in parallel programming (CPU/GPU) that manipulates a variable safely when multiple threads are accessing it at the same time.
//Atomic func for half precision, because cuda does not provide it
//Use this implementation when the type T is at::Half.
template <> struct AtomicFPOp<at::Half> {
    //call it like: AtomicFPOp<at::Half>()(ptr, value, func);
    //it applies a generic operation func 
    template <typename func_t> inline __device__ at::Half operator()(at::Half *address, at::Half val, const func_t &func) {
        unsigned int *address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
        unsigned int old = *address_as_ui;
        unsigned int assumed;

        at::Half hsum;
        do {
            assumed = old;
            hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
            hsum = func(hsum, val);
            old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
            old = atomicCAS(address_as_ui, assumed, old);
        } while (assumed != old);
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        return hsum;
    }
};

//gpuAtomicAdd is a portable atomic addition helper for GPU kernels
static inline __device__ at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
#if defined(USE_ROCM) || ((defined(CUDA_VERSION) && CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))

    unsigned int *aligned = (unsigned int *)((size_t)address - ((size_t)address & 2));
    unsigned int old = *aligned;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
        __half sum = c10::Half(__ushort_as_half(old_as_us)) + c10::Half(__float2half((float)val));
        unsigned short sum_as_us = __half_as_ushort(sum);
        unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | sum_as_us;
        old = atomicCAS(aligned, assumed, sum_as_ui);
    } while (assumed != old);
    unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
    return c10::Half((__half_raw)__ushort_as_half(old_as_us));
#else
    return atomicAdd(reinterpret_cast<__half *>(address), val);
#endif
}
//Those lines exist mainly for API consistency and overloading, so the same function name (gpuAtomicAdd) can be used for different numeric types.
static inline __device__ float gpuAtomicAdd(float *address, float val) { return atomicAdd(address, val); }

static inline __device__ double gpuAtomicAdd(double *address, double val) { return atomicAdd(address, val); }




/**********************************************************************************************************************/
/**  TRAINING MODE  ***************************************************************************************************/
/**********************************************************************************************************************/

//it implements the cuda forward pass of a LogicLayer
//__global__ → this is a CUDA kernel that runs on the GPU and is launched from CPU code
//this function, behaves the same as bin_op_s
template <typename scalar_t>
__global__ void logic_layer_cuda_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y
) { //Inputs
    //x : [num_inputs, batch]
    //a : [num_neurons]
    //b : [num_neurons]
    //w : [num_neurons, nuber_of_gates]
    //Output
    //y : [num_neurons, batch]

    for (  // batch dim, 
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {
            //indexes
            const auto idx_a = a[col];
            const auto idx_b = b[col];
            //Values based on indexes
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];
            
            // ===============================================================
            // CHANGED: ternary polynomial basis instead of binary logic basis
            // ===============================================================

            const auto w_ = w[col];

            y[col][row] =
                w_[0]                                       // -1
                + w_[1] * a_                                  // A
                + w_[2] * b_                                  // B
                + w_[3] * (-a_)                               // -A
                + w_[4] * (-b_)                               // -B
                + w_[5] * (a_ * b_)                           // AB
                + w_[6] * (-a_ * b_)                          // -AB
                + w_[7] * static_cast<scalar_t>(0)             // 0
                + w_[8] * (a_ * a_)                           // A²
                + w_[9] * (b_ * b_)                           // B²
                + w_[10] * (-a_ * a_)                         // -A²
                + w_[11] * (-b_ * b_)                         // -B²
                + w_[12] * static_cast<scalar_t>(1);          // 1
    }}
}

//Computes the dL/dw
template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_w_
) {

    const auto row_ = blockIdx.x * blockDim.x + threadIdx.x;

    for (  // neuron dim
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        col < grad_y.size(0);
        col += blockDim.y * gridDim.y
    ) {
        const auto idx_a = a[col];
        const auto idx_b = b[col];
        scalar_t grad_w_a  = 0;
        scalar_t grad_w_b  = 0;
        scalar_t grad_w_ab = 0;
        scalar_t grad_w_aa = 0;
        scalar_t grad_w_bb = 0;
        scalar_t grad_w_1  = 0;
        for (int row = row_; row < grad_y.size(1); row += BACKWARD_W_BATCH_THREADS) {  // batch dim
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];
            const auto grad_y_ = grad_y[col][row];
            //its not the anaytical gradient. Computes the structural gradients that form the total.
            // compute grad_w
            grad_w_a  += a_ * grad_y_;
            grad_w_b  += b_ * grad_y_;
            grad_w_ab += (a_ * b_) * grad_y_;
            grad_w_aa += (a_ * a_) * grad_y_;
            grad_w_bb += (b_ * b_) * grad_y_;
            grad_w_1  += grad_y_;
        }

        grad_w_[col][row_][0] = grad_w_a;
        grad_w_[col][row_][1] = grad_w_b;
        grad_w_[col][row_][2] = grad_w_ab;
        grad_w_[col][row_][3] = grad_w_aa;
        grad_w_[col][row_][4] = grad_w_bb;
        grad_w_[col][row_][5] = grad_w_1;
    }
}

//computes the dL/dx
//When col == idx_a, we compute ∂y/∂a.
//When col == idx_b, we compute ∂y/∂b.
//given_x_indices_of_y → flat list of all neurons depending on each input
//given_x_indices_of_y_start → start index of each input’s neurons in the flat list
template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y_start,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_x.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_x.size(0);
            col += blockDim.y * gridDim.y
        ) {

            scalar_t grad_x_ = 0;

            const auto start = given_x_indices_of_y_start[col];
            const auto end = given_x_indices_of_y_start[col + 1];

            for (int cur = start; cur < end; ++cur) {
                const auto idx_y = given_x_indices_of_y[cur];
                const auto idx_a = a[idx_y];
                const auto idx_b = b[idx_y];
                const auto grad_y_ = grad_y[idx_y][row];
                const auto idx_is_a = idx_a == col;

                // compute grad_x
                //derivate with respect to a
                if (idx_is_a) {
                    const auto b_ = x[idx_b][row];
                    const auto a_ = x[idx_a][row];
                    const auto dy_dx = (
                        (w[idx_y][1]
                        + (-w[idx_y][3])
                        + w[idx_y][5] * b_)
                        + (w[idx_y][6] * -b_
                        + w[idx_y][8] * static_cast<scalar_t>(2) * a_
                        + w[idx_y][10] * static_cast<scalar_t>(2) * -a_));
                    grad_x_ += dy_dx * grad_y_;
                //derivate with respect to b
                } else {
                    const auto b_ = x[idx_b][row];
                    const auto a_ = x[idx_a][row];
                    const auto dy_dx = (
                        (w[idx_y][2]
                        + (-w[idx_y][4])
                        + w[idx_y][5] * a_)
                        + (w[idx_y][6] * -a_
                        + w[idx_y][9] * static_cast<scalar_t>(2) * b_
                        + w[idx_y][11] * static_cast<scalar_t>(2) * -b_));
                    grad_x_ += dy_dx * grad_y_;
                }
            }
            grad_x[col][row] = grad_x_;
    }}
}

//PyTorch CUDA wrapper for logic layer’s forward pass. 
//It handles tensor preparation, launching the CUDA kernel, and returning the output.
torch::Tensor logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    //inputs check
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = w.size(0);

    //Allocates memory for the forward output
    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    //2D thread block: 32 × 32 threads per block
    dim3 threads_per_block(32, 32);

    //65535 blocks per dimension
    const dim3 blocks_per_grid(
        std::min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        std::min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    //Launch the CUDA Kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_cuda_forward", ([&] {
                           logic_layer_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                           );
                       }));

    //Checks for kernel launch errors and Synchronizes the device to make sure kernel finishes
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}

//PyTorch CUDA wrapper for logic layer’s backward pass. dL/dw 
torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
) {
    //inputs check
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(grad_y);


    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = grad_y.size(0);

    //Holds per-thread partial sums of the 6 independent gradients. Shape: [neurons, threads_per_block, number_of_fund_gradients]
    auto grad_w_6 = torch::empty({out_size, BACKWARD_W_BATCH_THREADS, 6}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(BACKWARD_W_BATCH_THREADS, 1024 / BACKWARD_W_BATCH_THREADS);

    const dim3 blocks_per_grid(
        1,
        std::min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );
    //Launch CUDA Kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_cuda_backward_w", ([&] {
                           logic_layer_cuda_backward_w_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_w_6.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Sums over BACKWARD_W_BATCH_THREADS to get full batch accumulation
    //Shape becomes [out_size, 4]
    const auto grad_w_components = grad_w_6.sum(1);
    //Then each component is extracted
    const auto grad_w_a = grad_w_components.index({torch::indexing::Slice(), 0});
    const auto grad_w_b = grad_w_components.index({torch::indexing::Slice(), 1});
    const auto grad_w_ab = grad_w_components.index({torch::indexing::Slice(), 2});
    const auto grad_w_aa = grad_w_components.index({torch::indexing::Slice(), 3});
    const auto grad_w_bb = grad_w_components.index({torch::indexing::Slice(), 4});
    const auto grad_w_ = grad_w_components.index({torch::indexing::Slice(), 5});

    //The logic layer uses number_of_gates weights but only 6 independent gradients are computed (AB, AA, BB, A, B, 1).
    //All other weights are linear combinations of these 4 basis gradients.
    //The reconstruction is done with torch::stack:
    //Shape: [out_size, number_of_gates]
    const auto grad_w = torch::stack({
        -grad_w_,                         // w0  = -1
        grad_w_a,                         // w1  = A
        grad_w_b,                         // w2  = B
        -grad_w_a,                        // w3  = -A
        -grad_w_b,                        // w4  = -B
        grad_w_ab,                        // w5  = AB
        -grad_w_ab,                       // w6  = -AB
        torch::zeros({out_size}, torch::dtype(x.dtype()).device(x.device())), // w7 = 0
        grad_w_aa,                        // w8  = A²
        grad_w_bb,                        // w9  = B²
        -grad_w_aa,                       // w10 = -A²
        -grad_w_bb,                       // w11 = -B²
        grad_w_                           // w12 = 1
    }, 1);


    return grad_w;
}

//PyTorch CUDA wrapper for logic layer’s backward pass. dL/dx 
torch::Tensor logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
) {
    //inputs check
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(given_x_indices_of_y_start);
    CHECK_INPUT(given_x_indices_of_y);

    auto grad_x = torch::empty_like(x);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        std::min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        std::min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );
    //Launch CUDA Kernel
    //the last two arguments describe which neurons depend on each input (Instead of checking all neurons, the kernel only loops over relevant ones)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_cuda_backward_x", ([&] {
                           logic_layer_cuda_backward_x_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               given_x_indices_of_y_start.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               given_x_indices_of_y.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
                           );
                       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return grad_x;
}


/**********************************************************************************************************************/
/**  INFERENCE MODE  **************************************************************************************************/
/**********************************************************************************************************************/

// ===============================================================
// CHANGED: ternary logic operator instead of binary bitwise gates
// supports inputs in {-1,0,1}
// ===============================================================

template <typename T>
__device__ __forceinline__ T tern_op_eval(const T a_, const T b_, const int op_idx)
{
    switch(op_idx)
    {
        case 0:  return static_cast<T>(-1);      // constant -1
        case 1:  return a_;                      // A
        case 2:  return b_;                      // B
        case 3:  return -a_;                     // -A
        case 4:  return -b_;                     // -B
        case 5:  return a_ * b_;                 // A*B
        case 6:  return -a_ * b_;                // -A*B
        case 7:  return static_cast<T>(0);       // constant 0
        case 8:  return a_ * a_;                 // A^2
        case 9:  return b_ * b_;                 // B^2
        case 10: return -a_ * a_;                // -A^2
        case 11: return -b_ * b_;                // -B^2
        default: return static_cast<T>(1);       // constant 1
    }
}

//The Evaluation Kernel
//Here w is uint8_t, not float weights.(operation index (0-15))
//That means each neuron stores just the index of the chosen logic gate.
template <typename scalar_t>
__global__ void logic_layer_cuda_eval_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y
) {
    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {

            const auto idx_a = a[col];
            const auto idx_b = b[col];
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];
            const auto w_ = w[col];
            //applies one logic operation per neuron
            y[col][row] = tern_op_eval(a_, b_, w_);
        }
    }
}
//PyTorch CUDA wrapper for logic layer’s evaluation
torch::Tensor logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    //inputs check
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = w.size(0);

    //Creates the output tensor filled with 0
    auto y = torch::zeros({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        std::min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        std::min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    //Dispatch for Integer Types.
    //Unlike training kernels (which used floating types), this kernel expects integer inputs, because it uses bitwise logic operations.
    AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "logic_layer_cuda_eval_kernel", ([&] {
                                   logic_layer_cuda_eval_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                       x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                       b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                       w.packed_accessor64<uint8_t, 1, torch::RestrictPtrTraits>(),
                                       y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                                   );
                               }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}


/**********************************************************************************************************************/
// Packs a 2D ternary tensor (int8: -1,0,1) into 2-bit packed uint32_t
__global__ void tensor_packtern_cuda_kernel(
    const int8_t* __restrict__ t, // [neurons, batch]
    uint32_t* __restrict__ out,   // packed output
    int neurons,
    int batch,
    int pad_len
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x; // batch index
        int row = blockIdx.y * blockDim.y + threadIdx.y; // neuron index

        if (col >= batch || row >= neurons) return;

        int8_t val = t[row * batch + col];
        // map ternary to 2-bit
        uint32_t bits = (val == -1) ? 0b00 : ((val == 0) ? 0b01 : 0b10);

        int packed_idx = (row * batch + col) / 16; // 16 2-bit values per uint32
        int shift = ((row * batch + col) % 16) * 2;

        atomicOr(&out[packed_idx], bits << shift); // set 2-bit
}

// CUDA wrapper
std::tuple<torch::Tensor,int> tensor_packtern_cuda(
    torch::Tensor t
) {
    CHECK_INPUT(t);
    const auto neurons = t.size(0);
    const auto batch = t.size(1);

    // pad to multiple of 16 (because 16 2-bit values per 32-bit integer)
    int pad_len = (16 - (neurons * batch) % 16) % 16;
    int packed_size = (neurons * batch + pad_len) / 16;

    auto out = torch::zeros({packed_size}, torch::dtype(torch::kUInt32).device(t.device()));

    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid(
        (batch + threads_per_block.x - 1) / threads_per_block.x,
        (neurons + threads_per_block.y - 1) / threads_per_block.y
    );

    tensor_packtern_cuda_kernel<<<blocks_per_grid, threads_per_block>>>(
        t.data_ptr<int8_t>(),
        out.data_ptr<uint32_t>(),
        neurons,
        batch,
        pad_len
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return std::make_tuple(out, pad_len);
}

/**********************************************************************************************************************/

// Sum ternary values over groups of neurons
__global__ void groupternsum_kernel(
    const uint32_t* __restrict__ t, // packed tensor
    int* __restrict__ out,          // [batch, k]
    int neurons,
    int batch,
    int k,
    int pad_len
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // batch
    int row = blockIdx.y * blockDim.y + threadIdx.y; // class/group

    if (col >= batch || row >= k) return;

    int group_size = neurons / k;
    int sum = 0;

    for (int i = 0; i < group_size; ++i) {
        int neuron_idx = row * group_size + i;
        int flat_idx = neuron_idx * batch + col;
        int packed_idx = flat_idx / 16;
        int shift = (flat_idx % 16) * 2;
        uint32_t bits = (t[packed_idx] >> shift) & 0b11;

        // map back to ternary value
        int val = (bits == 0b00) ? -1 : ((bits == 0b01) ? 0 : 1);
        sum += val;
    }

    out[col * k + row] = sum;
}

// Wrapper
torch::Tensor groupternsum(torch::Tensor t, int pad_len, int k, int neurons, int batch) {
    CHECK_INPUT(t);
    auto out = torch::zeros({batch, k}, torch::dtype(torch::kInt32).device(t.device()));

    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid(
        (batch + threads_per_block.x - 1) / threads_per_block.x,
        (k + threads_per_block.y - 1) / threads_per_block.y
    );

    groupternsum_kernel<<<blocks_per_grid, threads_per_block>>>(
        t.data_ptr<uint32_t>(),
        out.data_ptr<int>(),
        neurons,
        batch,
        k,
        pad_len
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return out;
}

/**********************************************************************************************************************/

