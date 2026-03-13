#include <torch/extension.h>
//Includes PyTorch C++ API for building extensions

#include <c10/util/Half.h>
//Provides PyTorch half-precision utilities.
#include <cuda.h>
#include <cuda_runtime.h>
//CUDA driver/runtime APIs for GPU programming.

#include <array>
#include <cmath>
#include <vector>
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
    //w : [num_neurons, 16]
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
            //w_=[1x16]
            const auto w_ = w[col];

            //y[col][row] = weighted sum of logical functions of (a_, b_)
            y[col][row] = (
                 ((w_[1] * (a_ * b_)
                 + w_[2] * (a_ - a_ * b_))
                + (w_[3] * a_
                 + w_[4] * (b_ - a_ * b_)))
               + ((w_[5] * b_
                 + w_[6] * (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
                + (w_[7] * (a_ + b_ - a_ * b_)
                 + w_[8] * (static_cast<scalar_t>(1) - (a_ + b_ - a_ * b_)))))
              + (((w_[9] * (static_cast<scalar_t>(1) - (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
                 + w_[10] * (static_cast<scalar_t>(1) - b_)) +
                  (w_[11] * (static_cast<scalar_t>(1) - b_ + a_ * b_)
                 + w_[12] * (static_cast<scalar_t>(1) - a_))) +
                  (w_[13] * (static_cast<scalar_t>(1) - a_ + a_ * b_)
                 + w_[14] * (static_cast<scalar_t>(1) - a_ * b_)
                 + w_[15])
            );
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
        scalar_t grad_w_local_1 = 0;
        scalar_t grad_w_local_3 = 0;
        scalar_t grad_w_local_5 = 0;
        scalar_t grad_w_local_15 = 0;
        for (int row = row_; row < grad_y.size(1); row += BACKWARD_W_BATCH_THREADS) {  // batch dim
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];
            const auto grad_y_ = grad_y[col][row];
            //its not the anaytical gradient. Computes the structural gradients that form the total.
            // compute grad_w
            grad_w_local_1 += (a_ * b_) * grad_y_;
            grad_w_local_3 += a_ * grad_y_;
            grad_w_local_5 += b_ * grad_y_;
            grad_w_local_15 += grad_y_;
        }

        grad_w_[col][row_][0] = grad_w_local_1;
        grad_w_[col][row_][1] = grad_w_local_3;
        grad_w_[col][row_][2] = grad_w_local_5;
        grad_w_[col][row_][3] = grad_w_local_15;
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
                    const auto dy_dx = (
                        (w[idx_y][1] * b_
                       + w[idx_y][2] * (static_cast<scalar_t>(1) - b_)
                       + w[idx_y][3]) +
                        (w[idx_y][4] * -b_
                       + w[idx_y][6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * b_)
                       + w[idx_y][7] * (static_cast<scalar_t>(1) - b_)))
                     + ((w[idx_y][8] * (b_ - static_cast<scalar_t>(1))
                       + w[idx_y][9] * (static_cast<scalar_t>(2) * b_ - static_cast<scalar_t>(1))
                       + w[idx_y][11] * b_)
                     + (-w[idx_y][12]
                       + w[idx_y][13] * (b_ - static_cast<scalar_t>(1))
                       + w[idx_y][14] * -b_)
                    );
                    grad_x_ += dy_dx * grad_y_;
                //derivate with respect to b
                } else {
                    const auto a_ = x[idx_a][row];
                    const auto dy_dx = (
                         (w[idx_y][1] * a_
                        + w[idx_y][2] * -a_
                        + w[idx_y][4] * (static_cast<scalar_t>(1) - a_))
                       + (w[idx_y][5]
                        + w[idx_y][6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * a_)
                        + w[idx_y][7] * (static_cast<scalar_t>(1) - a_)))
                      + ((w[idx_y][8] * (a_ - static_cast<scalar_t>(1))
                        + w[idx_y][9] * (static_cast<scalar_t>(2) * a_ - static_cast<scalar_t>(1))
                        - w[idx_y][10])
                       + (w[idx_y][11] * (a_ - static_cast<scalar_t>(1))
                        + w[idx_y][13] * a_
                        + w[idx_y][14] * -a_)
                    );
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
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    //Launch the CUDA Kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_forward", ([&] {
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

    //Holds per-thread partial sums of the 4 independent gradients. Shape: [neurons, threads_per_block, 4]
    auto grad_w_4 = torch::empty({out_size, BACKWARD_W_BATCH_THREADS, 4}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(BACKWARD_W_BATCH_THREADS, 1024 / BACKWARD_W_BATCH_THREADS);

    const dim3 blocks_per_grid(
        1,
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );
    //Launch CUDA Kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_backward_w", ([&] {
                           logic_layer_cuda_backward_w_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_w_4.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                       }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Sums over BACKWARD_W_BATCH_THREADS to get full batch accumulation
    //Shape becomes [out_size, 4]
    const auto grad_w_components = grad_w_4.sum(1);
    //Then each component is extracted
    const auto grad_w_ab = grad_w_components.index({torch::indexing::Slice(), 0});
    const auto grad_w_a = grad_w_components.index({torch::indexing::Slice(), 1});
    const auto grad_w_b = grad_w_components.index({torch::indexing::Slice(), 2});
    const auto grad_w_ = grad_w_components.index({torch::indexing::Slice(), 3});

    //The logic layer uses 16 weights but only 4 independent gradients are computed (AB, A, B, 1).
    //All other weights are linear combinations of these 4 basis gradients.
    //The reconstruction is done with torch::stack:
    //Shape: [out_size, 16]
    const auto grad_w = torch::stack({
        torch::zeros({out_size}, torch::dtype(x.dtype()).device(x.device())),// w0 not used
        grad_w_ab,  // w1 = AB
        grad_w_a - grad_w_ab, // w2 = A - AB
        grad_w_a,// w3 = A
        grad_w_b - grad_w_ab,// w4 = B - AB
        grad_w_b,// w5 = B
        grad_w_a + grad_w_b - grad_w_ab - grad_w_ab,// w6 = A + B - 2AB
        grad_w_a + grad_w_b - grad_w_ab,// w7 = A + B - AB
        grad_w_ - grad_w_a - grad_w_b + grad_w_ab,// w8 = 1 - A - B + AB
        grad_w_ - grad_w_a - grad_w_b + grad_w_ab + grad_w_ab,// w9 = 1 - A - B + 2AB
        grad_w_ - grad_w_b, // w10 = 1 - B
        grad_w_ - grad_w_b + grad_w_ab,// w11 = 1 - B + AB
        grad_w_ - grad_w_a,// w12 = 1 - A
        grad_w_ - grad_w_a + grad_w_ab,// w13 = 1 - A + AB
        grad_w_ - grad_w_ab,// w14 = 1 - AB
        grad_w_,// w15 = bias
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
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );
    //Launch CUDA Kernel
    //the last two arguments describe which neurons depend on each input (Instead of checking all neurons, the kernel only loops over relevant ones)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_backward_x", ([&] {
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


// | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
// |----|----------------------|-------|-------|-------|-------|
// | 0  | 0                    | 0     | 0     | 0     | 0     |
// | 1  | A and B              | 0     | 0     | 0     | 1     |
// | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
// | 3  | A                    | 0     | 0     | 1     | 1     |
// | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
// | 5  | B                    | 0     | 1     | 0     | 1     |
// | 6  | A xor B              | 0     | 1     | 1     | 0     |
// | 7  | A or B               | 0     | 1     | 1     | 1     |
// | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
// | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
// | 10 | not(B)               | 1     | 0     | 1     | 0     |
// | 11 | B implies A          | 1     | 0     | 1     | 1     |
// | 12 | not(A)               | 1     | 1     | 0     | 0     |
// | 13 | A implies B          | 1     | 1     | 0     | 1     |
// | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
// | 15 | 1                    | 1     | 1     | 1     | 1     |

//returns the result of one of 16 binary logic operations depending on op_idx
template <typename T> __device__ __forceinline__ T bin_op_eval(const T a_, const T b_, const int op_idx) {
    switch (op_idx) {
    case 0:
        return static_cast<T>(0);
    case 1:
        return a_ & b_;
    case 2:
        return a_ & ~b_;
    case 3:
        return a_;
    case 4:
        return b_ & ~a_;
    case 5:
        return b_;
    case 6:
        return a_ ^ b_;
    case 7:
        return a_ | b_;
    case 8:
        return ~(a_ | b_);
    case 9:
        return ~(a_ ^ b_);
    case 10:
        return ~b_;
    case 11:
        return ~b_ | a_;
    case 12:
        return ~a_;
    case 13:
        return ~a_ | b_;
    case 14:
        return ~(a_ & b_);
    default:
        return ~static_cast<T>(0);
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
            y[col][row] = bin_op_eval(a_, b_, w_);
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
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    //Dispatch for Integer Types.
    //Unlike training kernels (which used floating types), this kernel expects integer inputs, because it uses bitwise logic operations.
    AT_DISPATCH_INTEGRAL_TYPES(x.type(), "logic_layer_cuda_eval_kernel", ([&] {
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

//This kernel packs boolean values into the bits of an integer
//t : [neurons, batch_in]
//b : [neurons, batch_out]
//batch_out = ceil(batch_in / bit_count)
//Even though bool tensor stores bool, on GPUs (and CPUs) a boolean element usually occupies 1 byte, not 1 bit. 
//we use less memory by packing
template <typename scalar_t>
__global__ void tensor_packbits_cuda_kernel(
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> t,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b
) {

    for (  // neuron in b and t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in b
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < b.size(1);
            col += blockDim.x * gridDim.x
        ) {

            //Convert Integer Type to Unsigned (bit operations behave correctly only on unsigned integers)
            typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
            //A union shares memory between variables.
            union {
                unsigned_scalar_t unsigned_scalar;
                scalar_t signed_scalar;
            } val;
            //Determine Bit Capacity
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            val.signed_scalar = b[row][col];
            for (unsigned int i = 0; i < bit_count; ++i) {
                const auto t_col = bit_count * col + i;
                if (t_col < t.size(1)) {    
                    const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(t[row][t_col]) << i;
                    val.unsigned_scalar = val.unsigned_scalar | bit_mask;
                }
            }
            b[row][col] = val.signed_scalar;
        }
    }
}

//PyTorch CUDA wrapper for tensor_packbits
//batch_out = ceil(batch_in / bit_count)
std::tuple<torch::Tensor, int> tensor_packbits_cuda(
    torch::Tensor t,
    const int bit_count
) {
    //input check
    CHECK_INPUT(t);

    const auto batch_in_size = t.size(1);
    //Packed batch size
    const auto batch_out_size = ceil_div(batch_in_size, static_cast<int64_t>(bit_count));
    const auto out_size = t.size(0);
    //Padding length
    const auto pad_len = (bit_count - batch_in_size % bit_count) % bit_count;

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    //Choose integer type
    auto dispatch_type = [bit_count]() {
        switch (bit_count) {
        case 8:
            return torch::kInt8;
        case 16:
            return torch::kInt16;
        case 32:
            return torch::kInt32;
        case 64:
            return torch::kInt64;
        default:
            throw std::invalid_argument("`bit_count` has to be in { 8, 16, 32, 64 }");
        }
    }();
    //Allocate output tensor
    auto b = torch::zeros({out_size, batch_out_size}, torch::dtype(dispatch_type).device(t.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "tensor_packbits_cuda_kernel", ([&] {
                                   tensor_packbits_cuda_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(t.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
                                                                                                                            b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
                               }));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {b, pad_len};
}


/**********************************************************************************************************************/

//This kernel reads bits from packed integers and counts them per class.
//b : packed bit tensor, shape = [neurons, packed_batch]
//t : integer tensor, shape = [classes, batch]
//Without packing: neurons × batch, ex: 512 neurons × 1024 samples
//With packing: 512 × 16 integers. Much smaller memory and fewer operations.
template <typename scalar_t>
__global__ void groupbitsum_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> t
) {

    for (  // class in t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in t
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < t.size(1);
            col += blockDim.x * gridDim.x
        ) {

            //Convert to Unsigned Type
            typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
            //Union Trick
            union scalar_t_ {
                unsigned_scalar_t unsigned_scalar;
                scalar_t signed_scalar;
            };
            //Determine Bit Capacity
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            int res = 0;
            //Neurons Per Class
            const auto class_size = b.size(0) / t.size(0);
            for (int i = 0; i < class_size; ++i) {
                //Read Packed Integer
                const scalar_t_ val = {.signed_scalar = b[row * class_size + i][col / bit_count]};
                const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(1) << static_cast<uint32_t>(col % bit_count);
                //Convert to boolean and add to count
                res += !!(val.unsigned_scalar & bit_mask);
            }
            t[row][col] = res;
        }
    }
}
//PyTorch CUDA wrapper for groupbitsum_kernel
torch::Tensor groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k
) {
    //input check
    CHECK_INPUT(b);

    //Determine Bit Size
    const int bit_count = 8 * b.element_size();

    const auto batch_in_size = b.size(1);
    const auto in_size = b.size(0);
    //Recover Original Batch Size
    const auto batch_out_size = batch_in_size * bit_count - pad_len;
    const auto out_size = static_cast<int64_t>(k);
    assert(in_size % k == 0);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    //Allocate Output Tensor
    auto t = torch::zeros({out_size, batch_out_size}, torch::dtype(torch::kInt32).device(b.device()));

    //Kernel Dispatch
    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "groupbitsum_kernel", ([&] {
                                   groupbitsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                        b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                        t.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
                                        );
                               }));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Transpose Output. Before transpose: t shape = [classes, batch], After transpose:[batch, classes]
    return t.transpose(0, 1).contiguous();
}


/**********************************************************************************************************************/

