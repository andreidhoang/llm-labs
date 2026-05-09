import torch

# def weighted_sum(x, weight):
#     # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
#     return (weight * x).sum(axis=-1)

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]], device="cuda")
w = torch.tensor([10., 20., 30.], device="cuda")

# expected:
# row0 = 1*10 + 2*20 + 3*30 = 140
# row1 = 4*10 + 5*20 + 6*30 = 320

print("x.shape:", x.shape)
print("x.stride:", x.stride())
print("x.dtype:", x.dtype)
print("x.device:", x.device)
print("x.is_contiguous:", x.is_contiguous())
print("w.shape:", w.shape)
print("w.stride:", w.stride())
print("M rows:", x.numel() // x.shape[-1])
print("D reduce dim:", x.shape[-1])
print("memory allocated:", torch.cuda.memory_allocated())


import triton
import triton.language as tl
@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_stride_row, x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,  # Likely 1
    output_stride_row,  # Likely 1
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)
    # Block pointers give us a way to select from an ND region of memory
    #  and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to load/store at a time
    # - The order of the dimensions in memory from major to minor
    #    axes (= np.argsort(strides)) for optimizations, needed for
    #    TMA support on >=Hopper
    pass
