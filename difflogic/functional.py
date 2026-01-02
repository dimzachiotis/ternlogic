import torch
import numpy as np

BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


# | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
# |----|----------------------|-------|-------|-------|-------|
# | 0  | 0                    | 0     | 0     | 0     | 0     |
# | 1  | A and B              | 0     | 0     | 0     | 1     |
# | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
# | 3  | A                    | 0     | 0     | 1     | 1     |
# | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
# | 5  | B                    | 0     | 1     | 0     | 1     |
# | 6  | A xor B              | 0     | 1     | 1     | 0     |
# | 7  | A or B               | 0     | 1     | 1     | 1     |
# | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
# | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
# | 10 | not(B)               | 1     | 0     | 1     | 0     |
# | 11 | B implies A          | 1     | 0     | 1     | 1     |
# | 12 | not(A)               | 1     | 1     | 0     | 0     |
# | 13 | A implies B          | 1     | 1     | 0     | 1     |
# | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
# | 15 | 1                    | 1     | 1     | 1     | 1     |
"""
A implies B = Not(A) or B
B implies A = Not(B) or A
"""

#This function returns the value of the form of the ith logic gate shown above
def bin_op(a, b, i):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    #if a and b have incompatible number of rows then stop and print their number of their rows
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)
    #if a and b have incompatible number of collumns then stop and print their number of their collumns
    #Makes sure a and b have the same size 
    if i == 0:
        return torch.zeros_like(a)
    elif i == 1:
        return a * b
    elif i == 2:
        return a - a * b
    elif i == 3:
        return a
    elif i == 4:
        return b - a * b
    elif i == 5:
        return b
    elif i == 6:
        return a + b - 2 * a * b
    elif i == 7:
        return a + b - a * b
    elif i == 8:
        return 1 - (a + b - a * b)
    elif i == 9:
        return 1 - (a + b - 2 * a * b)
    elif i == 10:
        return 1 - b
    elif i == 11:
        return 1 - b + a * b
    elif i == 12:
        return 1 - a
    elif i == 13:
        return 1 - a + a * b
    elif i == 14:
        return 1 - a * b
    elif i == 15:
        return torch.ones_like(a)
#Implements the relaxation form of each logic gate


def bin_op_s(a, b, i_s):
    r = torch.zeros_like(a)
    for i in range(16):
        u = bin_op(a, b, i)
        r = r + i_s[..., i] * u
    return r
#Apply all logic operations to a and b, weight each result based on i_s[..., i], and sum them


########################################################################################################################

#constructs out_dim random index pairs from x 
def get_unique_connections(in_dim, out_dim, device='cuda'):
    assert out_dim * 2 >= in_dim, 'The number of neurons ({}) must not be smaller than half of the number of inputs ' \
                                  '({}) because otherwise not all inputs could be used or considered.'.format(
        out_dim, in_dim
    )
    #In case you want cpu, you have to initialize it device='cpu'

    x = torch.arange(in_dim).long().unsqueeze(0)
    #x is a 2D shape (1,in_dim)

    # Take pairs (0, 1), (2, 3), (4, 5), ...
    a, b = x[..., ::2], x[..., 1::2]
    #a takes elements of x even indices, b of odd indices

    if a.shape[-1] != b.shape[-1]:
        m = min(a.shape[-1], b.shape[-1])
        a = a[..., :m]
        b = b[..., :m]
    #if a and b have different lenghts, thim them to the same lenght

    # If this was not enough, take pairs (1, 2), (3, 4), (5, 6), ...
    if a.shape[-1] < out_dim:
        a_, b_ = x[..., 1::2], x[..., 2::2]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        if a.shape[-1] != b.shape[-1]:
            m = min(a.shape[-1], b.shape[-1])
            a = a[..., :m]
            b = b[..., :m]

    # If this was not enough, take pairs with offsets >= 2:
    offset = 2
    while out_dim > a.shape[-1] > offset:
        a_, b_ = x[..., :-offset], x[..., offset:]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        offset += 1
        assert a.shape[-1] == b.shape[-1], (a.shape[-1], b.shape[-1])
    #generate shifted pairs, until a.shape[-1] >= out_dim
    #    (0,2), (1,3), (2,4), ...
    #    (0,3), (1,4), ...
    #    ...
    

    if a.shape[-1] >= out_dim:
        a = a[..., :out_dim]
        b = b[..., :out_dim]
    else:
        assert False, (a.shape[-1], offset, out_dim)
    #keep out_dim pairs, else fail if we didnt get enough pairs

    perm = torch.randperm(out_dim)
    #Generates a random permutation of indices 
    a = a[:, perm].squeeze(0)
    b = b[:, perm].squeeze(0)
    #shuffle pairs
    a, b = a.to(torch.int64), b.to(torch.int64)
    a, b = a.to(device), b.to(device)
    a, b = a.contiguous(), b.contiguous()
    return a, b


########################################################################################################################

#GradFactor is a class with functions that pass the input x forward unchanged (forward) but scale the gradient during backpropagation by a factor f(bachward)
class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x
# ctx is a context object that can be used to stash information
# for backward computation
    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None


########################################################################################################################



