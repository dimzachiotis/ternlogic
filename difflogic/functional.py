import torch
import numpy as np

BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
number_of_gates=13
# | id | Operator             | AB=-1-1 | AB=-10  | AB=-11  | AB=0-1  | AB=00   | AB=01   | AB=1-1  | AB=10   | AB=11   |
# |----|----------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
# | 0  | -1                   | -1      | -1      | -1      | -1      | -1      | -1      | -1      | -1      | -1      |
# | 1  | A                    | -1      | -1      | -1      | 0       | 0       | 0       | 1       | 1       | 1       |
# | 2  | B                    | -1      | 0       | 1       | -1      | 0       | 1       | -1      | 0       | 1       |
# | 3  | -A                   | 1       | 1       | 1       | 0       | 0       | 0       | -1      | -1      | -1      |
# | 4  | -B                   | 1       | 0       | -1      | 1       | 0       | -1      | 1       | 0       | -1      |
# | 5  | A*B                  | 1       | 0       | -1      | 0       | 0       | 0       | -1      | 0       | 1       |
# | 6  | -A*B                 | -1      | 0       | 1       | 0       | 0       | 0       | 1       | 0       | -1      |
# | 7  | 0                    | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0       | 0       |
# | 8  | A*A                  | 1       | 1       | 1       | 0       | 0       | 0       | 1       | 1       | 1       |
# | 9  | B*B                  | 1       | 0       | 1       | 1       | 0       | 1       | 1       | 0       | 1       |
# | 10 | -A*A                 | -1      | -1      | -1      | 0       | 0       | 0       | -1      | -1      | -1      |
# | 11 | -B*B                 | -1      | 0       | -1      | -1      | 0       | -1      | -1      | 0       | -1      |
# | 12 | 1                    | 1       | 1       | 1       | 1       | 1       | 1       | 1       | 1       | 1       |




#This function returns the value of the form of the ith ternary logic gate shown above
def tern_op(a, b, i):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    #if a and b have incompatible number of rows then stop and print their number of their rows
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)
    #if a and b have incompatible number of collumns then stop and print their number of their collumns
    #Makes sure a and b have the same size 
    if i == 0:
        return torch.full_like(a,-1)
    elif i == 1:
        return a
    elif i == 2:
        return b
    elif i == 3:
        return -a
    elif i == 4:
        return -b
    elif i == 5:
        return a*b
    elif i == 6:
        return -a*b
    elif i == 7:
        return torch.zeros_like(a)
    elif i == 8:
        return a*a
    elif i == 9:
        return b*b
    elif i == 10:
        return -a*a
    elif i == 11:
        return -b*b
    elif i == 12:
        return torch.ones_like(a)

#Implements the relaxation form of each ternary logic gate


def tern_op_s(a, b, i_s):
    r = torch.zeros_like(a)
    for i in range(number_of_gates):
        u = tern_op(a, b, i)
        r = r + i_s[..., i] * u
    return r
#Apply all ternary logic operations to a and b, weight each result based on i_s[..., i], and sum them


########################################################################################################################

#constructs out_dim random index pairs from x 
def get_unique_connections(in_dim, out_dim, device='cpu'):
    assert out_dim * 2 >= in_dim, 'The number of neurons ({}) must not be smaller than half of the number of inputs ' \
                                  '({}) because otherwise not all inputs could be used or considered.'.format(
        out_dim, in_dim
    )
    #In case you want cpu, you have to initialize it device='cpu' else 'cuda'

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



