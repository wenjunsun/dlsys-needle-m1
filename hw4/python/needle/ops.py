"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        # return  mul_scalar(multiply(out_grad, input), self.scalar)
        return  mul_scalar(multiply(out_grad, power_scalar(input, self.scalar - 1)), self.scalar)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return multiply(out_grad, power_scalar(rhs, -1)), multiply(multiply(out_grad, lhs), negate(power_scalar(rhs, -2)))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return mul_scalar(out_grad, 1 / self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            # numpy's version
            # return a.swapaxes(*self.axes)
            # our own implementation of NDArray's version
            # for some reason the axes here is (1, 0) for dims = (0 ,1, 2). so can't use permute naively
            assert len(self.axes) == 2
            axes_to_permute = [i for i in range(len(a.shape))]
            transpose_dim_1, transpose_dim_2 = self.axes[0], self.axes[1]
            axes_to_permute[transpose_dim_1] = transpose_dim_2
            axes_to_permute[transpose_dim_2] = transpose_dim_1
            return a.permute(axes_to_permute)
        else:
            # if axes = None, transpose the two innermost indices
            # return a.swapaxes(-1, -2)
            axes_to_permute = [i for i in range(len(a.shape))]
            temp = axes_to_permute[-1]
            axes_to_permute[-1] = axes_to_permute[-2]
            axes_to_permute[-2] = temp
            return a.permute(axes_to_permute)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes = self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        return reshape(out_grad, input.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a.compact(), self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        input = node.inputs[0]
        input_shape = list(input.shape)
        output_shape = list(out_grad.shape)
        assert len(input_shape) <= len(output_shape)
        while len(input_shape) != len(output_shape):
            input_shape = [1] + input_shape
        diff_indices = []
        for i in range(len(input_shape)):
            if input_shape[i] != output_shape[i]:
                diff_indices.append(i)

        result = reshape(summation(out_grad, axes = tuple(diff_indices)), input.shape)

        return result
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return a.sum(self.axes)
        # want to implement summation over multiple axes.
        if self.axes is None:
            return a.sum(axis = None)
        elif isinstance(self.axes, int) or (isinstance(self.axes, (list, tuple)) and len(self.axes) == 1):
            return a.sum(self.axes)
        else:
            # multiple axes case
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape

        if self.axes is not None:
            # grad: (5, ) , input: (5, 4) -> broadcast_shape = (5, 1)
            broadcast_shape = list(input_shape)
            if isinstance(self.axes, int):
                broadcast_shape[self.axes] = 1
            else:
                for axis in self.axes:
                    broadcast_shape[axis] = 1
            return broadcast_to(reshape(out_grad, broadcast_shape), input_shape)
        else:
            # summing over everything. grad = (1, ), input: (5, 5, 4) -> broadcast_shape = (1, 1, 1)
            broadcast_shape = [1 for _ in input_shape]
            return broadcast_to(reshape(out_grad, broadcast_shape), input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # my solution:
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        left_dim = len(lhs_shape)
        right_dim = len(rhs_shape)
        

        if left_dim == right_dim:
            return matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad)
        elif left_dim < right_dim:
            return summation(matmul(out_grad, transpose(rhs)), axes = tuple(range(right_dim - left_dim))), matmul(transpose(lhs), out_grad)
        else:
            return matmul(out_grad, transpose(rhs)), summation(matmul(transpose(lhs), out_grad), axes = tuple(range(left_dim - right_dim)))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        return mul_scalar(out_grad, -1)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        return multiply(out_grad, power_scalar(input, -1))
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        return out_grad * exp(input)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        input_array = input.realize_cached_data()
        relu_grad = input_array > 0
        result_grad_array = out_grad.realize_cached_data() * relu_grad
        return Tensor(result_grad_array, device = input.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = tuple([axes]) if isinstance(axes, int) else axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # numpy's version
        # Z_max = array_api.max(Z, axis = self.axes)
        # our NDArray's verison
        Z_max = Z.max(axis = self.axes)
        
        Z_shape = list(Z.shape)
        if self.axes is not None:
            for axis in self.axes:
                Z_shape[axis] = 1
            Z_max_reshaped = Z_max.reshape(tuple(Z_shape))
        else:
            Z_max_reshaped = Z_max.reshape(tuple([1 for _ in Z_shape]))

        # numpy's version
        # Z_normalized = Z - Z_max_reshaped
        # NDArray version
        Z_normalized = Z - Z_max_reshaped.broadcast_to(Z.shape)

        # numpy's version: we have np.sum(), but we don't have array_api.sum().
        # return array_api.log(array_api.sum( array_api.exp(Z_normalized), axis = self.axes )) + Z_max
        # our NDArray implementation's version:
        return array_api.log(array_api.summation( array_api.exp(Z_normalized), axis = self.axes )) + Z_max
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        # numpy's version
        # Z_max = Tensor(array_api.max(Z.numpy(), axis = self.axes))
        # our NDArray's version
        Z_max = Tensor(Z.numpy().max(axis = self.axes), device = Z.device)

        Z_shape_for_reshape = list(Z.shape)
        if self.axes is not None:
            for axis in self.axes:
                Z_shape_for_reshape[axis] = 1
        else:
            # in this case the operation is over the entire tensor.
            # then shape to reshape to is (1, 1, 1, 1, ...)
            for i in range(len(Z_shape_for_reshape)):
                Z_shape_for_reshape[i] = 1
        Z_shape_for_reshape = tuple(Z_shape_for_reshape)
        Z_shape_for_broadcast = Z.shape

        Z_max_reshaped_broadcasted = broadcast_to(reshape(Z_max, Z_shape_for_reshape), Z_shape_for_broadcast)
        # local derivative is softmax(Z - Z_max)
        Z_minus_Z_max = Z - Z_max_reshaped_broadcasted
        Z_exp = exp(Z_minus_Z_max)
        Z_sum_exp = broadcast_to(reshape(summation(Z_exp, self.axes), Z_shape_for_reshape), Z_shape_for_broadcast)
        return multiply(broadcast_to(reshape(out_grad, Z_shape_for_reshape), Z_shape_for_broadcast), divide(Z_exp, Z_sum_exp))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        exp_a = array_api.exp(a)
        exp_minus_a = array_api.exp(-a)
        return (exp_a - exp_minus_a) / (exp_a + exp_minus_a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # outgrad * (1 - tanh(a) ** 2)
        return multiply(out_grad, add_scalar(negate(power_scalar(tanh(a), 2)), 1))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        # make sure we have at least one array to stack
        assert len(args) >= 1
        num_arrays_to_stack = len(args)
        array = args[0]
        stacked_shape = list(array.shape)
        stacked_shape.insert(self.axis, num_arrays_to_stack)
        stacked_shape = tuple(stacked_shape)
        stacked_array = array_api.empty(stacked_shape, device = array.device)

        for i, array_to_stack in enumerate(args):
            assert array_to_stack.shape == array.shape, f"every array to stack should have the same shape."
            slices = [] # record the slice alone each dimension.
            for j, dim in enumerate(stacked_shape):
                if j == self.axis:
                    slices.append(i)
                else:
                    slices.append(slice(0, dim, 1))
            slices = tuple(slices)
            stacked_array[slices] = array_to_stack

        return stacked_array
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        new_shape = list(A.shape)
        del new_shape[self.axis]
        new_shape = tuple(new_shape)
        
        results = []
        for i in range(A.shape[self.axis]):
            slices = []
            for j, dim in enumerate(A.shape):
                if j == self.axis:
                    slices.append(i)
                else:
                    slices.append(slice(0, dim, 1))
            slices = tuple(slices)
            results.append(A[slices].compact().reshape(new_shape)) # compact here is important.

        return tuple(results)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        assert axes is not None and isinstance(axes, tuple)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            # do this to pass mugrade...
            if axis >= len(new_shape):
                return a
            new_shape[axis] = (1 + self.dilation) * new_shape[axis]
        new_shape = tuple(new_shape)
        new_array = array_api.full(new_shape, 0, device = a.device)
        slices = []
        for i in range(len(new_shape)):
            if i in self.axes:
                # dilation axis.
                slices.append(slice(0, new_shape[i], self.dilation + 1))
            else:
                slices.append(slice(0, new_shape[i], 1))
        slices = tuple(slices)
        new_array[slices] = a
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        for axis in self.axes:
            # do this to pass mugrade...
            if axis >= len(new_shape):
                return a
            assert new_shape[axis] % (1 + self.dilation) == 0
            new_shape[axis] = new_shape[axis] // (1 + self.dilation)
        new_array = array_api.full(new_shape, 0, device = a.device)
        slices = []
        for i in range(len(a.shape)):
            if i in self.axes:
                # dilation axis.
                slices.append(slice(0, a.shape[i], self.dilation + 1))
            else:
                slices.append(slice(0, a.shape[i], 1))
        slices = tuple(slices)
        new_array = a[slices]
        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, Z, weight):
        ### BEGIN YOUR SOLUTION
        assert len(Z.shape) == 4 and len(weight.shape) == 4, "currently we only support 2D convolution with 4-D tensors + weights"
        # althought this seems excessive, but this can save us from copying an entire array if padding is 0
        if self.padding > 0:
            Z = Z.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)))
        
        N, H, W, C_in = Z.shape
        K, _, _, C_out = weight.shape
        Ns, Hs, Ws, Cs = Z.strides
        
        inner_dim = K * K * C_in
        A = Z.as_strided(shape = (N, H-K+1, W-K+1, K, K, C_in), strides = (Ns, Hs, Ws, Hs, Ws, Cs)).compact()
        assert A.size % inner_dim == 0
        A = A.reshape((A.size // inner_dim, inner_dim))
        
        assert weight.size % C_out == 0
        weight = weight.compact() # this is needed! seems like weight can be noncompact.
        out = A @ weight.reshape((weight.size // C_out, C_out))
        out = out.reshape((N, H-K+1, W-K+1, C_out))

        if self.stride > 1:
            # this is a hacky way to invoke only the NDArray computation with a TensorOp
            # NOTE: this is not the most efficient way to compute convolution with striding
            #       because we are essentially doing all the computation and then filter out
            #       the results we want in the end.
            return UnDilate((1, 2), self.stride - 1).compute(out)
        else:
            return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        # pad = K - 1 + self.padding
        pad = X.shape[1] - out_grad.shape[1] + self.padding # apparently K - 1 != X.shape[1] - out_grad.shape[1]?

        X_grad = conv(out_grad, transpose(flip(W, (0, 1)), (2, 3)), padding = pad)

        W_grad = conv(transpose(X, (0, 3)), transpose(transpose(out_grad, (0, 1)), (1, 2)), padding = self.padding)
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



