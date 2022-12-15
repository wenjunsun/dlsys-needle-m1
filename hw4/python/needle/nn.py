"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # NOTE: need to set self.weight and self.bias to Parameter() + requires_grad = True so these weights are trainable
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True, device=device, dtype=dtype))
        self.has_bias = bias
        self.out_features = out_features
        if bias:
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1, requires_grad=True, device=device, dtype=dtype), shape = (1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output_shape = list(X.shape)
        output_shape[-1] = self.out_features
        if self.has_bias:
            return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias, output_shape)
        else:
            return ops.matmul(X, self.weight)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        shape_prod = np.prod(shape)
        return ops.reshape(X, (X.shape[0], shape_prod // X.shape[0]))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(ops.add_scalar(ops.exp(ops.negate(x)), 1), -1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_examples, output_dim = logits.shape[0], logits.shape[1] # assume logits are just a matrix here.
        y_onehot = init.one_hot(output_dim, y, device = logits.device, dtype = logits.dtype)
        # we have to use axes = 1, and not summing over everything at once,
        # because remember in the logsumexp we use Z_max to normalize things,
        # and that Z_max is different depending on the axes we sum over.
        # turns out the correct solution uses axes = 1 here.
        lhs = ops.logsumexp(logits, axes = 1)
        rhs = ops.summation(ops.multiply(logits, y_onehot), axes = 1)
        return ops.divide_scalar(ops.summation(lhs - rhs), num_examples)
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # init.zeros() and init.ones() default to have requires_grad = False.
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert x.shape[1] == self.dim
        batch_size, num_features = x.shape
        if self.training:
            # use batch average
            # NOTE: here we don't just set x = x.data to detach x because we do need
            #       x's computational graph to backpropate through the self.weight and bias below.
            mean_vec_1d = ops.divide_scalar(ops.summation(x, axes = 0), batch_size)
            mean_row_vec = ops.reshape(mean_vec_1d, (1, num_features))
            mean = ops.broadcast_to(mean_row_vec, x.shape)
            var_vec_1d = ops.divide_scalar(ops.summation(ops.power_scalar(x - mean, 2), axes = 0), batch_size)
            var_row_vec = ops.reshape(var_vec_1d, (1, num_features))
            var = ops.broadcast_to(var_row_vec, x.shape)
            
            # here we calculate the running average of means and variances.
            # NOTE: here we need to detach the mean and variance tensor calculated from our data before
            #       calculating the running mean and running variance, or else it will build up giant
            #       computational graphs and we will run out of memory.
            # self.running_mean = ops.mul_scalar(self.running_mean, 1 - self.momentum) + ops.mul_scalar(mean_vec_1d, self.momentum)
            # self.running_var = ops.mul_scalar(self.running_var, 1 - self.momentum) + ops.mul_scalar(var_vec_1d, self.momentum)
            self.running_mean = ops.mul_scalar(self.running_mean, 1 - self.momentum) + ops.mul_scalar(mean_vec_1d.data, self.momentum)
            self.running_var = ops.mul_scalar(self.running_var, 1 - self.momentum) + ops.mul_scalar(var_vec_1d.data, self.momentum)
 
            std = ops.power_scalar(var + self.eps, 1/2)

            # this will backpropagate through self.weight and self.bias.
            return ops.broadcast_to(ops.reshape(self.weight, (1, num_features)), x.shape) * ops.divide(x - mean, std) + ops.broadcast_to(ops.reshape(self.bias, (1, num_features)), x.shape)
        else:
            # use running average
            mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, self.dim)), x.shape)
            var = ops.broadcast_to(ops.reshape(self.running_var, (1, self.dim)), x.shape)
            std = ops.power_scalar(var + self.eps, 1/2)

            # NOTE: backpropagate through the the weights and biases if we are training, otherwise detach weights and biases.
            return ops.broadcast_to(ops.reshape(self.weight.data, (1, num_features)), x.shape) * ops.divide(x - mean, std) + ops.broadcast_to(ops.reshape(self.bias.data, (1, num_features)), x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # we can assume the input is 2-D tensor, where first dim = batch size, second dim = features
        # Remember to pass in the device and dtype here to init function calls
        self.weight = Parameter(init.ones(dim, requires_grad=True, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, dtype=dtype, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert x.shape[1] == self.dim
        batch_size, num_features = x.shape
        mean = ops.broadcast_to(ops.reshape(ops.divide_scalar(ops.summation(x, axes = 1), self.dim), (batch_size, 1)), x.shape)
        var = ops.broadcast_to(ops.reshape(ops.divide_scalar(ops.summation(ops.power_scalar(x - mean, 2), axes = 1), self.dim), (batch_size, 1)), x.shape)
        std = ops.power_scalar(var + self.eps, 1/2)

        # NOTE: backpropagate through the the weights and biases if we are training, otherwise detach weights and biases.
        if self.training:
            return ops.broadcast_to(ops.reshape(self.weight, (1, num_features)), x.shape) * ops.divide(x - mean, std) + ops.broadcast_to(ops.reshape(self.bias, (1, num_features)), x.shape)
        else:
            return ops.broadcast_to(ops.reshape(self.weight.data, (1, num_features)), x.shape) * ops.divide(x - mean, std) + ops.broadcast_to(ops.reshape(self.bias.data, (1, num_features)), x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p = 1 - self.p) # 1 - p entries to be 1, p entries to be 0
            return ops.divide_scalar(ops.multiply(x, mask), 1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        
        self.weight = Parameter(init.kaiming_uniform(fan_in, fan_out, weight_shape, requires_grad = True, device = device, dtype = dtype))
        if bias:
            bound = 1.0 / (fan_in ** 0.5)
            self.bias = Parameter(init.rand(out_channels, low = -bound, high = bound, requires_grad = True, device = device, dtype = dtype))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # NOTE: Accepts inputs in NCHW format, outputs also in NCHW format
        # convert input to NHWC format
        x = x.transpose((1, 2)).transpose((2, 3))
        # calculate padding so input and output size is the same.
        # assumes that the kernel size is odd.
        padding = (self.kernel_size - 1) // 2
        x = ops.conv(x, self.weight, self.stride, padding)
        # add bias.
        if self.bias is not None:
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            bias = ops.broadcast_to(bias, x.shape)
            x = x + bias
        # convert output to NCHW format
        return x.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size

        assert nonlinearity == 'tanh' or nonlinearity == 'relu', "only tanh and relu are currently supported as nonlinearity"
        if nonlinearity == 'tanh':
            self.nonlinearity = Tanh()
        elif nonlinearity == 'relu':
            self.nonlinearity = ReLU()
        else:
            raise ValueError
        # initializing learnable weights:
        k = 1 / hidden_size
        bound = k ** 0.5
        W_ih = Parameter(init.rand(input_size, hidden_size, low = -bound, high = bound, device = device, dtype = dtype, requires_grad = True))
        W_hh = Parameter(init.rand(hidden_size, hidden_size, low = -bound, high = bound, device = device, dtype = dtype, requires_grad = True))
        if bias:
            bias_ih = Parameter(init.rand(hidden_size, low = -bound, high = bound, device = device, dtype = dtype, requires_grad = True))
            bias_hh = Parameter(init.rand(hidden_size, low = -bound, high = bound, device = device, dtype = dtype, requires_grad = True))
        else:
            bias_ih = None
            bias_hh = None
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device = self.device, dtype = self.dtype, requires_grad = False)
        bs, hidden_size = h.shape
        if self.bias_ih is not None or self.bias_hh is not None:
            bias_ih = ops.broadcast_to(ops.reshape(self.bias_ih, (1, hidden_size)), (bs, hidden_size))
            bias_hh = ops.broadcast_to(ops.reshape(self.bias_hh, (1, hidden_size)), (bs, hidden_size))
            return self.nonlinearity(ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh) + bias_ih + bias_hh)
        else:
            return self.nonlinearity(ops.matmul(X, self.W_ih) + ops.matmul(h, self.W_hh))
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        # for the 2+ layers, the input to RNN is just the hidden state.
        self.rnn_cells += [RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        assert X.device == self.device
        assert X.dtype == self.dtype

        seq_len, bs = X.shape[0], X.shape[1]
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device = self.device, dtype = self.dtype, requires_grad = False)
        
        results = []
        X_split = ops.split(X, axis = 0) # X[0] = batches of inputs at time step 0, of shape (bs, input_size)
        h_split = list(ops.split(h0, axis = 0))

        for t in range(seq_len):
            X_t = X_split[t]
            for layer in range(self.num_layers):
                h_layer = h_split[layer]
                X_t = self.rnn_cells[layer](X_t, h_layer)
                h_split[layer] = X_t
            results.append(X_t)
        
        return ops.stack(results, axis = 0), ops.stack(h_split, axis = 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        # initializing learnable weights:
        k = 1 / hidden_size
        bound = k ** 0.5
        W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low = -bound, high = bound, device = device, dtype = dtype, requires_grad = True))
        W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low = -bound, high = bound, device = device, dtype = dtype, requires_grad = True))
        if bias:
            bias_ih = Parameter(init.rand(4 * hidden_size, low = -bound, high = bound, device = device, dtype = dtype, requires_grad = True))
            bias_hh = Parameter(init.rand(4 * hidden_size, low = -bound, high = bound, device = device, dtype = dtype, requires_grad = True))
        else:
            bias_ih = None
            bias_hh = None
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        assert X.device == self.device and X.dtype == self.dtype

        bs = X.shape[0]
        hidden_size = self.hidden_size
        sigmoid = Sigmoid()
        tanh = Tanh()

        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device = self.device, dtype = self.dtype, requires_grad = False)
            c0 = init.zeros(bs, self.hidden_size, device = self.device, dtype = self.dtype, requires_grad = False)
        else:
            h0, c0 = h

        result = ops.matmul(X, self.W_ih) + ops.matmul(h0, self.W_hh)
        if self.bias_ih is not None or self.bias_hh is not None:
            bias_ih = ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4 * hidden_size)), (bs, 4 * hidden_size))
            bias_hh = ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4 * hidden_size)), (bs, 4 * hidden_size))
            result = result + bias_ih + bias_hh
        # result.shape = (bs, 4 * hidden_size)
        # NOTE: use tuple() to wrap around ops.split(...) is important here!
        #       else the test won't pass
        result_all_split = tuple(ops.split(result, axis = 1))
        result_split_4 = []
        for i in range(4):
            result_split_4.append(ops.stack(result_all_split[i * hidden_size : (i + 1) * hidden_size], axis = 1))
        i,f,g,o = result_split_4
        i,f,g,o = sigmoid(i), sigmoid(f), tanh(g), sigmoid(o)
        c_out = ops.multiply(f, c0) + ops.multiply(i, g)
        h_out = ops.multiply(o, tanh(c_out))
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        self.lstm_cells += [LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        assert X.device == self.device
        assert X.dtype == self.dtype

        seq_len, bs = X.shape[0], X.shape[1]
        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device = self.device, dtype = self.dtype, requires_grad = False)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device = self.device, dtype = self.dtype, requires_grad = False)
        else:
            h0, c0 = h

        results = []
        X_split = ops.split(X, axis = 0)
        h_split = list(ops.split(h0, axis = 0))
        c_split = list(ops.split(c0, axis = 0))

        for t in range(seq_len):
            X_t = X_split[t]
            for layer in range(self.num_layers):
                h_layer = h_split[layer]
                c_layer = c_split[layer]
                h_layer, c_layer = self.lstm_cells[layer](X_t, (h_layer, c_layer))
                # NOTE: only the hidden cells are considered "outputs" of the LSTM system.
                X_t = h_layer
                h_split[layer] = h_layer
                c_split[layer] = c_layer
            results.append(X_t)
        
        return ops.stack(results, axis = 0), (ops.stack(h_split, axis = 0), ops.stack(c_split, axis = 0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weights = Parameter(init.randn(num_embeddings, embedding_dim, device = device, dtype = dtype, requires_grad = True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        assert self.device == x.device and self.dtype == x.dtype
        num_embeddings, embedding_dim = self.weights.shape
        seq_len, bs = x.shape

        x = x.reshape((seq_len * bs,))
        # one_hot_vecs.shape = (seq_len * bs, num_embeddings)
        one_hot_vecs = init.one_hot(num_embeddings, x, device = self.device, dtype = self.dtype)
        # result.shape = (seq_len * bs, embedding_dim)
        result = ops.matmul(one_hot_vecs, self.weights)
        return result.reshape((seq_len, bs, embedding_dim))
        ### END YOUR SOLUTION
