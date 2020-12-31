import numpy as np

from mytorch import tensor
from mytorch.autograd_engine import Function

def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T), None

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data), None

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod    
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data*np.exp(a.data)), None

class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.sqrt(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        output = tensor.Tensor(grad_output.data/2*(a.data**(-1/2)))
        print(output.shape,a.shape)

        return output, None

class Power(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        if (type(b).__name__ == 'int'):
            b = tensor.Tensor(np.array(b))
       
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data**b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = b.data*(a.data**(b.data-1)) * grad_output.data
        #grad_b = (a.data**b.data+np.log(a.data)) * grad_output.data

        # the order of gradients returned should match the order of the arguments

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        #grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, None
        

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        #if a.data.shape != b.data.shape:
        #    raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape != b.data.shape:
            pass

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

         # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = -np.ones(b.shape) * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        # the order of gradients returned should match the order of the arguments
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data*b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

         # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = b.data * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = a.data * grad_output.data


        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        # the order of gradients returned should match the order of the arguments
        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data/b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

         # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) / b.data * grad_output.data 
        # dL/db = dout/db * dL/dout
        grad_b = - a.data / np.square(b.data) * grad_output.data 

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        # the order of gradients returned should match the order of the arguments
        return grad_a, grad_b

class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None

class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
            a.data.shape[1] != b.data.shape[0]:
            pass

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data@b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

         # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = grad_output.data @ b.data.T
        # dL/db = dout/db * dL/dout
        grad_b = a.data.T @ grad_output.data

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor'):
            pass

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad 

        c = tensor.Tensor(np.where(a.data>0, a.data, 0), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

         # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.where(a.data>0, 1, 0) * grad_output.data 

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), None      



def cross_entropy(predicted, target):
    batch_size, num_classes = predicted.shape

    x = predicted
    y = to_one_hot(target,num_classes)

    C = tensor.Tensor(np.max(x.data,axis=1)) #batchsize x 1

    LSM = (x.T() - ((((x.T() - C).T()).exp()).sum(axis=1)).log() - C).T()
    Loss = (LSM*y).sum() / tensor.Tensor(-batch_size)

    return Loss


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)


#hw2
def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    return ((input_size - kernel_size)//stride) + 1

def get_conv2d_output_size(input_h, input_w, kernel_size, stride):
    return ((input_h - kernel_size)//stride) + 1, ((input_w - kernel_size)//stride) + 1

def upsample(grad_output, stride, input_size, kernel_size):
    batch_size, out_channel, output_size = grad_output.shape
    up = input_size - kernel_size + 1
    upz = np.zeros((batch_size,out_channel,up))
    for i in range(batch_size):
        for j in range(out_channel):
            for k in range(output_size):
                if(k*stride<up):
                    upz[i,j,k*stride] = grad_output.data[i,j,k]

    return upz

def upsample2d(grad_output, stride, input_height, input_width, kernel_size):
    batch_size, out_channel, output_height, output_width = grad_output.shape
    uph = input_height - kernel_size + 1
    upw = input_width - kernel_size + 1

    upz = np.zeros((batch_size,out_channel,uph,upw))

    for i in range(batch_size):
        for j in range(out_channel):
            for k in range(output_height):
                for m in range(output_width):
                    if(k*stride<uph or m*stride<upw):
                        upz[i,j,k*stride,m*stride] = grad_output.data[i,j,k,m]

    return upz
    

class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        
        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_size))

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for i in range(batch_size):
            for j in range(out_channel):
                for k in range(output_size):
                    sum = 0 
                    for l in range(in_channel):
                        sum += x.data[i,l,k*stride:k*stride+kernel_size].dot(weight.data[j,l])
                    out[i,j,k] = sum+bias.data[j]

        # TODO: Put output into tensor with correct settings and return 
        return tensor.Tensor(out, requires_grad = True, is_leaf = False)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
        upsample_size = get_conv1d_output_size(input_size, kernel_size, 1)

        #upsample
        if(stride>1):
            z = upsample(grad_output, stride, input_size, kernel_size)
        else:
            z = grad_output.data

        grad_x = np.zeros((batch_size, in_channel, input_size))
        grad_w = np.zeros((out_channel, in_channel, kernel_size))
        grad_b = np.zeros((out_channel,))

        zpad = np.zeros((batch_size,out_channel,upsample_size+2*kernel_size-2))
        wflip = np.flip(weight.data,2)

        for i in range(batch_size):
            for j in range(out_channel):
                zpad[i,j,kernel_size-1:kernel_size+upsample_size-1] = z[i,j]
                #x
                for k in range(upsample_size+kernel_size-1):
                    for l in range(in_channel):
                        grad_x[i,l,k] += zpad[i,j,k:k+kernel_size].dot(wflip[j,l])

                #w
                for k in range(kernel_size):
                    for l in range(in_channel):
                        grad_w[j,l,k] += x.data[i,l,k:k+input_size-kernel_size+1].dot(z[i,j])
                
                #b
                for k in range(output_size):
                    grad_b[j] += grad_output.data[i,j,k]
                
        return tensor.Tensor(grad_x), tensor.Tensor(grad_w), tensor.Tensor(grad_b)



class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad),
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad),


class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        ctx.save_for_backward(x)
        ctx.indices = indices

        return tensor.Tensor(x.data[indices], requires_grad=x.requires_grad, is_leaf = not x.requires_grad)

    @staticmethod
    def backward(ctx,grad_output):
        a = ctx.saved_tensors[0]
        out = np.zeros(a.shape)
        out[ctx.indices] = grad_output.data
        return tensor.Tensor(out), None

class Cat(Function):
    @staticmethod
    def forward(ctx,*args):
        '''
        Args:
            args (list): [*seq, dim] 
        
        NOTE: seq (list of tensors) contains the tensors that we wish to concatenate while dim (int) is the dimension along which we want to concatenate 
        '''
        *seq, dim = args
        seq = seq[0]
        ctx.list = seq
        ctx.dim = dim
        
        requires_grad = False
        for t in seq:
            if t is None:
                continue
            requires_grad = requires_grad or t.requires_grad

        seq = [t.data for t in seq if t is not None]
        output = tensor.Tensor(np.concatenate(seq,dim), requires_grad=requires_grad,is_leaf=not requires_grad)

        return output
            
    @staticmethod
    def backward(ctx,grad_output):
        seq = ctx.list
        dim = ctx.dim

        sizes = []
        for i in seq:
            sizes.append(i.shape[dim])
        for idx,i in enumerate(sizes):
            if idx>0:
                sizes[idx] = sizes[idx] + sizes[idx-1]

        grad_output = np.split(grad_output.data,sizes,dim)        
        output = ()
        for i in grad_output:
            output += (tensor.Tensor(i),)
        
        return output

class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).

                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """
        if not type(x).__name__ == 'Tensor':
            raise Exception("Only dropout for tensors is supported")

        mask = np.random.binomial(n=1, p=1-p, size=x.shape)
        mask = mask/(1-p)
        mask = tensor.Tensor(mask, requires_grad=False)

        ctx.save_for_backward(mask)

        if not is_train:
            return x
        else:
            return x*mask

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data * mask.data), None


class Conv2d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_height, input_width = x.shape
        out_channel, _, kernel_size, _ = weight.shape
        
        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_height, output_width = get_conv2d_output_size(input_height, input_width, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_height, output_width))

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for i in range(batch_size):
            for j in range(out_channel):
                for k in range(output_height):
                    for m in range(output_width):
                        total = 0 
                        for l in range(in_channel):
                            total += x.data[i,l,k*stride:k*stride+kernel_size,m*stride:m*stride+kernel_size].flatten().dot(weight.data[j,l].flatten())
                        out[i,j,k,m] = total+bias.data[j]

        # TODO: Put output into tensor with correct settings and return 
        return tensor.Tensor(out, requires_grad = True, is_leaf = False)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        batch_size, in_channel, input_height, input_width = x.shape
        out_channel, _, kernel_size, _ = weight.shape
        output_height, output_width = get_conv2d_output_size(input_height, input_width, kernel_size, stride)
        upsample_h, upsample_w = get_conv2d_output_size(input_height, input_width, kernel_size, 1)

        #upsample
        if(stride>1):
            z = upsample2d(grad_output, stride, input_height, input_width, kernel_size)
        else:
            z = grad_output.data

        grad_x = np.zeros((batch_size, in_channel, input_height, input_width))
        grad_w = np.zeros((out_channel, in_channel, kernel_size, kernel_size))
        grad_b = np.zeros((out_channel,))

        zpad = np.zeros((batch_size,out_channel,upsample_h+2*kernel_size-2,upsample_w+2*kernel_size-2))
        wflip = np.flip(weight.data,(2,3))

        for i in range(batch_size):
            for j in range(out_channel):
                zpad[i,j,kernel_size-1:kernel_size+upsample_h-1,kernel_size-1:kernel_size+upsample_w-1] = z[i,j]
                #x
                for k in range(upsample_h+kernel_size-1):
                    for m in range(upsample_w+kernel_size-1):
                        for l in range(in_channel):
                            grad_x[i,l,k,m] += zpad[i,j,k:k+kernel_size,m:m+kernel_size].flatten().dot(wflip[j,l].flatten())

                #w
                for k in range(kernel_size):
                    for m in range(kernel_size):
                        for l in range(in_channel):
                            grad_w[j,l,k,m] += x.data[i,l,k:k+input_height-kernel_size+1,m:m+input_width-kernel_size+1].flatten().dot(z[i,j].flatten())
                
                #b
                for k in range(output_height):
                    for m in range(output_width):
                        grad_b[j] += grad_output.data[i,j,k,m]
                
        return tensor.Tensor(grad_x), tensor.Tensor(grad_w), tensor.Tensor(grad_b)

class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
            kernel_size (int): the size of the window to take a max over
            stride (int): the stride of the window. Default value is kernel_size.
        Returns:
            y (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        batch_size, in_channel, input_height, input_width = x.shape
        if stride is None:
            stride = 1

        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x)
        ctx.stride = stride
        ctx.kernel_size = kernel_size

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_height, output_width = get_conv2d_output_size(input_height, input_width, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, in_channel, output_height, output_width))

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for i in range(batch_size):
            for j in range(in_channel):
                for k in range(output_height):
                    for m in range(output_width):
                        total = np.max(x.data[i,j,k*stride:k*stride+kernel_size,m*stride:m*stride+kernel_size])
                        out[i,j,k,m] = total

        # TODO: Put output into tensor with correct settings and return 
        return tensor.Tensor(out, requires_grad = True, is_leaf = False)



    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to match forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
        x = ctx.saved_tensors[0]
        stride = ctx.stride
        kernel_size = ctx.kernel_size
        batch_size, in_channel, input_height, input_width = x.shape

        upsample_h, upsample_w = get_conv2d_output_size(input_height, input_width, kernel_size, 1)

        #upsample
        if(stride>1):
            z = upsample2d(grad_output, stride, input_height, input_width, kernel_size)
        else:
            z = grad_output.data

        grad_x = np.zeros((batch_size, in_channel, input_height, input_width))

        zpad = np.zeros((batch_size,in_channel,upsample_h+2*kernel_size-2,upsample_w+2*kernel_size-2))

        for i in range(batch_size):
            for j in range(in_channel):
                zpad[i,j,kernel_size-1:kernel_size+upsample_h-1,kernel_size-1:kernel_size+upsample_w-1] = z[i,j]
                for k in range(upsample_h+kernel_size-1):
                    for m in range(upsample_w+kernel_size-1):
                        array = x.data[i,j,k:k+kernel_size,m:m+kernel_size].flatten()
                        idx = np.argmax(array)
                        grad_x[i,j,k,m] = zpad[i,j,k:k+kernel_size,m:m+kernel_size].flatten()[idx]

        return tensor.Tensor(grad_x), None, None


class AvgPool2d(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None):
        """
        Args:
            x (Tensor): (batch_size, in_channel, input_height, input_width)
            kernel_size (int): the size of the window to take a mean over
            stride (int): the stride of the window. Default value is kernel_size.
        Returns:
            y (Tensor): (batch_size, out_channel, output_height, output_width)
        """
        batch_size, in_channel, input_height, input_width = x.shape
        if stride is None:
            stride = 1

        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x)
        ctx.stride = stride
        ctx.kernel_size = kernel_size

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_height, output_width = get_conv2d_output_size(input_height, input_width, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, in_channel, output_height, output_width))

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for i in range(batch_size):
            for j in range(in_channel):
                for k in range(output_height):
                    for m in range(output_width):
                        total = np.mean(x.data[i,j,k*stride:k*stride+kernel_size,m*stride:m*stride+kernel_size])
                        out[i,j,k,m] = total

        # TODO: Put output into tensor with correct settings and return 
        return tensor.Tensor(out, requires_grad = True, is_leaf = False)


    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            ctx (autograd_engine.ContextManager): for receiving objects you saved in this Function's forward
            grad_output (Tensor): (batch_size, out_channel, output_height, output_width)
                                  grad. of loss w.r.t. output of this function

        Returns:
            dx, None, None (tuple(Tensor, None, None)): Gradients of loss w.r.t. input
                                                        `None`s are to match forward's num input args
                                                        (This is just a suggestion; may depend on how
                                                         you've written `autograd_engine.py`)
        """
