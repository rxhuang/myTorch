import numpy as np
from mytorch import tensor
from mytorch.tensor import Tensor
from mytorch.nn.module import Module
from mytorch.nn.activations import Tanh, ReLU, Sigmoid
from mytorch.nn.util import PackedSequence, pack_sequence, unpack_sequence


class RNNUnit(Module):
    '''
    This class defines a single RNN Unit block.

    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 
    '''

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh' ):
        
        super(RNNUnit,self).__init__()
        
        # Initializing parameters
        self.weight_ih = Tensor(np.random.randn(hidden_size,input_size), requires_grad=True, is_parameter=True)
        self.bias_ih   = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)
        self.weight_hh = Tensor(np.random.randn(hidden_size,hidden_size), requires_grad=True, is_parameter=True)
        self.bias_hh   = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.hidden_size = hidden_size
        
        # Setting the Activation Unit
        if nonlinearity == 'tanh':
            self.act = Tanh()
        elif nonlinearity == 'relu':
            self.act = ReLU()

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)

    def forward(self, input, hidden = None):
        '''
        Args:
            input (Tensor): (effective_batch_size,input_size)
            hidden (Tensor,None): (effective_batch_size,hidden_size)
        Return:
            Tensor: (effective_batch_size,hidden_size)
        '''
        
        # TODO: INSTRUCTIONS
        # Perform matrix operations to construct the intermediary value from input and hidden tensors
        # Apply the activation on the resultant
        # Remeber to handle the case when hidden = None. Construct a tensor of appropriate size, filled with 0s to use as the hidden.
        effective_batch_size, input_size = input.shape

        if hidden is None:
            hidden = Tensor(np.zeros((effective_batch_size, self.hidden_size)))

        output = input@(self.weight_ih.T())+self.bias_ih+hidden@(self.weight_hh.T())+self.bias_hh
        return Tanh().forward(output)


class TimeIterator(Module):
    '''
    For a given input this class iterates through time by processing the entire
    seqeunce of timesteps. Can be thought to represent a single layer for a 
    given basic unit which is applied at each time step.
    
    Args:
        basic_unit (Class): RNNUnit or GRUUnit. This class is used to instantiate the unit that will be used to process the inputs
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 

    '''

    def __init__(self, basic_unit, input_size, hidden_size, nonlinearity = 'tanh' ):
        super(TimeIterator,self).__init__()

        # basic_unit can either be RNNUnit or GRUUnit
        self.unit = basic_unit(input_size,hidden_size,nonlinearity)  
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)
    
    def forward(self,input,hidden = None):
        
        '''
        NOTE: Please get a good grasp on util.PackedSequence before attempting this.

        Args:
            input (PackedSequence): input.data is tensor of shape ( total number of timesteps (sum) across all samples in the batch, input_size)
            hidden (Tensor, None): (batch_size, hidden_size)
        Returns:
            PackedSequence: ( total number of timesteps (sum) across all samples in the batch, hidden_size)
            Tensor (batch_size,hidden_size): This is the hidden generated by the last time step for each sample joined together. Samples are ordered in descending order based on number of timesteps. This is a slight deviation from PyTorch.
        '''
        
        # Resolve the PackedSequence into its components
        data, sorted_indices, batch_sizes = input
        b = batch_sizes.copy()
        counter = 0
        for i in range(len(b)):
            counter = counter + b[i]
            b[i] = counter
        b = [int(x) for x in b]
        
        #Get input data
        output_ps = []
        for i in range(len(b)):
            if i==0:
                temp_data = data[:b[i]]
            else:
                temp_data = data[b[i-1]:b[i]]
            #temp_data = [i.unsqueeze(1).T() for i in temp_data]

            #get part of hidden needed
            if hidden is not None:
                hidden = hidden[0:b[i]-b[i-1]]

            hidden = self.unit(temp_data, hidden)
            output_ps.append(hidden)

        #cat hiddens
        output_ps = tensor.cat(output_ps, 0)
        input.data = output_ps

        #get final hidden
        unpack = unpack_sequence(input)
        output_hid = []
        for t in unpack:
            output_hid.append(t[-1].unsqueeze(0))

        output_hid = [output_hid[i] for i in input.sorted_indices]
        output_hid = tensor.cat(output_hid, 0)
        
        return input,output_hid



class RNN(TimeIterator):
    '''
    Child class for TimeIterator which appropriately initializes the parent class to construct an RNN.
    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 
    '''

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh' ):
        # TODO: Properly Initialize the RNN class
        super(RNN,self).__init__(RNNUnit, input_size, hidden_size, nonlinearity)

