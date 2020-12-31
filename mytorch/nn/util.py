from mytorch import tensor
import numpy as np

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))


def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K) where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    
    # TODO: INSTRUCTIONS
    # Find the sorted indices based on number of time steps in each sample
    # Extract slices from each sample and properly order them for the construction of the packed tensor. __getitem__ you defined for Tensor class will come in handy
    # Use the tensor.cat function to create a single tensor from the re-ordered segements
    # Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.
    lengths = []
    for t in sequence:
        lengths.append(t.shape[0])
    sorted_indices = np.argsort(lengths)[::-1]
    batch_sizes = np.zeros(max(lengths))

    #initialize list of lists
    packs = []
    for i in range(max(lengths)):
        packs.append([])
    #slice each tensor and append them to appropriate lists
    for i in range(len(sequence)):
        t = sequence[sorted_indices[i]]
        for j in range(len(t)):
            packs[j].append(t[j:j+1])
            batch_sizes[j]+=1


    #concat the slices
    cat_packs = []
    for i in packs:
        cat_packs += i
    #create the tensor
    output = tensor.cat(cat_packs, 0)

    return PackedSequence(output,sorted_indices,batch_sizes)
    


def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    
    # TODO: INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list (assuming the tensors were sorted in a descending fashion based on number of timesteps)
    # Construct these individual tensors using tensor.cat
    # Re-arrange this list of tensor based on ps.sorted_indices

    b = ps.batch_sizes.copy()
    counter = 0
    for i in range(len(b)):
        counter = counter + b[i]
        b[i] = counter
    b = [int(x) for x in b]

    #split data by time
    packs = []
    for i in range(len(b)):
        if i>0:
            packs.append(ps.data[b[i-1]:b[i]])
        else:
            packs.append(ps.data[:b[i]])
    
    #split the data
    output = []
    for i in range(len(packs[0])):
        temp = []
        for j in range(len(packs)):
            if len(packs[j])>i:
                temp.append(packs[j][i])
        temp = [i.unsqueeze(0) for i in temp]
        output.append(tensor.cat(temp, 0))
    
    #sort it 
    output = [output[i] for i in ps.sorted_indices]
    return output


    


