from torch.utils.data import Subset

def get_subset(data, labels):
    indices = []
    for i in range(len(data)):
        label = data[i][1]
        if label not in labels:
            indices.append(i)
    return Subset(data, indices)

def conv_dim(in_dim, kernel, stride, padding, dilation):
    out_dim = int((in_dim + 2*padding - dilation*(kernel - 1) - 1)/stride + 1)
    return out_dim

def deconv_dim(in_dim, kernel, stride, padding, output_padding, dilation):
    out_dim = (in_dim - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1
    return out_dim