from torch.utils.data import Subset

def get_subset(data, labels):
    indices = []
    for i in range(len(data)):
        label = data[i][1]
        if label in labels:
            indices.append(i)
    return Subset(data, indices)

def print_dec(func):
    def wrapper(*args, **kwargs):
        print('Output Dimension:', func(*args, **kwargs))
        return func(*args, **kwargs)
    return wrapper

@print_dec
def conv_dim(in_dim, kernel, stride, padding, dilation):
    out_dim = int((in_dim + 2*padding - dilation*(kernel - 1) - 1)/stride + 1)
    return out_dim

@print_dec
def deconv_dim(in_dim, kernel, stride, padding, output_padding, dilation):
    out_dim = (in_dim - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1
    return out_dim

def get_acc(mse_dict, thresholds, positive_class):
    tpr = []
    tnr = []
    for t in thresholds:
        true_pos = 0
        true_neg = 0
        total_pos = 0
        total_neg = 0
        for k, v in mse_dict.items():
            for mse in v:
                if k in positive_class:
                    total_pos += 1
                    if mse < t:
                        true_pos += 1
                else:
                    total_neg += 1
                    if mse > t:
                        true_neg += 1
        tpr.append(true_pos/total_pos)
        tnr.append(true_neg/total_neg)
    return tpr, tnr