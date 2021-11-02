import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    CAE with linear latent layer.
    """
    def __init__(self, Z, C, H, W, hid=[4, 8, 16], stride=1, activation=nn.ReLU()):
        super().__init__()
        self.Z = Z
        self.C = C
        self.H = H
        self.W = W
        self.hid = hid
        self.stride = stride
        self.activation = activation
        self._plan_model()
        self._build_model()
    
    def _plan_model(self):
        def conv_dim(in_dim, kernel, stride, padding, dilation):
            out_dim = int((in_dim + 2*padding - dilation*(kernel - 1) - 1)/stride + 1)
            return out_dim
        
        def deconv_dim(in_dim, kernel, stride, padding, output_padding, dilation):
            out_dim = (in_dim - 1)*stride - 2*padding + dilation*(kernel - 1) + output_padding + 1
            return out_dim
        
        def get_kernel(dim):
            if self.stride == 1:
                return 3
            elif self.stride == 2:
                if dim%2==0:
                    return 4
                else:
                    return 3
            else:
                raise ValueError('Only strides of 1 or 2 are supported.')
        
        # Encoder
        self.kh = [] 
        self.kw = []
        h, w = self.H, self.W
        for i in range(len(self.hid)):
            kh, kw = get_kernel(h), get_kernel(w)
            self.kh.append(kh), self.kw.append(kw)
            h, w = conv_dim(h, kh, self.stride, 0, 1), conv_dim(w, kw, self.stride, 0, 1)
        
        # Linear
        self.l = self.hid[-1]*h*w
        assert(self.l != 0)
        
        # Decoder
        for i in range(1, len(self.hid)+1):
            h, w = deconv_dim(h, self.kh[-i], self.stride, 0, 0, 1), deconv_dim(w, self.kw[-i], self.stride, 0, 0, 1)
        
        # Ensuring Input Dimensions Equals Output Dimensions
        assert (self.H == h) and (self.W == w)
        
    def _build_model(self):
        # Encoder
        modules = []
        in_channels = self.C
        for i in range(len(self.hid)):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hid[i], (self.kh[i], self.kw[i]), stride=self.stride, padding=0, dilation=1),
                    nn.BatchNorm2d(self.hid[i]),
                    self.activation
                )
            )
            in_channels = self.hid[i]
        self.encoder = nn.Sequential(*modules)
        
        # Linear
        self.fc1 = nn.Linear(self.l, self.Z)
        self.fc2 = nn.Linear(self.Z, self.l)
        
        # Decoder
        modules = []
        out_channels = self.C
        for i in range(len(self.hid)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hid[i], out_channels, (self.kh[i], self.kw[i]), stride=self.stride, padding=0, output_padding=0, dilation=1),
                    nn.BatchNorm2d(out_channels),
                    self.activation
                )
            )
            out_channels = self.hid[i]
        self.decoder = nn.Sequential(*modules[::-1])
    
    def forward(self, x):
        x = self.encoder(x)
        shape = x.shape
        x = x.view(shape[0], -1) # flatten
        x = self.fc1(x) # linear to Z # Note: no activation bc it increases error and makes Z less interpretable
        x = self.activation(self.fc2(x)) # Z to linear
        x = x.view(*shape) # unflatten
        x = self.decoder(x)       
        return x


# class ConvAutoencoder(nn.Module):
#     """
#     CAE w/o linear latent layer
#     """
#     def __init__(self):
#         super().__init__()
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(16, 4, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2)
#         )
        
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(4, 16, 2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, 2, stride=2),
#             nn.Sigmoid()
#         )


#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)            
#         return x