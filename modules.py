import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    CAE with linear latent layer.
    """
    def __init__(self, Z, C, H, W, stride=1, activation=nn.ReLU()):
        super().__init__()
        self.Z = Z
        self.C = C
        self.H = H
        self.W = W
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
        
        self.hk1, self.wk1 = get_kernel(self.H), get_kernel(self.W)
        self.h1, self.w1 = conv_dim(self.H, self.hk1, self.stride, 0, 1), conv_dim(self.W, self.wk1, self.stride, 0, 1)
        self.hk2, self.wk2 = get_kernel(self.h1), get_kernel(self.w1)
        self.h2, self.w2 = conv_dim(self.h1, self.hk2, self.stride, 0, 1), conv_dim(self.w1, self.wk2, self.stride, 0, 1)
        self.hk3, self.wk3 = get_kernel(self.h2), get_kernel(self.w2)
        self.h3, self.w3 = conv_dim(self.h2, self.hk3, self.stride, 0, 1), conv_dim(self.w2, self.wk3, self.stride, 0, 1)
        
        self.l = 16*self.h3*self.w3
        
        self.h1_, self.w1_ = deconv_dim(self.h3, self.hk3, self.stride, 0, 0, 1), deconv_dim(self.w3, self.wk3, self.stride, 0, 0, 1)
        self.h2_, self.w2_ = deconv_dim(self.h1_, self.hk2, self.stride, 0, 0, 1), deconv_dim(self.w1_, self.wk2, self.stride, 0, 0, 1)
        self.h3_, self.w3_ = deconv_dim(self.h2_, self.hk1, self.stride, 0, 0, 1), deconv_dim(self.w2_, self.wk1, self.stride, 0, 0, 1)
        
        assert (self.H == self.h3_) and (self.W == self.w3_)
        
    def _build_model(self):
        self.encoder = nn.Sequential(
            # BatchNorm2d?
            nn.Conv2d(self.C, 4, (self.hk1, self.wk1), stride=self.stride, padding=0, dilation=1),
            self.activation,
            nn.Conv2d(4, 8, (self.hk2, self.wk2), stride=self.stride, padding=0, dilation=1),
            self.activation,
            nn.Conv2d(8, 16, (self.hk3, self.wk3), stride=self.stride, padding=0, dilation=1)
        )
        
        # Dropout?
        self.fc1 = nn.Linear(self.l, self.Z)
        self.fc2 = nn.Linear(self.Z, self.l)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, (self.hk3, self.wk3), stride=self.stride, padding=0, output_padding=0, dilation=1),
            self.activation,
            nn.ConvTranspose2d(8, 4, (self.hk2, self.wk2), stride=self.stride, padding=0, output_padding=0, dilation=1),
            self.activation,
            nn.ConvTranspose2d(4, self.C, (self.hk1, self.wk1), stride=self.stride, padding=0, output_padding=0, dilation=1)
        )
    
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