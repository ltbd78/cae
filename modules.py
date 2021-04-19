import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    CAE with linear latent layer.
    """
    def __init__(self, Z):
        super().__init__()
        self.activation = nn.ReLU()
        
        self.encoder = nn.Sequential(
            # BatchNorm2d?
            nn.Conv2d(1, 8, 3, stride=1, padding=0, dilation=1),
            self.activation,
            nn.Conv2d(8, 16, 3, stride=1, padding=0, dilation=1),
            self.activation,
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1),
        )
        
        self.fc1 = nn.Linear(16*12*12, Z)
        # Dropout?
        self.fc2 = nn.Linear(Z, 16*12*12)
            
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0, output_padding=0, dilation=1),
            self.activation,
            nn.ConvTranspose2d(8, 4, 3, stride=1, padding=0, output_padding=0, dilation=1),
            self.activation,
            nn.ConvTranspose2d(4, 1, 3, stride=1, padding=0, output_padding=0, dilation=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        shape = x.shape
        x = x.view(shape[0], -1) # flatten
        x = self.fc1(x)
        x = self.activation(self.fc2(x))
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