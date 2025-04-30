import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=64, depth=5):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input layer
        self.inc = DoubleConv(in_channels, base_filters)
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_features = base_filters
        for i in range(depth-1):
            out_features = in_features * 2
            self.encoder.append(Down(in_features, out_features))
            in_features = out_features
            
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(depth-1):
            self.decoder.append(Up(in_features, in_features // 2))
            in_features = in_features // 2
            
        # Output layer
        self.outc = OutConv(base_filters, out_channels)
        
        # Store skip connections
        self.skip_connections = []

    def forward(self, x):
        # Reset skip connections
        self.skip_connections = []
        
        # Initial convolution
        features = self.inc(x)
        self.skip_connections.append(features)
        
        # Encoder path
        for encoder_block in self.encoder:
            features = encoder_block(features)
            self.skip_connections.append(features)
            
        # Remove the last skip connection
        x = self.skip_connections.pop()
        
        # Decoder path
        for decoder_block in self.decoder:
            skip = self.skip_connections.pop()
            x = decoder_block(x, skip)
            
        # Output
        logits = self.outc(x)
        
        return logits

    def get_features(self):
        return self.skip_connections