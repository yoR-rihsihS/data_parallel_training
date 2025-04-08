import torch
import torch.nn as nn

class Subsampling(nn.Module):
    def __init__(self, input_channels, kernel_size=(2, 2), stride=(2, 2)):
        super(Subsampling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg_pool = nn.AvgPool2d(self.kernel_size, self.stride)
        self.multiply_coefficient_add_bias = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=input_channels,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.kernel_size[0] * self.kernel_size[1] * x
        x = self.multiply_coefficient_add_bias(x)
        x = self.sigmoid(x)
        return x

class ScaledHyperbolicTangent(nn.Module):
    def __init__(self, S=2/3, A=1.7159):
        super(ScaledHyperbolicTangent, self).__init__()
        self.S = S
        self.A = A
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.S * x
        x = self.tanh(x)
        x = self.A * x
        return x
    
class EuclideanRadialBasisFunction(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize centers (-1 or 1 values)
        self.centers = nn.Parameter(
            2 * torch.randint(0, 2, (out_features, in_features), dtype=torch.float32) - 1
        )
        
    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        centers_expanded = self.centers.unsqueeze(0)
        squared_distances = torch.sum((x_expanded - centers_expanded) ** 2, dim=-1)
        return squared_distances
    
class ConvolutionLayer(nn.Module):
    def __init__(self, kernel_size, stride, padding, input_channels, output_channels, type='normal'):
        super(ConvolutionLayer, self).__init__()

        if type == 'normal':
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif type == 'depthwise-separable':
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=input_channels),
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise ValueError("type should be normal for normal convolution layer and depthwise-separable for depthwise separable convolution layer")

    def forward(self, x):
        x = self.conv(x)
        return x
    
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, conv_type='normal'):
        super(LeNet5, self).__init__()
        
        self.C1 = ConvolutionLayer(input_channels=1, output_channels=6, kernel_size=5, stride=1, padding=0, type=conv_type)
        self.S2 = Subsampling(input_channels=6, kernel_size=(2, 2), stride=(2, 2))
        self.C3 = ConvolutionLayer(input_channels=6, output_channels=16, kernel_size=5, stride=1, padding=0, type=conv_type)
        self.S4 = Subsampling(input_channels=16, kernel_size=(2, 2), stride=(2, 2))
        self.C5 = ConvolutionLayer(input_channels=16, output_channels=120, kernel_size=5, stride=1, padding=0, type=conv_type)
        self.F6 = nn.Linear(120, 84)
        self.output_layer = EuclideanRadialBasisFunction(in_features=84, out_features=num_classes)

        self.squashing_function = ScaledHyperbolicTangent()
        
    def forward(self, x):
        x = self.squashing_function(self.C1(x))
        x = self.S2(x)
        x = self.squashing_function(self.C3(x))
        x = self.S4(x)
        x = self.squashing_function(self.C5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.squashing_function(self.F6(x))
        x = self.output_layer(x)
        return x
    

