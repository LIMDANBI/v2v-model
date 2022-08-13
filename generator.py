import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        # 채널(channel) 크기는 그대로 유지
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0] # 입력 이미지의 채널 수: 3

        # 초기 Convolution layer
        out_channels = 64
        model = [nn.ReflectionPad2d(channels)]
        model.append(nn.Conv2d(channels, out_channels, kernel_size=7))
        model.append(nn.InstanceNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        in_channels = out_channels

        # Downsampling
        for _ in range(2):
            out_channels *= 2
            model.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)) # 너비와 높이가 2배씩 감소
            model.append(nn.InstanceNorm2d(out_channels))
            model.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        # 출력: [256 X (4배 감소한 높이) X (4배 감소한 너비)]

        # 인코더와 디코더의 중간에서 Residual Blocks 사용 (차원 유지)
        for _ in range(num_residual_blocks):
            model.append(ResidualBlock(out_channels))

        # Upsampling
        for _ in range(2):
            out_channels //= 2
            model.append(nn.Upsample(scale_factor=2)) # 너비와 높이가 2배씩 증가
            model.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)) # 너비와 높이는 그대로
            model.append(nn.InstanceNorm2d(out_channels))
            model.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        # 출력: [256 X (4배 증가한 높이) X (4배 증가한 너비)]

        # 출력 Convolution Block layer
        model.append(nn.ReflectionPad2d(channels))
        model.append(nn.Conv2d(out_channels, channels, kernel_size=7))
        model.append(nn.Tanh())

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)