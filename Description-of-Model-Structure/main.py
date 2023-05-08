import torch
from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d,
                 use_bias=True, scale_factor=1):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=use_bias),

            norm_layer(in_channels * scale_factor),

            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=use_bias),
        )

    def forward(self, x):
        return self.conv(x)


class ResnetBlock(nn.Module):

    def __init__(self, dim, norm_layer, dropout_rate):
        super(ResnetBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1)]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_layer(dim)
        ]

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.model(x)


class MyResnetBlock(nn.Module):
    def __init__(self, ic, oc, norm_layer, dropout_rate):
        super(MyResnetBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1)]
        conv_block += [
            SeparableConv2d(in_channels=ic, out_channels=oc,
                            kernel_size=3, padding=0, stride=1),
            norm_layer(oc),
            nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [
            SeparableConv2d(in_channels=oc, out_channels=ic,
                            kernel_size=3, padding=0, stride=1),
            norm_layer(ic)
        ]

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.model(x)


class CycleGANGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_blocks=9):
        super(CycleGANGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2

        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2,
                                kernel_size=3, stride=2,
                                padding=1,
                                bias=True),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        # add ResNet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, dropout_rate=dropout_rate)]

        # add upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class MyCycleGANGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, config=None, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_blocks=9):
        super(MyCycleGANGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, config[0], kernel_size=7, padding=0, bias=True),
                 norm_layer(config[0]),
                 nn.ReLU(True)]

        n_downsampling = 2
        # add DownSampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            ic = config[i]
            oc = config[i + 1]
            model += [nn.Conv2d(ic * mult, oc * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      norm_layer(ic * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        ic = config[2]
        # add MyResnetBlock
        for i in range(n_blocks):
            offset = i // 3
            oc = config[offset + 3]
            model += [MyResnetBlock(ic * mult, oc * mult, norm_layer=norm_layer,
                                    dropout_rate=dropout_rate)]

        offset = 6
        # add UpSampling layers
        for i in range(n_downsampling):
            oc = config[offset + i]
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ic * mult, int(oc * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      norm_layer(int(oc * mult / 2)),
                      nn.ReLU(True)]
            ic = oc

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ic, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    # Dummy_input is used to export the onnx file, which can be opened by neutron to view the network model.
    # The neutron tool is located at: https://netron.app/.
    dummy_input = torch.randn((1, 3, 256, 256), dtype=torch.float)

    # Teacher network, which is original CycleGAN generator
    Teacher_Network = CycleGANGenerator(ngf=64)
    print(Teacher_Network)
    torch.onnx.export(Teacher_Network, dummy_input, "Teacher_Network.onnx", verbose=False,
                      input_names=["input"], output_names=["output"])

    # Teacher_Network* (replace conv with separable conv in resnet block)
    Teacher_Network_Separable_Conv = MyCycleGANGenerator(config=[64, 64, 64, 64, 64, 64, 64, 64])
    print(Teacher_Network_Separable_Conv)
    torch.onnx.export(Teacher_Network_Separable_Conv, dummy_input, "Teacher_Network_Separable_Conv.onnx", verbose=False,
                      input_names=["input"], output_names=["output"])

    # Student network
    Student_Network = MyCycleGANGenerator(config=[32, 32, 32, 32, 32, 32, 32, 32])
    print(Student_Network)
    torch.onnx.export(Student_Network, dummy_input, "Student_Network.onnx", verbose=False,
                      input_names=["input"], output_names=["output"])

    # Sub_Network1
    Sub_Network1 = MyCycleGANGenerator(config=[16, 16, 16, 16, 16, 16, 16, 16])
    print(Sub_Network1)
    torch.onnx.export(Sub_Network1, dummy_input, "Sub_Network1.onnx", verbose=False,
                      input_names=["input"], output_names=["output"])

    # Sub_Network2
    Sub_Network2 = MyCycleGANGenerator(config=[24, 16, 16, 16, 16, 16, 16, 16])
    print(Sub_Network2)
    torch.onnx.export(Sub_Network2, dummy_input, "Sub_Network2.onnx", verbose=False,
                      input_names=["input"], output_names=["output"])

    # Sub_Network3
    Sub_Network3 = MyCycleGANGenerator(config=[24, 24, 16, 16, 16, 16, 16, 16])
    print(Sub_Network3)
    torch.onnx.export(Sub_Network3, dummy_input, "Sub_Network3.onnx", verbose=False,
                      input_names=["input"], output_names=["output"])
