from torch import nn
import torch
import torch.nn.functional as F
from user_lib.maxpooling import MaxUnpool2d


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, kernel_size=3, stride=1, padding=1, bias=False)


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, kernel_size=1, stride=1, bias=False)


class Conv3X3Relu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU6(inplace=True)
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Conv1X1Relu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv1X1(in_, out)
        self.activation = torch.nn.ReLU6(inplace=True)
        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Conv1X1Linear(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv1X1(in_, out)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, nn):
        super(Down, self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape


class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool = torch.nn.MaxUnpool2d(2, 2)
        # self.unpool = MaxUnpool2d(2,2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs


class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64, 1)

    def forward(self, fuse_input):
        outputs = F.interpolate(fuse_input, scale_factor=self.scale, mode='bilinear')
        outputs = self.nn(outputs)

        return self.conv(outputs)


class AttentionGates2D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGates2D, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class DeepRidge_Res(nn.Module):

    def __init__(self, num_classes=1000):
        super(DeepRidge_Res, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            Conv1X1Relu(1, 128),
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.down2 = Down(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.down3 = Down(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.down4 = Down(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.down5 = Down(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.up1 = Up(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.up2 = Up(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.up3 = Up(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.up4 = Up(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        self.up5 = Up(torch.nn.Sequential(
            Conv1X1Relu(128, 32),
            Conv3X3Relu(32, 32),
            Conv3X3Relu(32, 32),
            Conv1X1Relu(32, 128),
        ))

        # Attention Gate
        self.att5 = AttentionGates2D(F_g=128, F_l=128, F_int=128)
        self.att4 = AttentionGates2D(F_g=128, F_l=128, F_int=128)
        self.att3 = AttentionGates2D(F_g=128, F_l=128, F_int=128)
        self.att2 = AttentionGates2D(F_g=128, F_l=128, F_int=128)
        self.att1 = AttentionGates2D(F_g=128, F_l=128, F_int=128)

        self.fuse5 = Fuse(Conv1X1Relu(128, 64), scale=16)
        self.fuse4 = Fuse(Conv1X1Relu(128, 64), scale=8)
        self.fuse3 = Fuse(Conv1X1Relu(128, 64), scale=4)
        self.fuse2 = Fuse(Conv1X1Relu(128, 64), scale=2)
        self.fuse1 = Fuse(Conv1X1Relu(128, 64), scale=1)

        self.final = Conv3X3(5, 1)

    def forward(self, inputs, is_train=False):
        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)
        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        atte5 = self.att5(g=up5, x=down5)
        fuse5 = self.fuse5(fuse_input=atte5)
        atte4 = self.att4(g=up4, x=down4)
        fuse4 = self.fuse4(fuse_input=atte4)
        atte3 = self.att3(g=up3, x=down3)
        fuse3 = self.fuse3(fuse_input=atte3)
        atte2 = self.att2(g=up2, x=down2)
        fuse2 = self.fuse2(fuse_input=atte2)
        atte1 = self.att1(g=up1, x=down1)
        fuse1 = self.fuse1(fuse_input=atte1)

        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))

        if is_train:
            return output, fuse5, fuse4, fuse3, fuse2, fuse1
        else:
            return output


class DeepRidge_Inverted_lite(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepRidge_Inverted_lite, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            Conv1X1Relu(1, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down3 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.down4 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.down5 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up1 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.up4 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up5 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        # Attention Gate
        self.att5 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att4 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att3 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att2 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att1 = AttentionGates2D(F_g=64, F_l=64, F_int=64)

        self.fuse5 = Fuse(Conv1X1Relu(64, 64), scale=16)
        self.fuse4 = Fuse(Conv1X1Relu(64, 64), scale=8)
        self.fuse3 = Fuse(Conv1X1Relu(64, 64), scale=4)
        self.fuse2 = Fuse(Conv1X1Relu(64, 64), scale=2)
        self.fuse1 = Fuse(Conv1X1Relu(64, 64), scale=1)

        self.final = torch.nn.Sequential(
            Conv3X3(5, 5),
            Conv1X1(5, 1)
        )

    def forward(self, inputs, is_train=False):
        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)
        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        atte5 = self.att5(g=up5, x=down5)
        fuse5 = self.fuse5(fuse_input=atte5)
        atte4 = self.att4(g=up4, x=down4)
        fuse4 = self.fuse4(fuse_input=atte4)
        atte3 = self.att3(g=up3, x=down3)
        fuse3 = self.fuse3(fuse_input=atte3)
        atte2 = self.att2(g=up2, x=down2)
        fuse2 = self.fuse2(fuse_input=atte2)
        atte1 = self.att1(g=up1, x=down1)
        fuse1 = self.fuse1(fuse_input=atte1)

        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))

        if is_train:
            return output, fuse5, fuse4, fuse3, fuse2, fuse1
        else:
            return output


class DeepRidge_Inverted(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepRidge_Inverted, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            Conv1X1Relu(1, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down3 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.down4 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.down5 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up1 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.up4 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up5 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        # Attention Gate
        self.att5 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att4 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att3 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att2 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att1 = AttentionGates2D(F_g=64, F_l=64, F_int=64)

        self.fuse5 = Fuse(Conv1X1Relu(64, 64), scale=16)
        self.fuse4 = Fuse(Conv1X1Relu(64, 64), scale=8)
        self.fuse3 = Fuse(Conv1X1Relu(64, 64), scale=4)
        self.fuse2 = Fuse(Conv1X1Relu(64, 64), scale=2)
        self.fuse1 = Fuse(Conv1X1Relu(64, 64), scale=1)

        self.final = torch.nn.Sequential(
            Conv3X3(5, 5),
            Conv1X1(5, 1)
        )

    def forward(self, inputs, is_train=False):
        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)
        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        atte5 = self.att5(g=up5, x=down5)
        fuse5 = self.fuse5(fuse_input=atte5)
        atte4 = self.att4(g=up4, x=down4)
        fuse4 = self.fuse4(fuse_input=atte4)
        atte3 = self.att3(g=up3, x=down3)
        fuse3 = self.fuse3(fuse_input=atte3)
        atte2 = self.att2(g=up2, x=down2)
        fuse2 = self.fuse2(fuse_input=atte2)
        atte1 = self.att1(g=up1, x=down1)
        fuse1 = self.fuse1(fuse_input=atte1)

        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))
        if is_train:
            return output, fuse5, fuse4, fuse3, fuse2, fuse1
        else:
            return output


class DeepRidge_Inverted_heavy(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepRidge_Inverted_heavy, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            Conv1X1Relu(1, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down3 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.down4 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.down5 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up1 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.up4 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up5 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        # Attention Gate
        self.att5 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att4 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att3 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att2 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att1 = AttentionGates2D(F_g=64, F_l=64, F_int=64)

        self.fuse5 = Fuse(Conv1X1Relu(64, 64), scale=16)
        self.fuse4 = Fuse(Conv1X1Relu(64, 64), scale=8)
        self.fuse3 = Fuse(Conv1X1Relu(64, 64), scale=4)
        self.fuse2 = Fuse(Conv1X1Relu(64, 64), scale=2)
        self.fuse1 = Fuse(Conv1X1Relu(64, 64), scale=1)

        self.final = torch.nn.Sequential(
            Conv3X3(5, 5),
            Conv1X1(5, 1)
        )

    def forward(self, inputs, is_train=False):
        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)
        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        atte5 = self.att5(g=up5, x=down5)
        fuse5 = self.fuse5(fuse_input=atte5)
        atte4 = self.att4(g=up4, x=down4)
        fuse4 = self.fuse4(fuse_input=atte4)
        atte3 = self.att3(g=up3, x=down3)
        fuse3 = self.fuse3(fuse_input=atte3)
        atte2 = self.att2(g=up2, x=down2)
        fuse2 = self.fuse2(fuse_input=atte2)
        atte1 = self.att1(g=up1, x=down1)
        fuse1 = self.fuse1(fuse_input=atte1)

        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))
        if is_train:
            return output, fuse5, fuse4, fuse3, fuse2, fuse1
        else:
            return output


class DeepRidge_Inverted_deep(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepRidge_Inverted_deep, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            Conv1X1Relu(1, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down3 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.down4 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.down5 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.down6 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.down7 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up1 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.up4 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up5 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up6 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up7 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        # Attention Gate
        # self.att7 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att6 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att5 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att4 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att3 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att2 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att1 = AttentionGates2D(F_g=64, F_l=64, F_int=64)

        # self.fuse7 = Fuse(Conv1X1Relu(64, 64), scale=32)
        self.fuse6 = Fuse(Conv1X1Relu(64, 64), scale=32)
        self.fuse5 = Fuse(Conv1X1Relu(64, 64), scale=16)
        self.fuse4 = Fuse(Conv1X1Relu(64, 64), scale=8)
        self.fuse3 = Fuse(Conv1X1Relu(64, 64), scale=4)
        self.fuse2 = Fuse(Conv1X1Relu(64, 64), scale=2)
        self.fuse1 = Fuse(Conv1X1Relu(64, 64), scale=1)

        self.final = torch.nn.Sequential(
            Conv3X3(6, 6),
            Conv1X1(6, 1)
        )

    def forward(self, inputs, is_train=False):
        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)
        out, down6, indices_6, unpool_shape6 = self.down6(out)
        # out, down7, indices_7, unpool_shape7 = self.down7(out)

        # decoder part
        # out = self.up5(out, indices=indices_7, output_shape=unpool_shape7)
        # atte = self.att7(g=out, x=down7)
        # fuse7 = self.fuse7(fuse_input=atte)
        out = self.up5(out, indices=indices_6, output_shape=unpool_shape6)
        atte = self.att6(g=out, x=down6)
        fuse6 = self.fuse6(fuse_input=atte)
        out = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        atte = self.att5(g=out, x=down5)
        fuse5 = self.fuse5(fuse_input=atte)
        out = self.up4(out, indices=indices_4, output_shape=unpool_shape4)
        atte = self.att4(g=out, x=down4)
        fuse4 = self.fuse4(fuse_input=atte)
        out = self.up3(out, indices=indices_3, output_shape=unpool_shape3)
        atte = self.att3(g=out, x=down3)
        fuse3 = self.fuse3(fuse_input=atte)
        out = self.up2(out, indices=indices_2, output_shape=unpool_shape2)
        atte = self.att2(g=out, x=down2)
        fuse2 = self.fuse2(fuse_input=atte)
        out = self.up1(out, indices=indices_1, output_shape=unpool_shape1)
        atte = self.att1(g=out, x=down1)
        fuse1 = self.fuse1(fuse_input=atte)

        output = self.final(torch.cat([fuse6, fuse5, fuse4, fuse3, fuse2, fuse1], 1))

        if is_train:
            return output, fuse6, fuse5, fuse4, fuse3, fuse2, fuse1
        else:
            return output

class DeepRidge_Inverted_Shoal(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepRidge_Inverted_Shoal, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            Conv1X1Relu(1, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.down3 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.down4 = Down(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        self.up1 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 128),
            Conv3X3Relu(128, 128),
            Conv1X1Linear(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 256),
            Conv3X3Relu(256, 256),
            Conv1X1Linear(256, 64),
        ))

        self.up4 = Up(torch.nn.Sequential(
            Conv1X1Relu(64, 512),
            Conv3X3Relu(512, 512),
            Conv1X1Linear(512, 64),
        ))

        # Attention Gate
        self.att4 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att3 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att2 = AttentionGates2D(F_g=64, F_l=64, F_int=64)
        self.att1 = AttentionGates2D(F_g=64, F_l=64, F_int=64)

        self.fuse4 = Fuse(Conv1X1Relu(64, 64), scale=8)
        self.fuse3 = Fuse(Conv1X1Relu(64, 64), scale=4)
        self.fuse2 = Fuse(Conv1X1Relu(64, 64), scale=2)
        self.fuse1 = Fuse(Conv1X1Relu(64, 64), scale=1)

        self.final = torch.nn.Sequential(
            Conv3X3(4, 4),
            Conv1X1(4, 1)
        )

    def forward(self, inputs, is_train=False):
        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        # decoder part
        up4 = self.up4(out, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        atte4 = self.att4(g=up4, x=down4)
        fuse4 = self.fuse4(fuse_input=atte4)
        atte3 = self.att3(g=up3, x=down3)
        fuse3 = self.fuse3(fuse_input=atte3)
        atte2 = self.att2(g=up2, x=down2)
        fuse2 = self.fuse2(fuse_input=atte2)
        atte1 = self.att1(g=up1, x=down1)
        fuse1 = self.fuse1(fuse_input=atte1)

        output = self.final(torch.cat([fuse4, fuse3, fuse2, fuse1], 1))

        if is_train:
            return output, fuse4, fuse3, fuse2, fuse1
        else:
            return output

if __name__ == '__main__':
    inp = torch.randn(1, 1, 416, 416)
    model = DeepRidge_Res()
    out = model(inp)
    print(out.shape)
    model = DeepRidge_Inverted()
    out = model(inp)
    print(out.shape)
