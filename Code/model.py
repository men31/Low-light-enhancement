import torch
import pandas as pd


# pytorch image size = N X C X H X W
# N = number of data, C = Channel, H = height, W = width

class CABlock(torch.nn.Module):
    def __init__(self, channels, **kwargs):
        super(CABlock, self).__init__()
        # self.GAP = torch.nn.AdaptiveAvgPool3d((channels, 1, 1))
        self.conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, **kwargs)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = x.clone()
        # x1 = self.GAP(x1)
        x1 = torch.mean(x1, dim=[2, 3], keepdim=True)
        x1 = self.relu(self.conv(x1))
        x1 = self.sigmoid(self.conv(x1))
        return x * x1

class PABlock(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(PABlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=1, **kwargs)
        self.conv_2 = torch.nn.Conv2d(in_channels=1, out_channels=1, **kwargs)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = x.clone()
        x1 = self.relu(self.conv_1(x1))
        x1 = self.sigmoid(self.conv_2(x1))
        # print(x1.shape, x.shape)
        return x * x1
    
class FABBlock(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=16, dowsample=True, **kwargs):
        super(FABBlock, self).__init__()
        self.out_channels = out_channels
        self.dowsample = dowsample
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.conv_dow1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.conv_dow2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.CA = CABlock(out_channels)
        self.PA = PABlock(out_channels, kernel_size=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = x.clone()
        x1 = self.relu(self.conv(x1))
        res_x1 = x.clone()
        if self.dowsample:
            res_x1 = self.conv_dow1(res_x1)
        # print(x1.shape, res_x1.shape)
        x2 = x1 + res_x1
        x2 = self.PA(self.CA(x2))
        res_x2 = x.clone()
        if self.dowsample:
            res_x2 = self.conv_dow2(res_x2)
        x3 = x2 + res_x2
        return x3
    

class EnCNNBlock(torch.nn.Module):
    def __init__(self, in_channels, h_num=3):
        super(EnCNNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 ** h_num
        self.encoder = self._create_architecture(h_num)

    def _create_architecture(self, h_num):
        layers = []
        in_channels = self.in_channels
        for i in range(h_num-1):
            out_channels = in_channels * 2
            layers.append(torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            layers.append(torch.nn.ReLU())
            in_channels = out_channels
        out_channels = in_channels * 2
        layers.append(torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=0))
        layers.append(torch.nn.ReLU())
        self.out_channels = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
    

class DeCNNBlock(torch.nn.Module):
    def __init__(self, in_channels, h_num=3):
        super(DeCNNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels / 2 ** h_num
        self.decoder = self._create_architecture(h_num)

    def _create_architecture(self, h_num):
        layers = []
        in_channels = self.in_channels

        out_channels = int(in_channels / 2)
        layers.append(torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, output_padding=1))
        layers.append(torch.nn.ReLU())
        in_channels = out_channels
        for i in range(1, h_num-1):
            out_channels = int(in_channels / 2)
            layers.append(torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1))
            layers.append(torch.nn.ReLU())
            in_channels = out_channels
        out_channels = int(in_channels / 2)
        layers.append(torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1))
        self.out_channels = out_channels
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class LIE(torch.nn.Module):
    def __init__(self, in_channels, fab_num=3):
        super(LIE, self).__init__()
        self.in_channels = in_channels
        self.h_num = 3
        self.fab_num = fab_num
        self.encoder_1 = EnCNNBlock(in_channels, h_num=self.h_num)
        self.encoder_2 = EnCNNBlock(in_channels, h_num=self.h_num)
        self.FAB_b = FABBlock(in_channels=self.encoder_1.out_channels, kernel_size=3, padding=1,)
        self.FAB_b_n = FABBlock(in_channels=self.FAB_b.out_channels, kernel_size=3, padding=1,)
        self.FAB_n = FABBlock(in_channels=self.FAB_b_n.out_channels, out_channels=self.encoder_1.out_channels, kernel_size=3, padding=1,)
        self.decoder = DeCNNBlock(self.encoder_1.out_channels, h_num=self.h_num)

    def forward(self, x_lst):
        x = x_lst[0]
        x_grad = x_lst[1]
        x1 = x.clone()
        x1_grad = x_grad.clone()
        x1 = self.encoder_1(x1)
        x1 = self.FAB_b(x1)

        x1_grad = self.encoder_2(x1_grad)
        x1_grad = self.FAB_b(x1_grad)

        x1 = x1 + x1_grad
        for i in range(self.fab_num):
            # x1 = self.FAB_b_n(x1)
            # x1_grad = self.FAB_b_n(x1_grad)
            x1 = FABBlock(in_channels=self.FAB_b.out_channels, kernel_size=3, padding=1,)(x1)
            x1_grad = FABBlock(in_channels=self.FAB_b.out_channels, kernel_size=3, padding=1,)(x1_grad)
            x1 = x1 + x1_grad

        x1 = self.FAB_n(x1)
        x1_grad = self.FAB_n(x1_grad)
        x1 = x1 + x1_grad

        return self.decoder(x1)


       

if __name__ == '__main__':
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('on device:', DEVICE)
    # ca = CABlock(4)
    # FAB = FABBlock(kernel_size=5, padding=2)
    FAB = FABBlock(kernel_size=5, padding=2)
    model = torch.nn.Sequential(
        FAB
    )
    a = torch.randn(2, 3, 10, 10)
    b = model(a)
    print(a.shape)
    print(b.shape)