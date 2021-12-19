from math import exp
import torch.nn.functional as F
import torch.nn as nn
import torch

"""
    This script defines the MEF-SSIM loss function which is mentioned in the DeepFuse paper
    The code is heavily borrowed from: https://github.com/Po-Hsun-Su/pytorch-ssim

    Author: SunnerLi
"""


L1_NORM = lambda a: torch.sum(a + 1e-8)
L2_NORM = lambda b: torch.sqrt(torch.sum((b + 1e-8) ** 2))


class MEF_SSIM_Loss(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        """
            Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
            Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)  # sigma = 1.5    shape: [11, 1]
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(
            0)  # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # window shape: [1,1, 11, 11]
        return window

    def ssim(self, img1, img2, window, window_size, channel):
        """
            Compute the SSIM for the given two image
            The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        pad = window_size // 2
        L = 255

        mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding= pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding= pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding= pad, groups=channel) - mu1_mu2

        C2 = (0.03*L) ** 2
        ssim_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        return ssim_map.mean()



    def forward(self, y1, y2, yf):
        (_, channel, height, width) = yf.size()
        window_size = 11
        window = self.create_window(window_size, channel=channel).to(y1.device)
        pad = window_size // 2

        u1 = F.conv2d(y1, window, padding=pad, groups=channel)
        u2 = F.conv2d(y2, window, padding=pad, groups=channel)

        c1 = L2_NORM(y1 - u1)
        c2 = L2_NORM(y1 - u1)

        # 对比度
        c = torch.max(c1, c2)

        n1 = torch.sum(y1)
        n2 = torch.sum(y2)

        w1 = n1/(n1 + n2 + 1e-8)
        w2 = n2/(n1 + n2 + 1e-8)

        # 亮度
        l = torch.mean(w1 * y1 + w2 * y2)

        # Get the s_hat
        s1 = (y1 - u1) / L2_NORM(y1 - u1)
        s2 = (y2 - u2) / L2_NORM(y2 - u2)

        # 结构
        s = w1 * s1 + w2 * s2

        y = c * s + l

        # Check if need to create the gaussian window 
        (_, channel, _, _) = y.size()

        # Compute SSIM between y_hat and y_f
        score = self.ssim(y, yf, window, window_size, channel)

        return 1 - score


if __name__ == '__main__':
    criterion = MEF_SSIM_Loss()
    input = torch.rand([1, 1, 64, 64])
    output = torch.rand([1, 1, 64, 64])
    img_fuse = torch.rand([1, 1, 64, 64])

    input = input.cuda()
    output = output.cuda()
    img_fuse = img_fuse.cuda()
    loss = criterion(y1=input, y2=output, yf=img_fuse)
    print(loss)