import torch
import torch as T
from torch import nn
import torch.nn.functional as F


class Print(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class Cat(nn.Sequential):
    def forward(self, x):
        return T.cat([m(x) for m in self.children()], dim=1)


def cats(*blocks):
    return Cat(nn.Sequential(*blocks), nn.Sequential())


class Sum(nn.Sequential):
    def forward(self, x):
        first, *last = self.children()
        return sum((m(x) for m in last), first(x))


def conv(cin, cout=None, kernel_size=3, stride=1, padding=1):
    cout = cout or cin
    if cout != cin or (kernel_size, stride) != (3, 1):
        return nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, kernel_size, padding=padding, stride=stride),
        )

    return Sum(
        nn.Sequential(),
        nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cin // 4, 1),
            nn.BatchNorm2d(cin // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin // 4, cin // 4, kernel_size, padding=1),
            nn.BatchNorm2d(cin // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin // 4, cout, 1),
        )
    )


def upconv(cin, cout=None):
    cout = cout or cin
    return nn.Sequential(
        nn.BatchNorm2d(cin),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(cin, cout, 3, padding=1, stride=2),
    )


def rep(func, n, *args, **kwargs):
    for _ in range(n):
        yield func(*args, **kwargs)


def unet(cin, cout, init=32, enc_n=2, dec_n=0):
    # it's actually U-net
    return nn.Sequential(
        nn.BatchNorm2d(cin),
        conv(cin, init, kernel_size=2),  # for even inputs
        # conv(cin, init),  # for odd inputs
        cats(
            conv(init, init * 2, stride=2),
            *[conv(init * 2) for _ in range(enc_n)],
            cats(
                conv(init * 2, init * 4, stride=2),
                *[conv(init * 4) for _ in range(enc_n)],
                cats(
                    conv(init * 4, init * 8, stride=2),
                    *[conv(init * 8) for _ in range(enc_n)],
                    cats(
                        conv(init * 8, init * 16, stride=2),
                        *[conv(init * 16) for _ in range(enc_n)],
                        cats(
                            conv(init * 16, init * 32, stride=2),
                            *[conv(init * 32) for _ in range(enc_n + dec_n)],
                            upconv(init * 32, init * 16),
                        ),
                        conv(init * 32, init * 16),
                        *[conv(init * 16) for _ in range(dec_n)],
                        upconv(init * 16, init * 8),
                    ), conv(init * 16,
                            init * 8), *[conv(init * 8) for _ in range(dec_n)],
                    upconv(init * 8, init * 4)
                ), conv(init * 8,
                        init * 4), *[conv(init * 4) for _ in range(dec_n)],
                upconv(init * 4, init * 2)
            ), conv(init * 4,
                    init * 2), *[conv(init * 2) for _ in range(dec_n)],
            upconv(init * 2, init)
        ),
        conv(init * 2, init),

        # nn.Conv2d(init, cout, 3, padding=0), # for odd inputs
        nn.Conv2d(init, cout, 2, padding=0),  # for even inputs
        nn.Sigmoid(),
    )


def load_model(cin, cout, path=None):
    unet = unet(cin, cout)
    if path is not None:
        unet.load_state_dict(torch.load(path))
    return unet


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
