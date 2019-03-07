import sys
from contextlib import contextmanager
from time import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T
from PIL import Image, ImageFilter

from ._mir_hook import mir
from .data import get_data

_READER = mir.MultiResolutionImageReader()


@contextmanager
def timeit(message):
    start = time()
    try:
        yield
    finally:
        print(f'{message} done in {time() - start:.4f}')


# print('update')


def noisify_colors(x, sigma=10):
    return np.random.normal(x, scale=sigma).clip(0, 255).astype('uint8')


def noisify(x, sigma=5):
    scales = (1, 2, 4)
    octaves = (
        np.random.normal(
            scale=sigma * s, size=(x.shape[0] // s, x.shape[1] // s)
        ).astype('float32') for s in scales
    )
    simplex = sum(
        cv2.resize(oct, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        for s, oct in zip(scales, octaves)
    )
    return (x + simplex).clip(0, 255).astype('uint8')


def normalize(x):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    l, *ab = cv2.split(x)

    clahe = cv2.createCLAHE(2., (8, 8))
    l = clahe.apply(l)
    l = noisify(l)

    x = cv2.merge([l, *ab])
    x = cv2.cvtColor(x, cv2.COLOR_LAB2BGR)
    return x


# # pip install dataclasses --user
# @dataclass
# class Transformer:
#     max_angle: float = 90

#     def __call__(self, *arrays):
#         angle = np.random.uniform(-self.max_angle, self.max_angle)
#         return tuple(a.rotate(angle) for a in arrays)


class UniformSampler(Sampler):
    def __init__(self, list_of_data, patch_size, zoom, nbpoints, parts):
        self.list_of_data, self.patch_size, self.zoom, self.nbpoints = list_of_data, patch_size, zoom, nbpoints
        self.parts = parts

    def __iter__(self):
        with timeit('fetch'):
            all_items = list(
                get_data(
                    self.list_of_data, self.patch_size, self.zoom,
                    self.nbpoints
                )
            )
        indexes = sorted(all_items, key=lambda item: item.Whiteness)
        indexes = list(indexes)

        hist, bins = np.histogram(
            [int(i.Whiteness) for i in indexes],
            density=True,
            bins=np.arange(0, 101, 20)
        )
        print(hist / hist.sum(), bins, sep='\n')

        part = len(indexes) / self.parts
        heaps = tuple(
            indexes[int(part * i):int(part * i + part)]
            for i in range(self.parts)
        )
        for heap in heaps:
            np.random.shuffle(heap)

        for serie in zip(*heaps):
            yield from serie

    def __len__(self):
        return len(self.list_of_data) * self.nbpoints * 2


class TrainDataset(Dataset):
    def __init__(self, count, patch_size, zoom, transform=None):
        self.count = count
        self.patch_size = patch_size
        self.zoom = zoom
        self.transform = transform

    def __len__(self):
        return self.count

    def __getitem__(self, sample):
        slide = _READER.open(sample.SlideP)
        mask = _READER.open(sample.MaskP)
        slide_patch = slide.getUCharPatch(
            startY=int(sample.Coord[0] - self.patch_size[0] * 2**self.zoom),
            startX=int(sample.Coord[1] - self.patch_size[1] * 2**self.zoom),
            height=int(2 * self.patch_size[0]),
            width=int(2 * self.patch_size[1]),
            level=self.zoom
        )
        # try:
        #     cv2.imshow('image', slide_patch)
        #     cv2.waitKey(0)
        # finally:
        #     cv2.destroyWindow('image')

        # slide_patch = normalize(slide_patch)
        # slide_patch = noisify_colors(slide_patch)

        # cv2.imshow('image', slide_patch)
        # cv2.waitKey(16)

        # try:
        #     cv2.imshow('image', slide_patch)
        #     # cv2.waitKey(0)
        # finally:
        #     cv2.destroyWindow('image')
        # slide_patch = noisify_colors(slide_patch)
        slide_patch = Image.fromarray(slide_patch).convert('RGB')
        # slide_patch = slide_patch.filter(ImageFilter.MinFilter(3))
        mask_patch = mask.getUCharPatch(
            startY=int(sample.Coord[0] - self.patch_size[0] * 2**self.zoom),
            startX=int(sample.Coord[1] - self.patch_size[1] * 2**self.zoom),
            height=int(2 * self.patch_size[0]),
            width=int(2 * self.patch_size[1]),
            level=self.zoom
        )
        mask_patch = mask_patch[:, :, 0]
        mask_patch = Image.fromarray(mask_patch * 255)
        mask_patch = mask_patch.convert('L')

        if self.transform is not None:
            slide_patch, mask_patch = self.transform(slide_patch, mask_patch)
        else:
            slide_patch, mask_patch = T.Compose(
                [T.CenterCrop(self.patch_size),
                 T.ToTensor()]
            )(slide_patch, mask_patch)

        # slide_patch += torch.randn(*slide_patch.shape) * .03

        if __debug__:
            np_image = slide_patch.numpy().transpose(1, 2, 0)[..., ::-1]
            np_mask = mask_patch[0][..., None].expand(*self.patch_size,
                                                      3).numpy()
            cv2.imshow('image', np_image * (3 * np_mask + 1) * .25)
            cv2.waitKey(16)

        return slide_patch.share_memory_(), mask_patch.floor_().share_memory_()

    def __del__(self):
        cv2.destroyAllWindows()


# class TestDataset(Dataset):
#     def __init__(self, slide, grid, patch_size, zoom):
#         self.patch_size = patch_size
#         self.zoom = zoom
#         self.slide = slide
#         self.grid = grid.reshape((grid.shape[0] * grid.shape[1], 2))

#     def __len__(self):
#         return len(self.grid)

#     def __getitem__(self, index):
#         slide_patch = self.slide.getUCharPatch(
#             int(self.grid[index][0]), int(self.grid[index][1]),
#             self.patch_size[0], self.patch_size[1], self.zoom
#         )
#         slide_patch = Image.fromarray(slide_patch).convert('RGB')
#         slide_patch = T.ToTensor()(slide_patch)

#         return slide_patch.share_memory_()
