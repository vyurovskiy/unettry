import os
import xml.etree.ElementTree as ET
from collections import namedtuple
from itertools import chain
from multiprocessing import Process, Queue
from multiprocessing.dummy import Pool
from threading import local, get_ident
from time import time

import numpy as np
from PIL import Image
from shapely.geometry import MultiPolygon, Point, Polygon
from tqdm import tqdm

from ._mir_hook import mir

_TLS = local()  # thread-local storage


def tls_prng():
    # pylint: disable=no-member
    try:
        return _TLS.prng
    except AttributeError:
        _TLS.prng = np.random.RandomState(
            (get_ident() + np.random.get_state()[1][0]) % 2**32
        )
        return _TLS.prng


def get_files(imagesP, masksP, xmlsP):
    for name in os.listdir(imagesP):
        name = name.split('.tif')[0]
        yield {
            'imageP': imagesP + name + '.tif',
            'maskP': masksP + name + '_M.tif',
            'cancerxmlP': xmlsP + name + '.xml',
            'groundxmlP': xmlsP + name + '_G.xml'
        }


def get_coordinates(xmlP):
    for ann in ET.parse(xmlP).getroot().find('Annotations'):
        yield [
            (int(float(coord.attrib['Y'])), int(float(coord.attrib['X'])))
            for coord in ann.find('Coordinates').findall('Coordinate')
        ]  # <- it's "p"


def to_multipolygon(yx):
    return MultiPolygon([[p, []] for p in yx]).buffer(0)  # <- this one "p"


def get_random_points(polygon, number):
    prng = tls_prng()

    if not polygon.area:
        return

    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(number):
        while True:
            p = Point(prng.uniform(minx, maxx), prng.uniform(miny, maxy))
            if polygon.contains(p):
                yield (int(p.coords[0][0]), int(p.coords[0][1]))
                break

    # for _ in range(number):
    #     p = polygon.representative_point()
    #     yield (int(p.coords[0][0]), int(p.coords[0][1]))


_Cancer = namedtuple('_Cancer', ['SlideP', 'MaskP', 'Coord', 'Whiteness'])


def get_data_unsafe(list_of_data, patch_size, zoom, nbpoints):
    np.random.seed(int(time() * 1000) % 2**32)
    reader = mir.MultiResolutionImageReader()

    # for data in list_of_data:
    def read_generator(data):
        mask = reader.open(data['maskP'])
        canc_poly = to_multipolygon(get_coordinates(data['cancerxmlP']))
        whole_poly = to_multipolygon(get_coordinates(data['groundxmlP']))

        gt = get_random_points(whole_poly.difference(canc_poly), nbpoints)
        gt = list(gt)
        canc = get_random_points(canc_poly, 2 * nbpoints - len(gt))

        # gt = list(get_random_points(whole_poly, nbpoints))
        # canc = list(get_random_points(canc_poly, nbpoints))

        for yx in chain(gt, canc):
            # for yx in gt + canc:
            mask_patch = mask.getUCharPatch(
                startY=yx[0],
                startX=yx[1],
                height=patch_size[0],
                width=patch_size[1],
                level=zoom
            )
            mask_patch = mask_patch.astype(bool)
            values, counts = np.unique(mask_patch, return_counts=True)

            yield _Cancer(
                data['imageP'], data['maskP'], yx,
                sum(values * counts) / sum(counts) * 100
            )

    def read(data):
        return list(read_generator(data))

    with Pool(12) as pool:
        for subset in pool.imap_unordered(read, list_of_data):
            yield from subset

    # for data in list_of_data:
    #     yield from read_generator(data)


def get_data_unsafe_queue(q, list_of_data, patch_size, zoom, nbpoints):
    for item in get_data_unsafe(list_of_data, patch_size, zoom, nbpoints):
        q.put(item)
    q.put(None)


def get_data(list_of_data, patch_size, zoom, nbpoints):
    q = Queue()
    p = Process(
        target=get_data_unsafe_queue,
        args=(q, list_of_data, patch_size, zoom, nbpoints)
    )
    p.start()
    try:
        yield from iter(q.get, None)
    finally:
        p.join()
        p.terminate()


def gen_test_data(test_path, patch_size, zoom):
    reader = mir.MultiResolutionImageReader()
    slide = reader.open(test_path)
    w, h = slide.getDimensions()
    coord = np.stack(
        np.meshgrid(
            np.arange(0, h, patch_size[0] * 2**zoom),
            np.arange(0, w, patch_size[1] * 2**zoom),
            indexing='ij'
        ),
        axis=-1
    )
    # coord=coord.reshape((coord.shape[0]*coord.shape[1],2))
    return slide, coord