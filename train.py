"""
Model trainer

Usage:
  train2.py [options] EPOCHS

Options:
  -p,--patch-size PATCHSIZE  Size of patch [default: 512]
  -b,--batch-size BATCHSIZE  Size of minibatch [default: 3]

"""
import gc
import json
from collections import namedtuple
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torch
from docopt import docopt
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (
    Events, create_supervised_evaluator, create_supervised_trainer
)
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image

from models.unet import unet, count_params
from putils.data import get_files
from putils.loss import dice, dice_loss
from putils.generator import TrainDataset, UniformSampler
from putils.handlers import SummaryWriter


def main(patch_size, zoom, batch_size, epochs):
    with open('./config.json') as f:
        config = json.load(f)
    paths = list(get_files(**config))

    path_train, path_valid = (
        paths[:-len(paths) // 10], paths[-len(paths) // 10:]
    )

    transforms = T.Compose(
        [
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(90, resample=Image.BILINEAR),
            T.CenterCrop(patch_size),
            T.ToTensor()
        ]
    )

    # assert len(path_train) % batch_size == 0
    if __debug__:
        path_train = path_train[:batch_size]
        path_valid = path_valid[:batch_size]

    t_sampler = UniformSampler(
        path_train, patch_size, zoom, 1, parts=batch_size
    )
    trainDataset = TrainDataset(len(t_sampler), patch_size, zoom, transforms)

    trainDataloader = DataLoader(
        trainDataset,
        sampler=t_sampler,
        batch_size=batch_size,
        num_workers=6,
        drop_last=True
    )

    v_sampler = UniformSampler(
        path_valid, patch_size, zoom, 1, parts=batch_size
    )
    validDataset = TrainDataset(len(v_sampler), patch_size, zoom)
    validDataloader = DataLoader(
        validDataset,
        batch_size=batch_size,
        sampler=v_sampler,
        num_workers=6,
        drop_last=True
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # pylint: disable=no-member

    model = unet(3, 1)
    print('Param count:', count_params(model))

    tag = datetime.now().strftime('%m-%d--%H-%M')
    t_writer = SummaryWriter(model, f'./logs/{tag}/train', (3, 512, 512))
    v_writer = SummaryWriter(model, f'./logs/{tag}/val', (3, 512, 512))
    optimizer = Adam(model.parameters(), lr=3e-4)  # !!!!!!!!!

    trainer = create_supervised_trainer(
        model, optimizer, dice_loss, device=device
    )
    evaluator = create_supervised_evaluator(
        model, metrics={
            'acc': Accuracy(),
            'dice': Loss(dice)
        }, device=device
    )
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    ProgressBar(persist=True).attach(trainer, ['loss'])
    ProgressBar(persist=True).attach(evaluator)

    # @trainer.on(Events.STARTED)
    # def continuation_hook(engine):
    #     model.load_state_dict(torch.load(path))
    #     engine.state.epoch = 100

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        # pylint: disable=no-member
        for cat, loader, writer in (
            ('train', trainDataloader, t_writer),
            ('val', validDataloader, v_writer)
        ):
            for k, v in evaluator.run(loader).metrics.items():
                engine.state.metrics[f'{cat}_{k}'] = round(v, 4)
                writer.add_scalar(k, v, engine.state.epoch)
        print(f'[Epoch {engine.state.epoch}] metrics = {engine.state.metrics}')

    now = datetime.now()
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            f'./checkpoints/{now.year%100:02d}-{now.month:02d}-{now.day:02d}',
            'unet',
            n_saved=3,
            score_function=lambda e: e.state.metrics['val_dice'],
            score_name='dice',
            create_dir=True,
            require_empty=False,
            save_as_state_dict=True,
        ), {'unet': model}
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda *_: (print(datetime.now().strftime('%H:%M:%S'), end='\n\n'),
                    torch.cuda.empty_cache(),
                    gc.collect())
    )
    trainer.run(trainDataloader, max_epochs=epochs)


if __name__ == '__main__':
    freeze_support()
    args = docopt(__doc__)

    patch_size = int(args['--patch-size'])
    batch_size = int(args['--batch-size'])
    zoom = 0  ### !!
    main((patch_size, patch_size), zoom, batch_size, int(args['EPOCHS']))
