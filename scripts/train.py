import os

import utils

args = utils.ARArgs()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVICE

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn as nn

import data_loader as dl
from models import (
    SRResNet,  # courtesy of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
)
from models import Discriminator, GANModule
from pytorch_unet import SimpleResNet, SRUnet, UNet

if __name__ == "__main__":
    seed_everything(42, workers=True)

    args = utils.ARArgs()
    torch.autograd.set_detect_anomaly(True)

    print_model = args.VERBOSE
    arch_name = args.ARCHITECTURE
    dataset_upscale_factor = args.UPSCALE_FACTOR
    n_epochs = args.N_EPOCHS

    if arch_name == "srunet":
        generator = SRUnet(
            3,
            residual=True,
            scale_factor=dataset_upscale_factor,
            n_filters=args.N_FILTERS,
            downsample=args.DOWNSAMPLE,
            layer_multiplier=args.LAYER_MULTIPLIER,
        )
    elif arch_name == "unet":
        generator = UNet(
            3,
            residual=True,
            scale_factor=dataset_upscale_factor,
            n_filters=args.N_FILTERS,
        )
    elif arch_name == "srgan":
        generator = SRResNet()
    elif arch_name == "espcn":
        generator = SimpleResNet(n_filters=64, n_blocks=6)
    else:
        raise Exception("Unknown architecture. Select one between:", args.archs)

    if args.MODEL_NAME is not None:
        print("Loading model: ", args.MODEL_NAME)
        state_dict = torch.load(args.MODEL_NAME)
        generator.load_state_dict(state_dict)

    discriminator = Discriminator()

    dm = dl.FolderDataModule(
        path=str(args.DATASET_DIR),
        patch_size=96,
        crf=int(args.CRF),
        use_ar=True,
    )

    model = GANModule(generator, discriminator, args.W0, args.W1, args.L0)

    checkpoint_callback = ModelCheckpoint(every_n_epochs=20)
    trainer = Trainer(max_epochs=n_epochs, callbacks=[checkpoint_callback])

    trainer.fit(model, dm)
