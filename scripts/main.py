import torch
from lightning import Callback
from lightning.pytorch.cli import LightningCLI
from torchvision.utils import make_grid

from data_loader import FolderDataModule, de_normalize
from models import Discriminator, GANModule, SRResNet  # noqa: F401
from pytorch_unet import SimpleResNet, SRUnet, UNet  # noqa: F401
from style_srunet import UnetUpsampler  # noqa: F401


def cli_main():
    class ImageLog(Callback):
        def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int = 0,
        ) -> None:
            if batch_idx == trainer.num_val_batches[0] - 2:
                try:
                    pl_module.logger.log_image(
                        key="samples",
                        images=[
                            make_grid(
                                outputs[0],
                                nrow=outputs[1].shape[0],
                                normalize=True,
                            ),
                            make_grid(
                                torch.cat([outputs[1], outputs[2]]),
                                nrow=outputs[1].shape[0],
                                scale_each=True,
                                normalize=True,
                            ),
                        ],
                        caption=["lq", "hq vs fake"],
                    )
                except Exception as e:
                    print(e)

    log_images_callback = ImageLog()

    cli = LightningCLI(
        GANModule,
        FolderDataModule,
        trainer_defaults={
            "callbacks": [log_images_callback],
        },
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
