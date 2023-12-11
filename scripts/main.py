from lightning import Callback
from lightning.pytorch.cli import LightningCLI

from data_loader import FolderDataModule
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
                        images=[outputs[1], outputs[2]],
                        caption=["hq", "hq_fake"],
                    )
                except Exception as e:
                    print(e)

    log_images_callback = ImageLog()

    cli = LightningCLI(
        GANModule,
        FolderDataModule,
        trainer_defaults={
            "devices": 1,
            "callbacks": [log_images_callback],
        },
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
