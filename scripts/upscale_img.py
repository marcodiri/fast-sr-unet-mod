import argparse

import torch

import data_loader
from style_srunet import StyleSRUnet, UnetUpsampler

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--ckpt",
    type=str,
    help="Path to lightning model checkpoint",
)
parser.add_argument(
    "-i",
    "--image",
    type=str,
    help="Image to upscale",
)

args = parser.parse_args()

generator = StyleSRUnet(
    style_network=dict(
        dim=64,
        depth=4,
    ),
    upscale_factor=2,
)

checkpoint = torch.load(args.ckpt)
state_dict = {
    ".".join(k.split(".")[1:]): v
    for k, v in checkpoint["state_dict"].items()
    if "G." in k
}
generator.load_state_dict(state_dict)

lq = data_loader.load_img(args.image)
lq = data_loader.transform(lq)
hq_fake = generator(lq.unsqueeze(0))
output_name = args.image.split(".")
output_name[-1] = "fakehq." + output_name[-1]
output_name = ".".join(output_name)
data_loader.de_transform(hq_fake[0]).save(output_name)
