import os
import random
from io import BytesIO
from os.path import join
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torchvision
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_imlist(path, ext=".jpg"):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def get_imnames(path, ext=".jpg"):
    return [f.split(".")[0] for f in os.listdir(path) if f.endswith(ext)]


def load_img(path):
    img = Image.open(path)
    return img


def lq(img, qf=20):
    buffer = BytesIO()
    img.save(buffer, "JPEG", quality=qf)
    return Image.open(buffer)


def normalize_img(x):
    return (x - 0.5) * 2.0


def denormalize_img(x):
    return x * 0.5 + 0.5


def downsample(img):
    w, h = img.size
    img = img.resize((w // 2, h // 2), Image.ANTIALIAS)
    return img


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        normalize_img,
    ]
)

transform_from_np = torchvision.transforms.Compose(
    [
        lambda x: x.permute(2, 1, 0),
        normalize_img,
    ]
)

de_normalize = denormalize_img
de_transform = torchvision.transforms.Compose(
    [de_normalize, torchvision.transforms.ToPILImage()]
)


def _filter_bvidvc_path_by_res(path, target_res=1920):
    res_string = path.split("/")[-1].split("_")[1]
    res = res_string.split("x")
    if int(res[0]) == target_res:
        return True
    return False


def _get_pics_in_subfolder(path, ext="jpg"):
    return list(Path(path).rglob(f"*.{ext}"))


class ARDataset(Dataset):
    def __init__(
        self,
        hq_path,
        lq_path="",
        extension="jpg",
        *,
        patch_size,
        path_filter="",
        use_ar=True,
        dataset_upscale_factor=2,
        rescale_factor=None,
        **kwargs,
    ):
        """
        Custom dataloader for the training phase. The getitem method will return a couple (x, y), where x is the
        LowQuality input and y is the relative groundtruth. The relationship between the LQ and HQ samples depends on
        how the dataset is built.

        Args
            hq_path (str):
                base path of the hq dataset dir.
            lq_path (str):
                base path of the lq dataset dir. Optional if use_ar=False.
            extension (str, default="jpg"):
                extension of images.
            patch_size (int):
                width/height of the training patch. the model is going to be trained on patches,
                randomly extracted from the datasets.
            path_filter (str):
                additional string that must be present in the path.
            use_ar (bool, default=True):
                if False, it bypasses the loading of the Low Quality/Low Resolution samples. Then, the LR
                samples will be generated by downscaling the groundtruth, thus the model will be NOT trained for
                performing the Artifact Reduction function.
            dataset_upscale_factor (int, default=2):
                resolution relationship between LQ and HQ samples. Must be known in advance: in our experiments,
                we encoded the clips and halved their resolution, thus in our case upscale_factor is 2.
            rescale_factor (None or float):
                if not None, the groundtruth will be resized from upscale_factor*patch_size to rescale_factor*patch_size.
                This is useful for training on upscale factors lower than the one expected from the dataset. Use combined
                with 'downscale' parameter of the UNets classes.
        """

        self.patch_size = patch_size
        self.hq_path = hq_path
        self.lq_path = lq_path
        self.path_filter = path_filter
        self.extension = extension
        self.ar = use_ar
        self.upscale_factor = dataset_upscale_factor
        self.rf = rescale_factor

        self.hq_dir = sorted(
            filter(
                lambda p: str(path_filter) in str(p),
                _get_pics_in_subfolder(hq_path, ext=extension),
            )
        )
        self.lq_dir = (
            sorted(
                filter(
                    lambda p: str(path_filter) in str(p),
                    _get_pics_in_subfolder(lq_path, ext=extension),
                )
            )
            if lq_path != ""
            else []
        )

        assert not use_ar or len(self.hq_dir) == len(
            self.lq_dir
        ), "use_ar is True but num of lq images does not correspond to hq images"

        self.size = len(self.hq_dir)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # HQ version of the image

        hq = load_img(self.hq_dir[item])

        sf = self.upscale_factor

        w, h = (np.array(hq.size) / sf).astype(int)
        if w > self.patch_size:
            w = w - self.patch_size
            w_pos = random.randint(0, w - 1)
        else:
            w_pos = 0

        if h > self.patch_size:
            h = h - self.patch_size
            h_pos = random.randint(0, h - 1)
        else:
            h_pos = 0

        # left, upper, right, and lower
        crop_pos = (w_pos, h_pos, w_pos + self.patch_size, h_pos + self.patch_size)
        crop_pos_sr = (
            sf * w_pos,
            sf * h_pos,
            sf * (w_pos + self.patch_size),
            sf * (h_pos + self.patch_size),
        )
        hq = hq.crop(crop_pos_sr)
        if self.ar:
            lq = load_img(self.lq_dir[item])
            lq = lq.crop(crop_pos)
        else:
            lq = hq.resize((self.patch_size, self.patch_size))

        if self.rf is not None:
            hq = hq.resize(
                (int(self.rf * self.patch_size), int(self.rf * self.patch_size))
            )

        # random flip
        if torch.rand(1) < 0.5:
            lq = functional.hflip(lq)
            hq = functional.hflip(hq)

        x = transform(lq)
        y = transform(hq)

        return x, y


class FolderDataModule(L.LightningDataModule):
    def __init__(
        self,
        hq_path,
        lq_path="",
        extension="jpg",
        *,
        patch_size,
        crf,
        path_filter="",
        train_pct=0.8,
        use_ar=True,
        dataset_upscale_factor=2,
        rescale_factor=None,
        batch_size=32,
    ):
        """
        Custom PyTorch Lightning DataModule.

        See :class:`~data_loader.ARDataset` for details on args.

        Args
            batch_size (int):
                Size of every training batch.
        """

        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dataset = ARDataset(**self.hparams)
            train_set_size = int(len(dataset) * self.hparams.train_pct)
            valid_set_size = len(dataset) - train_set_size

            # split the train set into two
            self.train_set, self.valid_set = random_split(
                dataset, [train_set_size, valid_set_size]
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=12,
            shuffle=True,
            pin_memory=True,
        )
        return data_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        data_loader_eval = DataLoader(
            dataset=self.valid_set,
            batch_size=self.hparams.batch_size,
            num_workers=12,
            shuffle=False,
            pin_memory=True,
        )
        return data_loader_eval


def _to_lq_frameid(name):
    name_splitted, format = name.split(".")[0], name.split(".")[1]
    name_splitted = name_splitted.split("_")
    id, res, frame_id = name_splitted[0], name_splitted[1], name_splitted[2]
    name_hq = [id, str(int(res) // 2), frame_id]
    return "_".join(name_hq) + "." + format


def _to_lq(name):
    name_splitted, format = name.split(".")[0], name.split(".")[1]
    name_splitted = name_splitted.split("_")
    imid, res, frameid = name_splitted[0], name_splitted[1], name_splitted[2]
    name_hq = [imid, str(int(res) // 2), frameid]  # RIMETTERE IN CASO DI SR // 2)]
    return "_".join(name_hq) + "." + format


def _to_lq_vid(name):
    name_splitted, format = name.split(".")[0], name.split(".")[1]
    name_splitted = name_splitted.split("_")
    imid, res = name_splitted[0], name_splitted[1]
    name_hq = [imid, str(int(res) // 2)]  # RIMETTERE IN CASO DI SR // 2)]
    return "_".join(name_hq) + "." + format


def _stack(tensor_list):
    return torch.stack(tensor_list, dim=0)


def is_image(path):
    return path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".png")


def _imname(path):
    name = "_".join(path.split(".")[0].split("_")[:-1])
    return name


def _strip_ext(path):
    return path.strip(".png").strip(".jpg").strip(".jpeg")


def sort_by_frame_id(key):
    value = key.split(".")[-2].split("_")[-1].replace("frame", "")
    return int(value)


class TestDataLoader(Dataset):
    def __init__(self, dir, sr=False, video_prefix=None):
        self.dir = dir
        self.im_list = os.listdir(dir + "_LQ")
        if video_prefix is not None:
            self.im_list = [v for v in self.im_list if v.startswith(video_prefix)]
        self.im_list = sorted(self.im_list, key=sort_by_frame_id)
        self.sr = sr

    def __len__(self):
        return len(self.im_list)

    def cut_im_list(self, from_frame, to_frame):
        from_frame = int(from_frame)
        to_frame = int(to_frame)

        assert (
            from_frame < to_frame
        ), "Wrong attempt to cut the video. From frame {} is >= {}.".format(
            from_frame, to_frame
        )
        to_frame = min(to_frame, len(self.im_list))
        self.im_list = self.im_list[from_frame:to_frame]

    def __getitem__(self, id):
        pic_name = self.im_list[id].split(".")[0]

        frame_suffix = pic_name.split("_")[-1]
        res_suffix = pic_name.split("_")[-2]
        pic_name = pic_name.split("_")[:-2]

        pic_name_lq = pic_name + [res_suffix, frame_suffix]
        pic_name_lq = "_".join(pic_name_lq)

        pic_name_hq = pic_name + [
            str(int(res_suffix) * (2 if self.sr else 1)),
            frame_suffix,
        ]
        pic_name_hq = "_".join(pic_name_hq)

        lr = Image.open(join(self.dir + "_LQ", pic_name_lq + ".jpg"))
        hr = Image.open(join(self.dir + "_HQ", pic_name_hq + ".jpg"))

        w, h = lr.size
        sf = 2 if self.sr else 1
        hr = hr.resize((w * sf, h * sf))

        # w, h = hr.size
        # if w >= 1280 and h >= 720:
        #     w = w - 1280
        #     h = h - 720
        #
        #     w_pos = random.randint(0, w - 1)
        #     h_pos = random.randint(0, h - 1)
        #
        #     # left, upper, right, and lower
        #     crop_pos = (w_pos, h_pos, w_pos + 1280, h_pos + 720)
        #     hr = hr.crop(crop_pos)
        #     lr = lr.crop(crop_pos)

        return transform(lr), transform(hr)


class SingleFolderLoader(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.im_list = os.listdir(dir)
        self.im_list = sorted(self.im_list, key=sort_by_frame_id)
        normalize = torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
        self.preprocess = torchvision.transforms.Compose(
            [
                # torchvision.transforms.RandomCrop((64, 64)),
                torchvision.transforms.Resize((256, 256)),
                # transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, id):
        pic_name = self.im_list[id].split(".")[0]

        hr = Image.open(join(self.dir, pic_name + ".png"))

        return self.preprocess(hr)  # transform(hr)
        return self.preprocess(hr)  # transform(hr)
