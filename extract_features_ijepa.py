
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import os
import h5py
import tqdm
from PIL import Image

import src.models.vision_transformer as vit
from src.utils.tensors import trunc_normal_

from src.utils.distributed import init_distributed

import sys
import logging
import copy
import argparse

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def get_h5_list(dir_name):
    h5_files: list[str] = []
    for root, _, files in os.walk(dir_name):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.endswith("h5") and not file[0] == ".":
                h5_files.append(file)
    h5_files.sort()
    return h5_files


def init_encoder(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    encoder.to(device)
    logger.info(encoder)
    return encoder

def prepare_model(device, r_path, patch_size, crop_size, model_name):
    # -- init model
    encoder = init_encoder(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # encoder = DistributedDataParallel(encoder, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)

    # load from checkpoint
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # NOTE(pvalkema): for retrieving a general representation of the image, only the target encoder is used.

        # -- loading encoder
        # pretrained_dict = checkpoint['encoder']
        # msg = encoder.load_state_dict(pretrained_dict)
        # logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        exit(1)

    target_encoder.to(device)
    target_encoder.eval()

    return target_encoder


def extract_features_one_image(device, img, model):
    x = torch.tensor(img)

    # upload to the GPU
    if device == "cuda":
        x = x.cuda()

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run I-JEPA encoder
    features = model.forward(x.float())

    # dimension reduction
    features = features.mean(dim=1)

    return features


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def prepare_image(image_from_h5_file):
    img = Image.fromarray(image_from_h5_file)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    return img


def extract_features_from_h5(device, h5_filename, model, max_patches, out_features_root, model_name):
    all_features = []
    with h5py.File(h5_filename, "r") as hdf5_file:

        if max_patches == 0:
            index = slice(0, None)
        else:
            index = slice(0, max_patches)

        patches = hdf5_file["imgs"][index]
        coords = hdf5_file["coords"][index]

        for patch in patches:
            img = prepare_image(patch)

            with torch.no_grad():
                patch_features = extract_features_one_image(device, img, model).cpu().detach().squeeze(0).numpy()

            all_features.append(patch_features)

    if len(all_features) > 0:
        all_features = np.array(all_features)
        basename, _ = os.path.splitext(os.path.basename(h5_filename))
        feature_path = os.path.join(out_features_root, basename + "_" + model_name + "_features.h5")

        file = h5py.File(feature_path, 'w')
        dset = file.create_dataset('features', shape=all_features.shape, maxshape=all_features.shape, chunks=all_features.shape, dtype=np.float32)
        coord_dset = file.create_dataset('coords', shape=coords.shape, maxshape=coords.shape, chunks=coords.shape, dtype=np.int32)
        dset[:] = all_features
        coord_dset[:] = coords
        file.close()


def main(img_root, out_features_root, checkpoint_path, model_arch, patch_size, crop_size, max_patches_per_image, model_name):
    # Need to initialize multiprocessing, because the I-JEPA target encoder expects to be wrapped using DistributedDataParallel
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed(rank_and_world_size=(0, 1))
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = prepare_model(device, checkpoint_path, patch_size, crop_size, model_arch)

    h5_files = get_h5_list(img_root)
    for h5_filename in tqdm.tqdm(h5_files):
        full_filename = os.path.join(img_root, h5_filename)
        # print(f"Extracting features for {full_filename}")
        extract_features_from_h5(device, full_filename, model, max_patches_per_image, out_features_root, model_name)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-path', type=str,
        help='input directory containing h5 files with images')
    parser.add_argument(
        '--model-arch', type=str, default='vit_large',
        help='vit model arch')
    parser.add_argument(
        '--patch-size', type=int, default=16,)
    parser.add_argument(
        '--crop-size', type=int, default=224,)
    parser.add_argument(
        '--checkpoint', type=str,
        help='path to I-JEPA checkpoint file')
    parser.add_argument(
        '--out-path', type=str,
        help='directory to save the extracted features')
    parser.add_argument(
        '--max-per-h5-file', type=int, default=0,
        help='maximum number of features to extract per h5 file; set to 0 to extract from all')

    args = parser.parse_args()

    if args.data_path:
        # use command-line arguments

        img_root = args.data_path
        model_arch = args.model_arch
        patch_size = args.patch_size
        crop_size = args.crop_size
        checkpoint_path = args.checkpoint
        out_features_root = args.out_path
        max_patches_per_image = args.max_per_h5_file

        os.makedirs(out_features_root, exist_ok=True)

        print(f"Extracting features using checkpoint: {checkpoint_path}...")
        main(img_root, out_features_root, checkpoint_path, model_arch, patch_size, crop_size, max_patches_per_image, "IJEPA")
    else:
        # use hardcoded configs

        img_root = "/run/media/pieter/T7-Pieter/ssl/copy_TRAIN"
        # img_root = "/run/media/pieter/T7-Pieter/ssl/PATCHES"

        configs = [

            {"arch": "vit_large", "patch_size": 16, "crop_size": 224,
             "checkpoint_path": "/run/media/pieter/HugeTwo/projects/ssl/ijepa/result/kidney_vitl16_ep17/jepa-ep5.pt",
             "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/ijepa/kidney_vitl16/5513/ep5/",
             },

            {"arch": "vit_large", "patch_size": 16, "crop_size": 224,
             "checkpoint_path": "/run/media/pieter/HugeTwo/projects/ssl/ijepa/result/kidney_vitl16_ep17/jepa-ep3.pt",
             "out_features_root": "/run/media/pieter/T7-Pieter/ssl/new_features/ijepa/kidney_vitl16/5513/ep3/",
             },

        ]


        # Set to 0 to extract all images
        max_patches_per_image = 0

        for config in configs:
            out_features_root = config["out_features_root"]
            checkpoint_path = config["checkpoint_path"]
            arch = config["arch"]
            patch_size = config["patch_size"]
            crop_size = config["crop_size"]

            os.makedirs(out_features_root, exist_ok=True)

            print(f"Extracting features using checkpoint: {checkpoint_path}...")
            main(img_root, out_features_root, checkpoint_path, arch, patch_size, crop_size, max_patches_per_image, "IJEPA")
