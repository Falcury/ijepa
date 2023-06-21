import os
from logging import getLogger

import torch
import torchvision
import tqdm
from PIL import Image

from torch.utils.data import Dataset
import h5py

logger = getLogger()

class KidneyDataset(Dataset):
    def __init__(self, data_path, transform=None, training=True, csv_path=None):
        self.transform = transform
        self.training = True # TODO(pvalkema): load validation data if False?
        self.is_initialized = False
        self.image_dataset = []
        self.filenames = []
        if csv_path is None:
            self._load_h5(data_path)
        else:
            self._load_csv(csv_path)
        self.data_length = len(self.image_dataset)
        self.data_path = data_path

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # print("get item==========")
        if not self.is_initialized:
            self._load_h5(self.data_path)
            self.is_initialized = True
        try:
            filename, extension = os.path.splitext(self.image_dataset[idx])
            case = filename + ".h5"
            patch_id = extension.split("_")[1]
            # case, patch_id = self.image_dataset[idx].split('_')
            f = h5py.File(case, "r")
            img = f['imgs'][int(patch_id)]
            img = Image.fromarray(img)
            # img = torch.from_numpy(img)

            img = self.transform(img)
            return (img,)
            # img = torch.from_numpy(img).unsqueeze(0)
            # coords = f['coords'][int(patch_id)]
            # if is_success:
            #     self.on_sucess(img)
            # return img, coords  # , os.path.splitext(abs_p)[0].split('\\')[-1] # self.h5data[idx]
        except Exception as e:
            logger.warning(
                f"Couldn't load: {self.image_dataset[idx]}. Exception: \n{e}"
            )

    def _load_h5(self, data_path):
        for root, dirs, files in os.walk(data_path):
            for every_file in tqdm.tqdm(files):
                try:
                    full_filename = os.path.join(root, every_file)
                    f = h5py.File(full_filename, "r")
                    # file_name, file_type = os.path.splitext(every_file)
                    self.filenames.append(full_filename)
                    for i in range(len(f['imgs'])):
                        self.image_dataset.append(os.path.join(root, every_file + "_%d" % (i)))
                except Exception as e:
                    logger.warning(
                        f"Couldn't load: {every_file}. Exception: \n{e}"
                    )
        self.is_initialized = True

    def _load_csv(self, csv_path):
        with open(csv_path, "r") as csv:
            for line in csv:
                line = line.rstrip()
                cols = line.split(",")
                filename = cols[0]
                num_imgs = int(cols[1])
                self.filenames.append(filename)
                for i in range(num_imgs):
                    self.image_dataset.append(filename + "_%d" % (i))
        self.is_initialized = True


def make_kidney_dataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    csv_path=None
):
    dataset = KidneyDataset(
        data_path=root_path,
        transform=transform,
        training=training,
        csv_path=csv_path,
    )

    logger.info("Kidney dataset created")

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info("Kidney unsupervised data loader created")

    return dataset, data_loader, dist_sampler
