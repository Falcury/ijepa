from src.datasets.kidney import KidneyDataset
import h5py
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path', type=str,
    help='directory containing h5 files')

def main(root_path):
    dataset = KidneyDataset(
        data_path=root_path,
    )
    print(f"Number of h5 files: {len(dataset.filenames)}")
    print(f"Number of images: {len(dataset)}")

    img = None
    print("Testing h5 file integrity...")
    for filename in tqdm.tqdm(dataset.filenames):
        f = h5py.File(filename, "r")
        num_imgs = len(f["imgs"])
        first_img = f["imgs"][0]
        last_img = f["imgs"][num_imgs-1]
        del first_img
        del last_img
        f.close()
        pass

if __name__=="__main__":
    # dataset_path = "/gpfs/work1/0/einf2634/h5_patches_0.5mmp_256x256/TRAIN/"
    # dataset_path = "/run/media/pieter/T7-Pieter/ssl/PATCHES/"
    args = parser.parse_args()
    main(args.data_path)
