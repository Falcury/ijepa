from src.datasets.kidney import KidneyDataset
import h5py
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-path', type=str,
    help='directory containing h5 files')
parser.add_argument(
    '--test-integrity', action='store_true',
    help='test h5 file integrity by reading the first and last image'
)
parser.add_argument(
    '--create-csv', type=str,
    help='create csv file with h5 filenames and number of images per file'
)
parser.add_argument(
    '--init-from-csv', type=str,
    help='initialize dataset from csv file'
)

def main(args):
    dataset = KidneyDataset(
        data_path=args.data_path,
        csv_path=args.init_from_csv,
    )
    print(f"Number of h5 files: {len(dataset.filenames)}")
    print(f"Number of images: {len(dataset)}")

    if args.test_integrity:
        print("Testing h5 file integrity...")
        for i, ds in enumerate(tqdm.tqdm(dataset.h5_dsets)):
            if ds is None:
                f = h5py.File(dataset.filenames[i], "r")
                ds = f['imgs']
            num_imgs = len(ds)
            first_img = ds[0]
            last_img = ds[num_imgs-1]
            del first_img
            del last_img

    if args.create_csv:
        print("Creating csv file...")
        with open(args.create_csv, 'w') as csv:
            for filename in tqdm.tqdm(dataset.filenames):
                f = h5py.File(filename, "r")
                num_imgs = len(f["imgs"])
                print(f"{filename},{num_imgs}", file=csv)
                f.close()

if __name__=="__main__":
    args = parser.parse_args()
    main(args)
