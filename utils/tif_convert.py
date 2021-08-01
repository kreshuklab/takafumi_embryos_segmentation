import gzip
import os
import shutil

import h5py
import tifffile
import glob


def read_file(path):
    with gzip.open(path, "rb") as f_in:
        out_path = os.path.splitext(path)[0]
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return tifffile.imread(out_path)


def main(in_path):
    raw_dir = os.path.join(in_path, 'membranes')
    seg_dir = os.path.join(in_path, 'segmentation')

    raw_filenames = glob.glob(os.path.join(raw_dir, '*.tif'))
    sorted(raw_filenames)
    seg_filenames = glob.glob(os.path.join(seg_dir, '*.tif'))
    sorted(seg_filenames)

    for raw_file, seg_file in zip(raw_filenames, seg_filenames):
        print(raw_file, seg_file)
        raw = tifffile.imread(raw_file)
        seg = tifffile.imread(seg_file)
        seg[seg == 1] = 0

        out_fn = os.path.split(raw_file)[1].split('.')[0] + '.h5'
        out_path = os.path.join(in_path, out_fn)

        print('Saving', out_path)
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('raw', data=raw, compression='gzip')
            f.create_dataset('label', data=seg, compression='gzip')


if __name__ == '__main__':
    in_path = '/home/adrian/Downloads/210727_ground_truth'
    main(in_path)
