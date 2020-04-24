import glob
import os
import h5py
import imageio
import numpy as np


def tif_to_h5(raw_dir, seg_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for raw_file, seg_file in zip(
            sorted(glob.glob(os.path.join(raw_dir, '*.tif'))),
            sorted(glob.glob(os.path.join(seg_dir, '*.tif'))),
    ):
        raw_filename = os.path.split(raw_file)[1]
        seg_filename = os.path.split(seg_file)[1]
        assert raw_filename == seg_filename

        # read tifs
        raw_im = np.asarray(imageio.volread(raw_file))
        seg_im = np.asarray(imageio.volread(seg_file))

        # save to h5
        out_path = os.path.join(out_dir, os.path.splitext(raw_filename)[0] + '.h5')
        print(f'Saving to: {out_path}')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('raw', data=raw_im, compression='gzip')
            f.create_dataset('label', data=seg_im, compression='gzip')


if __name__ == '__main__':
    raw_dir = '/home/adrian/workspace/ilastik-datasets/Takafumi/raw_data'
    seg_dir = '/home/adrian/workspace/ilastik-datasets/Takafumi/chosen_segmentation'
    out_dir = '/home/adrian/workspace/ilastik-datasets/Takafumi/gt'
    tif_to_h5(raw_dir, seg_dir, out_dir)
