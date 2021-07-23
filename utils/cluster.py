import argparse
import glob
import os

import h5py
import hdbscan
import numpy as np
import vigra
from scipy.ndimage import binary_erosion
from sklearn.cluster import MeanShift


def expand_labels_watershed(seg, raw, erosion_iters=4):
    bg_mask = seg == 0
    # don't need to  do anything if we only have background
    if bg_mask.size == int(bg_mask.sum()):
        return seg

    hmap = vigra.filters.gaussianSmoothing(raw, sigma=1.)

    bg_mask = binary_erosion(bg_mask, iterations=erosion_iters)
    seg_new = seg.copy()
    bg_id = int(seg.max()) + 1
    seg_new[bg_mask] = bg_id

    seg_new, _ = vigra.analysis.watershedsNew(hmap, seeds=seg_new.astype('uint32'))

    seg_new[seg_new == bg_id] = 0
    return seg_new


def cluster(emb, clustering_alg, semantic_mask=None):
    output_shape = emb.shape[1:]
    # reshape (E, D, H, W) -> (E, D * H * W) and transpose -> (D * H * W, E)
    flattened_embeddings = emb.reshape(emb.shape[0], -1).transpose()

    result = np.zeros(flattened_embeddings.shape[0])

    if semantic_mask is not None:
        flattened_mask = semantic_mask.reshape(-1)
        assert flattened_mask.shape[0] == flattened_embeddings.shape[0]
    else:
        flattened_mask = np.ones(flattened_embeddings.shape[0])

    if flattened_mask.sum() == 0:
        # return zeros for empty masks
        return result.reshape(output_shape)

    # cluster only within the foreground mask
    clusters = clustering_alg.fit_predict(flattened_embeddings[flattened_mask == 1])
    # always increase the labels by 1 cause clustering results start from 0 and we may loose one object
    result[flattened_mask == 1] = clusters + 1

    return result.reshape(output_shape)


def cluster_hdbscan(emb, min_size, eps, min_samples=None, semantic_mask=None):
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_size, cluster_selection_epsilon=eps, min_samples=min_samples)
    return cluster(emb, clustering, semantic_mask)


def cluster_ms(emb, bandwidth, semantic_mask=None):
    clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    return cluster(emb, clustering, semantic_mask)


def run_clustering(emb, clustering, delta_var, min_size, expand_labels, remove_largest):
    assert clustering in ['ms', 'hdbscan']
    if clustering == 'hdbscan':
        clusters = cluster_hdbscan(emb, min_size, delta_var)
    else:
        clusters = cluster_ms(emb, delta_var)

    # watershed the empty (i.e. noise) region
    if expand_labels:
        clusters = expand_labels_watershed(clusters, raw)

    if remove_largest:
        ids, counts = np.unique(clusters, return_counts=True)
        clusters[ids[np.argmax(counts)] == clusters] = 0

    return clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment embryos')
    parser.add_argument('--emb_dir', type=str, help='Path to embedding predictions directory', required=True)
    parser.add_argument('--clustering', type=str, help='Clustering algorithm: ms or hdbscan', required=True)
    parser.add_argument('--seg_ds', type=str, help='Output seg dataset name', required=True)
    parser.add_argument('--delta_var', type=float, help='delta_var param', default=0.5)
    parser.add_argument('--min_size', type=int, help='HDBSCAN min_size param', default=50)
    parser.add_argument('--remove_largest', help='Remove largest instance (BG)', action='store_true')
    parser.add_argument('--expand_labels', help='Expand labels with watershed', action='store_true')
    parser.add_argument('--min_instance_size', type=int, help='Min instance size filtering', required=False,
                        default=None)

    args = parser.parse_args()

    assert os.path.isdir(args.emb_dir)

    for file_path in glob.glob(os.path.join(args.emb_dir, '*predictions.h5')):
        _, filename = os.path.split(file_path)

        print(f'Processing {filename}')

        with h5py.File(file_path, 'r+') as f:
            raw_sequence = f['raw_sequence'][:]
            embedding_sequence = f['embedding_sequence1'][:]
            seg_sequence = []

            i = 0
            for raw, emb in zip(raw_sequence, embedding_sequence):
                i += 1
                print(f'Processing patch {i}')
                seg = run_clustering(emb, args.clustering, args.delta_var, args.min_size, args.expand_labels,
                                     args.remove_largest)
                seg_sequence.append(seg)

            if args.seg_ds in f:
                del f[args.seg_ds]

            segments = np.stack(seg_sequence, axis=0)
            f.create_dataset(args.seg_ds, data=segments, compression='gzip')

    print('Done')
