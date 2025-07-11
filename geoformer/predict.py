# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

import os
from inference import InferenceModel

import torch
import h5py
import numpy as np


def predict(args):
    '''Example of prediction'''
    args.lr = float(args.lr)
    args.decay_epochs = int(args.decay_epochs)
    args.decay_rate = float(args.decay_rate)

    # Initialize the inference model
    im = InferenceModel(args)

    predicted_labels = []
    h5_path = args.predict_input
    print(f"Loading patches from: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        patches_dset = f['patches']
        num_patches = patches_dset.shape[0]
        print(f"  Number of patches: {num_patches}")

        for i in range(num_patches):
            patch = patches_dset[i]
            features = torch.tensor(patch[1:], dtype=torch.float32).unsqueeze(0)
            cluster_scores = im(features)[0]
            out_label = np.argmax(cluster_scores, axis=0)
            predicted_labels.append(out_label)

    predicted_labels = np.stack(predicted_labels, axis=0)

    # save the prediction result to a h5 file.
    dir_name, base_name = os.path.split(h5_path)
    name, ext = os.path.splitext(base_name)
    output_name = f"{name}_prediction{ext}"
    output_path = os.path.join(dir_name, output_name)

    with h5py.File(output_path, 'w') as outfile:
        outfile.create_dataset('labels', data=predicted_labels, compression='gzip')

    print(f"[DONE] Prediction saved at: {output_path}")


if __name__ == '__main__':
    import argparse
    import yaml
    from types import SimpleNamespace

    def convert_dict_to_namespace(d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = convert_dict_to_namespace(value)
        return SimpleNamespace(**d)

    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    args = convert_dict_to_namespace(config)

    predict(args)


# python predict.py --config ./configs/config.yaml
