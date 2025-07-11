# -------------------------------------------------------------------
# Plot and compare the predictions with the labels.
# -------------------------------------------------------------------

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Temporary workaround

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_and_compare():
    # Read the prediction
    path_prediction = r"../examples/input_data_prediction.h5"
    with h5py.File(path_prediction, 'r') as infile:
        predicted_labels = infile['labels'][:]

    # Read the initial label
    path_label = r"../examples/input_data.h5"
    with h5py.File(path_label, 'r') as infile:
        data = infile['patches'][:]

    n_patches = data.shape[0]

    overlay_cmap = ListedColormap([
        (0.0, 0.0, 0.0, 0.0),  # 0: fully transparent
        (0.0, 0.0, 1.0, 0.4),  # 1: blue
        (1.0, 0.0, 0.0, 0.4)   # 2: red
    ])
    overlay_bounds = [-0.5, 0.5, 1.5, 2.5]
    overlay_norm = BoundaryNorm(overlay_bounds, overlay_cmap.N)

    for i in range(n_patches):
        label = data[i, 0]
        predict = predicted_labels[i]

        # Convert CGMC labels to lithological groups
        mask_group1 = np.isin(label, [1, 2, 3, 4, 5])
        mask_group2 = np.isin(label, [17, 19, 33, 34, 35])
        label_LG = np.where(mask_group1, 1, np.where(mask_group2, 2, 0))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(label_LG, cmap=overlay_cmap, norm=overlay_norm)
        axes[0].set_title('Initial Label (Lithology Group)')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])

        axes[1].imshow(predict, cmap=overlay_cmap, norm=overlay_norm)
        axes[1].set_title('Prediction')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_xticklabels([])
        axes[1].set_yticklabels([])

        plt.subplots_adjust(top=0.929,
                         bottom=0.03,
                         left=0.015,
                         right=0.985,
                         hspace=0.2,
                         wspace=0.007)
        plt.show()
        plt.close()

if __name__ == '__main__':
    plot_and_compare()
