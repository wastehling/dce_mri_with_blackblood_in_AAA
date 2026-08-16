import os
import numpy as np
import matplotlib.pyplot as plt
import torchio as tio
import pandas as pd
from data_loading import get_subject_with_images
from helpers_statistics import get_paired_sample_statistics
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

FIGWIDTH_DOUBLE = 7.0
FIGHEIGHT_DOUBLE = 3.5
FIGWIDTH_SINGLE = 3.4
FIGHEIGHT_SINGLE = 2.4


def plot_mid_slice_mid_time(subject, eval_settings,  img_to_plot='image'):
    print(f'Starting to plot mid slice at mid time for subject: {subject.get("name")}')
    subject_w_img = get_subject_with_images(subject)
    img = subject_w_img[img_to_plot].data.numpy()  # shape: (1, X, Y, Z, T) or (T, X, Y, Z)
    mask = subject_w_img['mask'].data.numpy()  # shape: (1, X, Y, Z)

    #select 3 timepoints
    time = [0, img.shape[0] // 2, img.shape[0] - 1]
    slice = [img.shape[3] // 4, img.shape[3] // 2, (img.shape[3] // 4)*3]   # Middle slice in the first dimension

    fig, ax = plt.subplots(3, 3, figsize=(14, 14))


    for x, idx_time in enumerate(time):
        for y, idx_slice in enumerate(slice):
            if idx_time >= img.shape[0] or idx_slice >= img.shape[3]:
                raise ValueError(f"Time index {idx_time} or slice index {idx_slice} out of bounds for image shape {img.shape}.")
            img_slice = img[idx_time, :, :, idx_slice]  # Extract the middle slice at the middle time point
            mask_slice = mask[0, :, :, idx_slice]  # Extract the corresponding mask slice

            # --- Focus on mask region only ---
            # Find bounding box of mask
            mask_indices = np.argwhere(mask_slice)
            if mask_indices.size == 0:
                print(f'Warning: Mask is empty on this slice for subject {subject.get("name")}. Skipping this slice {idx_slice}.')
                continue
            x_min, y_min = mask_indices.min(axis=0)
            x_max, y_max = mask_indices.max(axis=0) + 1  # +1 for slicing

            # Pad slightly (optional)
            pad = 50
            x_min = max(x_min - pad, 0)
            x_max = min(x_max + pad, img_slice.shape[0])
            y_min = max(y_min - pad, 0)
            y_max = min(y_max + pad, img_slice.shape[1])

            # Crop to region of interest
            cropped_img = img_slice[x_min:x_max, y_min:y_max]
            cropped_mask = mask_slice[x_min:x_max, y_min:y_max]

            # --- Plot ---
            im = ax[x,y].imshow(cropped_img, cmap='gray')

            for idx_mask in np.unique(cropped_mask):
                if idx_mask == 0:
                    continue
                mask_region = np.zeros_like(cropped_mask)
                mask_region[cropped_mask == idx_mask] = 1
                contour = ax[x,y].contour(cropped_mask, linewidths=1)
                # for c in contour.collections:
                #     c.set_alpha(0.2)  # Set transparency for contour lines
            ax[x,y].set_title(f'Slice {idx_slice}, Time {idx_time}')
            ax[x,y].axis('off')

    fig.tight_layout()
    fig.suptitle(f'3 Slices at different timepoints, cropped to mask. Pt: {subject.get("name")}', fontsize=16)
    plt.subplots_adjust(top=0.9)  # Adjust top to make room for subtitle
    fig.savefig(os.path.join(subject['save_path'], f'Slices_{img_to_plot}_snapshot_scan{subject.get("name")}.png'))
    if eval_settings.get('show_plots', True):
        plt.show()
    else:
        plt.close()
    return fig, ax


def plot_mean_wall_intensity_over_time(subjects, eval_settings, image_key='image', mask_key='mask', show=True,  normalization=True):
    """
    Plot mean intensity over time within a binary mask from a torchio.Subject,
    regardless of time being first or last dimension.
    """
    if not subjects:
        print("No subjects provided for plotting.")
        return


    fig = plt.figure(figsize=(FIGWIDTH_DOUBLE, FIGHEIGHT_DOUBLE))
    markers = ['s', '.']
    #red c31e23, teal 0d7d87 to colors
    red = '#d41f11'
    orange = '#f47a00'
    light_blue = '#62c8d3'
    dark_blue = '#007191'
    colors = [red, dark_blue, orange, light_blue]

    for idx, subject in enumerate(subjects):
        from calculations import get_RE_and_mask_and_img
        RE, mask, img = get_RE_and_mask_and_img(subject)
        wall_re = RE['Wall'] #relative enhancement wall

        nr_samples = wall_re.shape[0]
        t_sample = np.linspace(0, eval_settings.get('scan_dur'), nr_samples)

        lbl = 'Baseline' if subject.get('idx_scan') == 1 else 'Follow-up'
        plt.plot(t_sample, wall_re, marker=markers[idx], color=colors[idx],
                 label=lbl)

        # Annotation
        dce_param = subject.get('dce_param', {})
        annot = (f'TTP: {dce_param.get("TTP", np.nan):.0f}sec, ME: {dce_param.get("ME", np.nan):.1f}, '
                 f'AUC: {round(dce_param.get("AUC", np.nan)):.0f}sec')
        ax = plt.gca()  # get current axes
        y_offset = 0.05 + idx * 0.08  # stack annotations upward

        plt.annotate(annot, xy=(0.6, y_offset), xycoords='axes fraction',
                     fontsize=8, color=colors[idx],
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='none', alpha=0.7))

    plt.xlim([0, eval_settings.get('scan_dur')])
    plt.annotate('Contrast \n Injection', xy=(32, 0), xytext=(10, 0.4),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=8)

    plt.title("Relative Signal Enhancement Aortic Wall")
    plt.xlabel('Time [sec]')
    plt.ylabel('Relative Signal Enhancement')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # IF pt_idx == 1 or 17, annotate with a) or b) in upper left corner
    if subject.get('pt_idx') in [1, 17]:
        panel_label = 'a)' if subject.get('pt_idx') == 1 else 'b)'
        plt.annotate(panel_label, xy=(-0.075, 1.1), xycoords='axes fraction',
                     fontsize=12, fontweight='bold',
                     horizontalalignment='left', verticalalignment='top')

    #plot vertical dashed line at time point 4
    fig.savefig(os.path.join(subject['save_path'], f'Masked_signal_scan_rescan_Pt{subject.get("pt_idx")}.tiff'), dpi=300)
    if eval_settings.get('show_plots', True):
        plt.show()
    plt.close(fig)


def save_all_images(subject):
    """
    Save all images in the subject to the save path.
    """
    save_path = subject['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for key, image in subject.items():
        if isinstance(image, tio.ScalarImage) or isinstance(image, tio.LabelMap):
            save_name = f'{key}_scan{subject.get("idx_scan")}.nii'
            image.save(os.path.join(save_path, save_name))
            print(f'Saved {key} to {os.path.join(save_path, save_name)}')
        else:
            # print(f'Skipping {key}, not a TorchIO image.')
            continue

