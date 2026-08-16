import numpy as np
import os
import torchio as tio
from utilities import convert_mask_index_to_organ, get_patient_eval_path
from data_loading import get_subject_with_images
import matplotlib.pyplot as plt
from plotting import plot_mid_slice_mid_time, save_all_images

# np.trapz was removed in newer NumPy releases in favor of np.trapezoid;
# fall back to whichever the installed NumPy version provides.
trapz = getattr(np, 'trapezoid', None) or np.trapz


def fit_ave_sig_change(img, mask, samplings):
    nr_samples = img.shape[0]
    selected_pixels = img[:, mask.squeeze() == 1]  # Select pixels where mask is idx_max_select
    mean_intensity = np.mean(selected_pixels, axis=1)  # Mean across selected pixels    x = np.arange(len(signal))

    x_fit = samplings[nr_samples // 3:]  # Use the last two thirds of the data for fitting
    y_fit = mean_intensity[nr_samples // 3:]  # Corresponding mean intensity values
    m,b = np.polyfit(x_fit, y_fit, 1)  # Fit a linear model to the last two thirds of the data
    return m,b


def calc_cov(meas_1, meas_2):
    #within subject coefficient of variation
    #calc according to https://www-users.york.ac.uk/~mb55/meas/cv.htm
    s2 = (meas_1 - meas_2) ** 2 / 2 # within_sub_var
    m = (meas_1 + meas_2) / 2 #sub_mean
    s2m2 = s2 / m**2
    wCV = np.sqrt(np.mean(s2m2))
    return wCV * 100


def get_static_eval_param():
    idx_calc_enhanc = 2
    del_time_washin = 2
    return idx_calc_enhanc, del_time_washin


def calc_kalifa_parameters(eval_settings, subject):
    RE, mask, img = get_RE_and_mask_and_img(subject)
    time_points = np.linspace(0, eval_settings.get('scan_dur'), img.shape[0])

    unique_labels = [1] #only evaluate wall
    dce_param = {}
    for lbl in unique_labels:
        re_masked = RE.get(convert_mask_index_to_organ(lbl))
        ttp_idx = re_masked.argmax()
        dce_param['TTP'] = time_points[ttp_idx]-32  # subtracting 32 seconds to account for the time before contrast arrival, as per Kalifa et al. 2014
        dce_param['AUC'] = round(trapz(re_masked)*time_points[1])  # AUC over entire curve, multiplied by time step
        dce_param['AUC_twothird'] = round(trapz(re_masked[img.shape[0] // 3:])*time_points[1])
        dce_param['ME'] = re_masked.max()
        idx_calc_enhanc, del_time_washin = get_static_eval_param()
        dce_param['wash_in'] = ((re_masked[del_time_washin + idx_calc_enhanc] - re_masked[idx_calc_enhanc])
                   / (time_points[idx_calc_enhanc + del_time_washin] - time_points[idx_calc_enhanc]))
        dce_param['wash_out'] = (dce_param['ME'] - re_masked[-1]) / (time_points[-1] - dce_param['TTP'])
    subject['dce_param'] = dce_param
    return subject


def get_mid_slice_masks(img_midslice, mask_midslice):
    mask_wall_midslice = np.where(mask_midslice == 1, mask_midslice, 0)

    from scipy.ndimage import label, binary_fill_holes
    mask_midslice = mask_wall_midslice.astype(bool)
    labeled, num = label(mask_midslice)

    # keep largest connected region
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()

    main_region = labeled == largest
    filled = binary_fill_holes(main_region)
    inner_part = filled & ~main_region

    # img_mid_mean = np.mean(img_midslice, axis=0)
    return mask_wall_midslice, inner_part


def get_background_mask(img):
    bg_mask = np.zeros(img.shape[1:], dtype=bool)

    # corners (safer assumption = air)
    bg_mask[0:10, 0:10,10:20] = True
    bg_mask[-10:, 0:10,10:20] = True
    bg_mask[0:10, -10:,10:20] = True
    bg_mask[-10:, -10:,10:20] = True

    return bg_mask


def plot_mask_test(eval_settings, mask_wall_midslice, inner_part, img_midslice, bg_mask, subject):
    # --- plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask_wall_midslice, cmap="gray")
    axes[0].set_title("Original mask")
    axes[0].axis("off")
    axes[1].imshow(inner_part, cmap="gray")
    axes[1].imshow(mask_wall_midslice + 2* inner_part + 4*bg_mask, cmap="gray")


    axes[1].set_title("Inner part (holes)")
    axes[1].axis("off")
    axes[2].imshow(img_midslice[0, :, :], cmap="gray")
    axes[2].set_title("Image midslice")
    axes[2].axis("off")
    plt.tight_layout()

    save_name = f'{subject.get("name")}_mask_test_scan{subject.get("idx_scan")}.png'
    plt.savefig(os.path.join(get_patient_eval_path(eval_settings, subject.get('pt_idx')), save_name))
    if eval_settings.get('show_plots', False):
        plt.show()
    else:
        plt.close()



def compute_lumen_wall_signal_over_time(eval_settings, img, mask_wall, mask_lumen, bg_mask, subject):
    """
    img: [T, X, Y]
    masks: [X, Y] boolean
    """

    # --- extract signals ---
    wall_signal = img[:, mask_wall].mean(axis=1)
    lumen_signal = img[:, mask_lumen].mean(axis=1)

    # --- noise estimate (from background ROI) ---
    noise_vals = img[:, bg_mask.astype(bool)]
    noise_sigma_t = noise_vals.std(axis=1)  # time-resolved noise

    # avoid division by zero
    noise_sigma_t[noise_sigma_t == 0] = np.nan

    # --- compute SNR and CNR over time ---
    snr_t = wall_signal / noise_sigma_t
    cnr_t = (wall_signal - lumen_signal) / noise_sigma_t

    # --- summary metrics (what you actually report) ---
    snr_mean = np.nanmean(snr_t)
    cnr_mean = np.nanmean(cnr_t)
    print(f'SNR: {snr_mean}, CNR: {cnr_mean}')
    # --- plot (optional) ---
    plt.figure(figsize=(10, 5))
    plt.plot(wall_signal, label="Wall signal")
    plt.plot(lumen_signal, label="Lumen signal")
    plt.plot(snr_t, label="SNR")
    plt.plot(cnr_t, label="CNR")
    plt.xlabel("Time")
    plt.ylabel("Signal / Ratio")
    plt.title("Signal, SNR, and CNR over time")
    plt.legend()
    plt.tight_layout()

    save_name = f'{subject.get("name")}_Lumen_Wall_SNR_scan{subject.get("idx_scan")}.png'
    plt.savefig(os.path.join(get_patient_eval_path(eval_settings, subject.get('pt_idx')), save_name))
    if eval_settings.get('show_plots', False):
        plt.show()
    else:
        plt.close()

    return {
        "wall_signal": wall_signal,
        "lumen_signal": lumen_signal,
        "snr_t": snr_t,
        "cnr_t": cnr_t,
        "snr_mean": snr_mean,
        "cnr_mean": cnr_mean
    }


def calc_noise_estimate(eval_settings, subject):
    RE, mask, img = get_RE_and_mask_and_img(subject)
    mask_wall = mask==1
    mask_lumen = mask==3
    mask_wall = mask_wall[0,:]
    mask_lumen = mask_lumen[0,:]
    bg_mask = get_background_mask(img)
    plot_mask_test(eval_settings, mask_wall[:,:,15], mask_lumen[:,:,15], img[:,:,:,15], bg_mask[:,:,15], subject)

    results = compute_lumen_wall_signal_over_time(
        eval_settings,
        img,
        mask_wall.astype(bool),
        mask_lumen.astype(bool),
        bg_mask,
        subject
    )
    subject['snr_mean'] = results['snr_mean']
    subject['cnr_mean'] = results['cnr_mean']

    return subject

def evaluate_subject(subject, eval_settings):
    """
    Evaluate the subject by creating AUC maps and fitting signal changes.
    """
    subject = calc_kalifa_parameters(eval_settings, subject)
    subject = calc_noise_estimate(eval_settings, subject)
    plot_kalifa_parameters_per_scan(subject, eval_settings)

    if eval_settings.get('plot_mid_slice_per_patient', True):
        fig, ax = plot_mid_slice_mid_time(subject, eval_settings)

    if eval_settings.get('save_all_images', True):
        save_all_images(subject)

    return subject


def plot_kalifa_parameters_per_scan(subject, eval_settings):
    """
    Compute Kalifa et al. 2014 semi-quantitative DCE-MRI parameters per subject.
    Voxel-wise maps are created temporarily for computation but not saved to the subject.

    Parameters
    ----------
    subject : dict
        Subject dictionary with 'image' and 'mask'.
    scan_dur : float
        Total scan duration in seconds or minutes.

    Returns
    -------
    subject : dict
        Updated subject dictionary with Kalifa ROI fit results only.
    """
    # unique_labels, fit_results = calc_kalifa_parameters(eval_settings, subject)
    RE, mask, img = get_RE_and_mask_and_img(subject)
    unique_labels = [1] #only evaluate wall
    time_points = np.linspace(0, eval_settings.get('scan_dur'), img.shape[0])
    idx_calc_enhanc, del_time_washin = get_static_eval_param()
    idx_time_twothird = img.shape[0] // 3

    for lbl in unique_labels:
        segment_name = convert_mask_index_to_organ(lbl)
        re_masked = RE.get(convert_mask_index_to_organ(lbl))

        plt.figure()
        plt.plot(time_points, re_masked, label='Mean ROI Curve')
        plt.xlabel('Time')
        plt.ylabel('Relative Enhancement')
        #subject name
        plt.title(f'{subject.get("name")} \n Kalifa, {segment_name}')

        # annotate at TTP, ME point
        dce_param = subject.get('dce_param', {})
        plt.annotate(f'ME ({dce_param["ME"]:.4f})\nTTP ({dce_param["TTP"]:.2f})',
                        xy=(dce_param['TTP'], dce_param['ME']),
                        xytext=(dce_param['TTP'] + 0.1 * eval_settings.get('scan_dur'), dce_param['ME'] - 0.1 * dce_param['ME']),
                        arrowprops=dict(arrowstyle='->', color='black'))


        #draw wash-in line
        plt.plot([time_points[idx_calc_enhanc], time_points[idx_calc_enhanc+del_time_washin]],
                    [re_masked[idx_calc_enhanc], re_masked[idx_calc_enhanc+del_time_washin]],
                    color='orange', linestyle=':', label=f'Wash-in rate {dce_param["wash_in"]:.4f}')
        #draw wash-out line
        plt.plot([dce_param['TTP'], time_points[-1]],
                    [dce_param['ME'], re_masked[-1]],
                    color='purple', linestyle=':', label=f'Wash-out rate {dce_param["wash_out"]:.4f}')

        plt.legend()
        save_name = f'{subject.get("name")}_Kalifa_{segment_name}_scan{subject.get("idx_scan")}.png'
        plt.savefig(os.path.join(get_patient_eval_path(eval_settings, subject.get('pt_idx')), save_name))
        if eval_settings.get('show_plots', False):
            plt.show()
        else:
            plt.close()

    return subject


def get_RE_and_mask_and_img(subject):
    subject_w_images = get_subject_with_images(subject)
    img = subject_w_images['image'].data.numpy()  # shape: (T, X, Y, Z)
    mask = subject_w_images['mask'].data.numpy()  # shape: (1, X, Y, Z)
    RE = {}
    #calc signal averaged over mask
    for idx_mask in np.unique(mask):
        if idx_mask == 0:
            continue
        sel_mask = (mask == idx_mask)
        sel_sig = np.mean(img[:, sel_mask.squeeze()], axis=1)
        sel_sig_ini = np.mean(img[0:3, sel_mask.squeeze()])
        rel_sig_enh = (sel_sig - sel_sig_ini) / np.maximum(sel_sig_ini, 1e-6)
        RE[convert_mask_index_to_organ(idx_mask)] = rel_sig_enh

    return RE, mask, img
