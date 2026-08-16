"""
Simulate a synthetic DCE-MRI dataset for testing evaluation code.

Creates 4D NIfTIs (x, y, z, time) with a cylindrical ROI that exhibits a
typical DCE contrast-agent uptake curve (baseline -> wash-in -> washout),
plus matching binary ROI masks. Everything outside the ROI is low-level
background noise.

Output structure:

    <output_dir>/
        reconstructed_images/
            Pt1/scan1.nii.gz
            Pt1/scan2.nii.gz
            Pt2/scan1.nii.gz
            ...
        masks/
            Pt1/mask1.nii.gz
            Pt1/mask2.nii.gz
            ...
"""

from pathlib import Path
import os
import numpy as np
import nibabel as nib


def make_cylinder_mask(shape, center=None, radius=8, axis=2, length=None, inner_radius=0):
    """Boolean cylinder (optionally hollow) mask inside a 3D volume.

    Set `inner_radius` > 0 to carve out the lumen, leaving a tube/ring
    cross-section - useful for mimicking a vessel wall (e.g. AAA wall).

    Parameters
    ----------
    shape : tuple of int
        (nx, ny, nz) volume shape.
    center : tuple of int, optional
        Cylinder center voxel. Defaults to the volume center.
    radius : float
        Outer cylinder radius in voxels (in the plane perpendicular to `axis`).
    axis : int
        Axis (0, 1, or 2) along which the cylinder extends.
    length : float, optional
        Cylinder length in voxels along `axis`. Defaults to shape[axis] / 2.
    inner_radius : float
        Inner (lumen) radius in voxels. 0 gives a solid cylinder (default,
        backward compatible). Must be < radius.
    """
    nx, ny, nz = shape
    if center is None:
        center = (nx // 2, ny // 2, nz // 2)
    if length is None:
        length = shape[axis] / 2
    if inner_radius >= radius:
        raise ValueError("inner_radius must be smaller than radius")

    xx, yy, zz = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    coords = [xx, yy, zz]
    other_axes = [i for i in range(3) if i != axis]

    r = np.sqrt(
        (coords[other_axes[0]] - center[other_axes[0]]) ** 2
        + (coords[other_axes[1]] - center[other_axes[1]]) ** 2
    )
    axis_dist = np.abs(coords[axis] - center[axis])

    return (r <= radius) & (r >= inner_radius) & (axis_dist <= length / 2)


def dce_uptake_curve(n_timepoints, t0=3, tau_up=1.5, tau_down=15, amplitude=1.0, baseline=1.0):
    """Simple wash-in/washout DCE signal curve, normalized around `baseline`.

    Flat at `baseline` until t0, then rises with time constant `tau_up`
    and decays back down with time constant `tau_down`.
    """
    t = np.arange(n_timepoints)
    curve = np.full(n_timepoints, baseline, dtype=float)
    post = t >= t0
    tt = t[post] - t0
    curve[post] = baseline + amplitude * (1 - np.exp(-tt / tau_up)) * np.exp(-tt / tau_down)
    return curve


def simulate_patient_scan(shape, n_timepoints, wall_mask, lumen_mask=None, baseline_signal=100.0,
                           roi_params=None, noise_std=5.0, rng=None):
    """Simulate one 4D (x, y, z, t) DCE volume.

    `wall_mask` gets the DCE wash-in/washout curve. `lumen_mask` (the vessel
    lumen) gets a flat, low-level blood-suppressed signal, as expected for a
    black-blood sequence - this keeps SNR/CNR computations meaningful instead
    of degenerate.
    """
    if rng is None:
        rng = np.random.default_rng()
    if roi_params is None:
        roi_params = dict(t0=3, tau_up=1.5, tau_down=15, amplitude=0.8)

    nx, ny, nz = shape
    data = np.zeros((nx, ny, nz, n_timepoints), dtype=np.float32)

    # Background: low-level constant + noise everywhere
    background = rng.normal(
        loc=baseline_signal * 0.1, scale=noise_std * 0.3,
        size=(nx, ny, nz, n_timepoints),
    )
    data += np.clip(background, 0, None)

    # Wall: baseline_signal scaled by the DCE curve, with mild voxel-wise
    # variability and noise on top
    curve = dce_uptake_curve(n_timepoints, baseline=1.0, **roi_params)
    voxel_variability = rng.normal(loc=1.0, scale=0.05, size=(nx, ny, nz))[..., None]
    wall_signal = baseline_signal * curve[None, None, None, :] * voxel_variability
    wall_signal += rng.normal(0, noise_std, size=wall_signal.shape)

    data[wall_mask] = wall_signal[wall_mask]

    if lumen_mask is not None and lumen_mask.any():
        # Blood-suppressed lumen: flat, low signal (black-blood sequence)
        lumen_signal = rng.normal(
            loc=baseline_signal * 0.15, scale=noise_std * 0.5,
            size=(nx, ny, nz, n_timepoints),
        )
        data[lumen_mask] = np.clip(lumen_signal, 0, None)[lumen_mask]

    return data.astype(np.float32)


def simulate_dce_dataset(output_dir=".", n_patients=5, n_scans=2, shape=(64, 64, 32),
                          n_timepoints=20, roi_radius=8, roi_inner_radius=5, roi_axis=2,
                          roi_length=None, voxel_size=(2.0, 2.0, 3.0), noise_std=5.0, seed=42):
    """Generate the full synthetic dataset and write it to disk.

    Parameters
    ----------
    output_dir : str or Path
        Root folder. `reconstructed_images/` and `masks/` are created inside it.
    n_patients : int
        Number of simulated patients (folders Pt1 ... PtN).
    n_scans : int
        Number of repeated scans per patient (scan1, scan2, ...).
    shape : tuple of int
        3D spatial shape (nx, ny, nz).
    n_timepoints : int
        Number of dynamic (DCE) timepoints.
    roi_radius, roi_inner_radius, roi_axis, roi_length : see `make_cylinder_mask`.
        `roi_inner_radius` hollows out the lumen, leaving a wall-like tube
        (e.g. to mimic an aneurysm wall). Set to 0 for a solid cylinder.
    voxel_size : tuple of float
        Voxel size in mm, used to build the NIfTI affine.
    noise_std : float
        Background/ROI noise standard deviation.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    img_root = output_dir / "reconstructed_images"
    mask_root = output_dir / "masks"

    affine = np.diag(list(voxel_size) + [1.0])

    for p in range(1, n_patients + 1):
        for s in range(1, n_scans + 1):
            pt_id = f"M_B1_0{p}_0{s}"
            img_dir = Path(output_dir/ img_root/ pt_id/'DCE'/'recon_ID')
            mask_dir = Path(output_dir/ mask_root/ pt_id /'mask')
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            # Per-patient variability so not every patient looks identical
            jitter = rng.integers(-2, 3, size=3)
            center = (
                shape[0] // 2 + int(jitter[0]),
                shape[1] // 2 + int(jitter[1]),
                shape[2] // 2 + int(jitter[2]),
            )
            patient_amplitude = 0.6 + 0.4 * rng.random()
            patient_tau_down = 10 + 10 * rng.random()


            wall_mask = make_cylinder_mask(
                shape, center=center, radius=roi_radius, axis=roi_axis,
                length=roi_length, inner_radius=roi_inner_radius,
            )
            lumen_mask = make_cylinder_mask(
                shape, center=center, radius=roi_inner_radius, axis=roi_axis,
                length=roi_length, inner_radius=0,
            ) if roi_inner_radius > 0 else None

            roi_params = dict(
                t0=3,
                tau_up=1.5 + 0.5 * rng.random(),
                tau_down=patient_tau_down,
                amplitude=patient_amplitude,
            )
            data = simulate_patient_scan(
                shape, n_timepoints, wall_mask, lumen_mask=lumen_mask,
                roi_params=roi_params, noise_std=noise_std, rng=rng,
            )

            # Labeled mask: 1 = Wall, 3 = Lumen (see utilities.convert_mask_index_to_organ)
            labeled_mask = np.zeros(shape, dtype=np.uint8)
            labeled_mask[wall_mask] = 1
            if lumen_mask is not None:
                labeled_mask[lumen_mask] = 3

            nib.save(nib.Nifti1Image(data, affine), img_dir / f"magn_scan{s}.nii")
            nib.save(
                nib.Nifti1Image(labeled_mask, affine),
                mask_dir / f"mask{s}.nii",
            )

    print(f"Simulated dataset written to: {output_dir.resolve()}")
    return img_root, mask_root


if __name__ == "__main__":
    simulate_dce_dataset(output_dir="./", n_patients=5, n_scans=2, shape=(64, 64, 32))