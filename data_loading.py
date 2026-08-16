import os
import SimpleITK as sitk
import torchio as tio
from utilities import get_patient_identifier, get_patient_eval_path, get_evaluation_settings
from utilities import extract_reconname_from_recon_folder


def load_dicom_partial(dicom_folder):
    """
    Load a subset of slices from a DICOM series as a TorchIO ScalarImage.

    Parameters:
        dicom_folder (str): Path to the folder containing the DICOM series.
        start_slice (int): Starting slice index (1-based).
        end_slice (int): Ending slice index (1-based, inclusive).

    Returns:
        torchio.ScalarImage: TorchIO image with selected slices.
    """
    if dicom_folder is None:
        return None
    # Step 1: Read the full DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_files)
    full_image = reader.Execute()

    # Step 2: Convert to NumPy
    full_array = sitk.GetArrayFromImage(full_image)  # (num_slices, H, W)

    # Step 3: Select the slice range (convert to 0-based indexing)
    start_slice = full_array.shape[0] // 4 *2+1
    end_slice = full_array.shape[0] // 4 *3
    selected_array = full_array[start_slice - 1:end_slice]

    # Step 4: Create new SimpleITK image from selected slices
    partial_image = sitk.GetImageFromArray(selected_array)

  # Step 5: Copy spacing and direction
    partial_image.SetSpacing(full_image.GetSpacing())
    partial_image.SetDirection(full_image.GetDirection())


    # Step 6: Convert to TorchIO ScalarImage
    tio_image = tio.ScalarImage.from_sitk(partial_image)

    return tio_image

def get_subject_with_images(subject):
    subject = tio.Subject(image=tio.ScalarImage(subject['paths']['nii_recon_dce']),
                          mask=tio.LabelMap(subject['paths']['nii_mask']),
                          name=subject['name'],
                          b1_map=load_dicom_partial(subject['paths']['dcm_b1']) )

    transformation_func = tio.Resample(subject['image'], image_interpolation='linear')
    resampled_mask = transformation_func(subject['mask'])
    subject['mask'] = resampled_mask
    return subject


def create_subject(eval_settings, pt_idx, idx_scan, recon_folder):
    path_nii_recon_dce, path_mask_nii, path_b1_dcm = \
        get_paths_for_patient(eval_settings, pt_idx, idx_scan)
    pt_name = get_patient_identifier(eval_settings.get('study_to_eval'), pt_idx, 'postprocessing', idx_scan=idx_scan)
    subject = {}
    subject['name'] = f'{pt_name}_{extract_reconname_from_recon_folder(recon_folder)}'
    subject['save_path'] = get_patient_eval_path(eval_settings, pt_idx)
    subject['pt_idx'] = pt_idx
    subject['idx_scan'] = idx_scan
    subject['recon_folder'] = recon_folder
    subject['paths'] = {
        'nii_recon_dce': path_nii_recon_dce,
        'nii_mask': path_mask_nii,
        'dcm_b1': path_b1_dcm
    }
    return subject


def get_path_to_dce_nii(eval_settings, pt_idx, idx_scan, recon_name):
    pt_id = get_patient_identifier(eval_settings.get('study_to_eval'), pt_idx, 'recon', idx_scan=idx_scan)
    path_to_recon = eval_settings.get('path_to_recon')
    path_base_recon = os.path.join(path_to_recon, pt_id, 'DCE')
    recon_folder = os.path.join(path_base_recon, recon_name)
    # Check if the recon folder exists
    if not os.path.exists(recon_folder):
        raise ValueError(f'Reconstruction folder {recon_folder} does not exist.')
    # List files starting with 'magn_' and ending with '.nii'
    nii_files = [f for f in os.listdir(recon_folder) if f.startswith('magn_') and f.endswith('.nii')]
    if len(nii_files) != 1:
        raise ValueError(f'Expected one nii file starting with magn_ in {recon_folder}, found {len(nii_files)} files.')
    path_nii_recon_dce = os.path.join(recon_folder, nii_files[0])
    print(f'Path to DCE NIfTI: {path_nii_recon_dce}')
    return path_nii_recon_dce


def get_path_to_mask(eval_settings, pt_idx, idx_scan):
    pt_id = get_patient_identifier(eval_settings.get('study_to_eval'), pt_idx, 'recon', idx_scan=idx_scan)
    path_base_recon = os.path.join(eval_settings.get('path_to_postprocessing'), pt_id, eval_settings.get('mask_name'))
    # Check if the mask folder exists
    if not os.path.exists(path_base_recon):
        raise ValueError(f'Mask folder {path_base_recon} does not exist.')
    # List files ending with '.nii'
    nii_files = [f for f in os.listdir(path_base_recon) if f.endswith('.nii')]
    if len(nii_files) != 1:
        raise ValueError(f'get_path_to_mask: Expected one nii file in {path_base_recon}, found {len(nii_files)} files.')
    path_nii_mask = os.path.join(path_base_recon, nii_files[0])
    print(f'Path to mask NIfTI: {path_nii_mask}')
    return path_nii_mask

def get_path_to_b1_dcm(eval_settings, pt_idx, idx_scan):
    pt_id = get_patient_identifier(1, pt_idx, 'recon', idx_scan=idx_scan)
    path_base_recon = os.path.join(eval_settings.get('path_to_dicom'), pt_id)
    #list folders having B1_map in their name
    b1_folders = [f for f in os.listdir(path_base_recon) if 'B1_map' in f and '.zip' not in f]
    if len(b1_folders) != 1:
        raise ValueError(f'Expected one folder with B1_map in {path_base_recon}, found {len(b1_folders)} folders.')
    path_b1_dcm = os.path.join(path_base_recon, b1_folders[0])
    return path_b1_dcm

def get_paths_for_patient(eval_settings, pt_idx, idx_scan):
    recon_folder_name = get_recon_folder_path_for_pt(eval_settings, pt_idx, idx_scan)
    path_nii_recon_dce = get_path_to_dce_nii(eval_settings, pt_idx, idx_scan, recon_folder_name)
    path_mask_nii = get_path_to_mask(eval_settings, pt_idx, idx_scan)
    # path_b1_dcm = get_path_to_b1_dcm(pt_idx, idx_scan)
    # return path_nii_recon_dce, path_mask_nii, path_b1_dcm
    return path_nii_recon_dce, path_mask_nii, None


def get_recon_folder_path_for_pt(eval_settings, pt_idx, idx_scan):
    recon_name_must_contain = eval_settings.get('recon_name_must_contain')
    if recon_name_must_contain is None or len(recon_name_must_contain) != 1:
        raise ValueError('recon_name_must_contain is not specified in eval_settings.')
    recon_folder_name = recon_name_must_contain[0]

    pt_id = get_patient_identifier(eval_settings.get('study_to_eval'), pt_idx, 'recon', idx_scan=idx_scan)
    path_to_recon = eval_settings.get('path_to_recon')
    path_base_recon = os.path.join(path_to_recon, pt_id, 'DCE')

    #list folders having recon_folder_name in their name at path_base_recon
    recon_folders = [f for f in os.listdir(path_base_recon) if recon_folder_name in f]
    if len(recon_folders) != 1:
        raise ValueError(f'Expected one folder with {recon_folder_name} in {path_base_recon}, found {len(recon_folders)} folders.')
    recon_folder_name = recon_folders[0]
    full_recon_folder_path = os.path.join(path_base_recon, recon_folder_name)
    return full_recon_folder_path
