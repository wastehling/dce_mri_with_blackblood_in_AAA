import nibabel as nib
import numpy as np
import os
import yaml
import pickle
import pandas as pd


# patient id is defined as M_B1_02
def get_study_and_idxPt_from_patient_identifier(pt_id):
    # check if pt_id is like M_B1_02, so M_B with study 1 and 02 as pt_idx
    if pt_id[0:3] != 'M_B' or len(pt_id) < 7:
        raise ValueError("Patient identifier must start with 'M_B' followed by study number and patient index.")
    study = int(pt_id[3])
    pt_idx = int(pt_id[5:7])
    return study, pt_idx


def get_ptid_and_idx_from_fittingname(fitting_name):
    # check if fitting_name is like P1030, so P with 4 digits
    if fitting_name[0] != 'P' or len(fitting_name) != 4:
        raise ValueError("Fitting name must start with 'P' followed by at least two digits.")
    study = int(fitting_name[1])
    pt_idx = int(fitting_name[2:4])
    idx_scan = int(fitting_name[4])
    return study, pt_idx, idx_scan


def extract_reconname_from_recon_folder(foldername):
    # foldername has somewhere a V4 in it, return the following
    if 'V4' in foldername:
        parts = foldername.split('V4')
        if len(parts) > 1:
            return parts[1]
        else:
            return parts[0]


def get_patient_identifier(study, pt_idx, format, idx_scan=None):
    """
    Generate a patient identifier based on the study, patient index, format, and optional scan index.
    Args:
        study (int): Study number.
        pt_idx (int): Patient index.
        format (str): Format type, either 'postprocessing' or 'recon'.
        idx_scan (int, optional): Scan index. Required for 'postprocessing' format.
    """
    if format not in ['postprocessing', 'recon']:
        raise ValueError("Format must be either 'postprocessing' or 'recon'.")

    if study == 1:
        if format == 'postprocessing':
            return f'P1{pt_idx:02d}{idx_scan}'
        else: # format == 'recon':
            return f'M_B1_{pt_idx:02d}_0{idx_scan}'
    elif study == 3:
        if format == 'postprocessing':
            s = f'P3{pt_idx//100:1d}{pt_idx%100:02d}'
            print(f'get_patient_identifier: Generated patient identifier: {s}')
            return s
        elif format =='recon':
                # check if pt_idx has three digits

                s = f'M_B3_{pt_idx//100:1d}_{pt_idx%100:02d}'
                print(f'get_patient_identifier: Generated patient identifier: {s}')
                return s
    else:
        raise ValueError("Study must be 1 for this function to work correctly.")


def get_patient_eval_path(eval_settings, pt_idx):
    """
    Generate the evaluation path for a patient based on the study, patient index, and scan index.
    Args:
        pt_idx (int): Patient index.
        idx_scan (int): Scan index.
    Returns:
        str: Path to the patient's evaluation folder.
    """
    path_to_evaluation = eval_settings.get('path_to_evaluation')
    study = eval_settings.get('study_to_eval')

    if study == 1:
        path = os.path.join(path_to_evaluation, f'M_B1_{pt_idx:02d}')
        if not os.path.exists(path):
            print(f'Creating path: {path}')
            os.makedirs(path)
        return path
    elif study == 3:
        s = f'M_B3_{pt_idx // 100:1d}_{pt_idx % 100:02d}'
        path = os.path.join(path_to_evaluation, s)
        if not os.path.exists(path):
            print(f'Creating path: {path}')
            os.makedirs(path)
        return path
    else:
        raise ValueError("Study must be 1 for this function to work correctly.")



def load_df_all_data(eval_settings):
    combined_df_path = get_path_df_all_data(eval_settings)
    # check if file exists
    if not os.path.exists(combined_df_path):
        raise FileNotFoundError(f"Combined data file not found at {combined_df_path}. Did you run main_eval_study?!")
    with open(combined_df_path, 'rb') as f:
        combined_df = pickle.load(f)
    return combined_df


def convert_mask_index_to_organ(idx):
    """
    Convert a mask index to an organ name.
    Args:
        idx (int): Index of the organ in the mask.
    Returns:
        str: Name of the organ.
    """
    organ_names = {
        1: 'Wall',
        2: 'Thrombus',
        3: 'Lumen',
        5: 'Posas',
        11: 'Endoleak Type 1, Nr. 1',
        12: 'Endoleak Type 1, Nr. 2',
        21: 'Endoleak Type 2, Nr. 1',
        31: 'Endoleak, Type unknow, Nr 1.'
    }
    return organ_names.get(idx, f'unknown_{idx}')  # Default to unknown if idx is not in organ_names


def get_evaluation_settings(file_name):
    """
    Load evaluation settings from a YAML file.
    Returns:
        dict: Configuration settings loaded from the YAML file.
    """
    with open(file_name, 'r') as file:
        config = yaml.safe_load(file)

    if config.get("debug"):
        # Apply overrides
        config.update(config.get("debug_overrides", {}))

    # Resolve to absolute paths: some path-joining logic elsewhere relies on
    # os.path.join's behavior of discarding earlier components once an
    # absolute path is joined in, so relative paths here must be resolved
    # up front rather than left relative.
    for key in ('path_to_recon', 'path_to_evaluation', 'path_to_postprocessing'):
        if config.get(key):
            config[key] = os.path.abspath(config[key])

    return config


def get_path_to_results(eval_settings):
    path = os.path.join(eval_settings.get('path_to_evaluation'), eval_settings.get('foldername_results'))
    os.makedirs(path, exist_ok=True)
    return path



def get_path_to_fig_paper(eval_settings):
    path = os.path.join(eval_settings.get('path_to_evaluation'), eval_settings.get('foldername_results'), 'figures_for_paper')
    os.makedirs(path, exist_ok=True)
    return path

def get_assoc_save_path(eval_settings):
    path = os.path.join(get_path_to_results(eval_settings), 'associations')
    os.makedirs(path, exist_ok=True)
    return path


def get_test_retest_save_path(eval_settings):
    path = os.path.join(get_path_to_results(eval_settings), 'test_retest')
    os.makedirs(path, exist_ok=True)
    return path


def get_path_dict_all_scans(eval_settings):
    dict_all_scans_path = os.path.join(eval_settings.get('path_to_evaluation'),
                                       eval_settings.get('foldername_results'))
    # check if the directory exists, if not create it
    if not os.path.exists(dict_all_scans_path):
        os.makedirs(dict_all_scans_path)
    dict_all_scans_path = os.path.join(dict_all_scans_path,  'dict_all_dce_scans.pkl')
    return dict_all_scans_path


def get_path_df_all_data(eval_settings):
    path = os.path.join(eval_settings.get('path_to_evaluation'), eval_settings.get('foldername_results'),
                        f"{eval_settings.get('name_df_all_data')}.pkl")
    return path


def combine_all_scans_dicts(eval_settings):
    marvy_dia_path ='data_processed/marvy_mri_diameters.pkl'
    marvy_dia = pickle.load(open(marvy_dia_path, 'rb'))
    prestudy_growth_path = 'data_processed/prestudy_growth_rates.pkl'
    prestudy_growth = pickle.load(open(prestudy_growth_path, 'rb'))
    dict_all_scans_path = get_path_dict_all_scans(eval_settings)
    with open(dict_all_scans_path, 'rb') as f:
        dict_all_scans = pickle.load(f)

    patients = marvy_dia["patient_id"].unique()
    combined_records = []

    for pid in patients:
        idx_study, pt_idx = get_study_and_idxPt_from_patient_identifier(pid)
        # prestudy info
        a = prestudy_growth[prestudy_growth["patient_id"] == pid]
        if a.empty:
            print(f'No prestudy growth data for patient {pid}, skipping.') #in csv with prestudy data these patients are missing
            continue
        prestudy_row = prestudy_growth[prestudy_growth["patient_id"] == pid].iloc[0]

        prestudy_dict = {
            "modality": prestudy_row["modality"],
            "avg_growth": prestudy_row["avg_growth_mm_per_year"],
            "intervals": prestudy_row["intervals"],
            "first_scan": prestudy_row["first_scan"],
            "last_scan": prestudy_row["last_scan"],
            "dia_last_meas": prestudy_row["dia_last_meas"]
        }

        scans_dict = {}

        # Explicitly handle scan1 and scan2
        for scan_idx in [1, 2]:
            pt_id_fitting = f'P{idx_study}{pt_idx:02d}{scan_idx}'
            scan_key = f"scan{scan_idx}"

            # get the measurement and scan_date from marvy_dia if it exists
            scan_row = marvy_dia[(marvy_dia["patient_id"] == pid) & (marvy_dia["scan_idx"] == scan_idx)]
            if not scan_row.empty:
                measurement = scan_row["measurement"].values[0]
                scan_date = scan_row["scan_date"].values[0]
            else:
                measurement = None
                scan_date = None

            # build the fitting dict key

            fit_dict = dict_all_scans.get(pt_id_fitting, {})

            scans_dict[scan_key] = {
                "dia": measurement,
                "scan_date": scan_date,
                "paths": fit_dict.get("paths"),
                "dce_param": fit_dict.get("dce_param"),
                "idx_scan": fit_dict.get("idx_scan"),
                "recon_name": fit_dict.get("recon_name"),
                'recon_folder': fit_dict.get('recon_folder'),
                'snr_mean': fit_dict.get('snr_mean'),
                'cnr_mean': fit_dict.get('cnr_mean'),
            }
        # check if scans_dict[2] exists, if not skip this patient
        if scans_dict["scan2"]['dia'] is not None:
            cds={} #changes during study
            cds["dia"] = scans_dict["scan2"]["dia"] - scans_dict["scan1"]["dia"]
            cds["time_delta_year"] = ((scans_dict["scan2"]["scan_date"] - scans_dict["scan1"]["scan_date"])/np.timedelta64(1, 'D'))/365.25
            cds["annualized_growth_mm_per_year"] = cds["dia"] / cds["time_delta_year"] if cds["time_delta_year"] !=0 else None
            # loop over each entry in scans_dict["scan1"]["fit_results"] and scans_dict["scan2"]["fit_results"]
            # check if scan2 fit_results exist
            if scans_dict["scan2"]["dce_param"] is None:
                print(f'Warning: scan2 dce_param is None for patient {pid}, cannot calculate changes during study.')
            else:
                print(f'Processing changes during study for patient {pid}.')
                cds["dce_param"] = {}
                for fit_method in scans_dict["scan1"]["dce_param"].keys():
                    val1 = scans_dict["scan1"]["dce_param"][fit_method]
                    val2 = scans_dict["scan2"]["dce_param"][fit_method]
                    if val1 is not None and val2 is not None:
                        cds["dce_param"][fit_method] = val2 - val1
                    else:
                        cds["dce_param"][fit_method] = None
        else:
            cds = {}
            print(f'Warning: scan2 diameter is None for patient {pid}, skipping changes during study calculation.')

        combined_records.append({
            "patient_id": pid,
            "prestudy": prestudy_dict,
            "scan1": scans_dict["scan1"],
            "scan2": scans_dict["scan2"],
            "changes_during_study": cds
        })
    combined_df = pd.DataFrame(combined_records).set_index("patient_id")
    combined_df_path = get_path_df_all_data(eval_settings)

    with open(combined_df_path, 'wb') as f:
        pickle.dump(combined_df, f)
        print(f'Saved combined scans data to {combined_df_path}')

