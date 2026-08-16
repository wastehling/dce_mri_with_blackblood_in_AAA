import os
from utilities import get_patient_identifier, get_patient_eval_path, combine_all_scans_dicts, get_path_dict_all_scans
from data_loading import create_subject
import pickle
from plotting import *
from calculations import evaluate_subject
from utilities import get_evaluation_settings


def create_scan_rescan_dict(eval_settings):
    recon_name_must_contain = eval_settings.get('recon_name_must_contain')
    dict_all_scans = {}
    for pt_idx in eval_settings.get('patients_to_process'):
        print('Processing patient', pt_idx)
        dict_scans_pt = {}
        for idx_scan in [1, 2]:
            if idx_scan==2 and eval_settings.get('study_to_eval') != 1:
                continue
            try:
                pt_id = get_patient_identifier(eval_settings.get('study_to_eval'), pt_idx, 'recon', idx_scan=idx_scan)
                pt_name = get_patient_identifier(eval_settings.get('study_to_eval'), pt_idx, 'postprocessing', idx_scan=idx_scan)
                found_folders = os.listdir(
                    os.path.join(eval_settings.get('path_to_recon'), pt_id, 'DCE'))
                recon_folder_names = [
                    f for f in found_folders if
                    any(substring in f for substring in recon_name_must_contain)
                 ]
                print('Found recon folders:', recon_folder_names)
                if len(recon_folder_names) != 1:
                    print(f'Expected one folder with {recon_name_must_contain} in {pt_id}, found {len(recon_folder_names)} folders.')
                    continue
                selected_recon_folder = recon_folder_names[0]
                # for selected_recon_folder in recon_folder_names:
                try:
                    subject = create_subject(eval_settings, pt_idx,
                                             idx_scan, selected_recon_folder)
                    subject = evaluate_subject(subject, eval_settings)
                    subject['recon_name'] = f'{selected_recon_folder}'
                    # save_name = f'{pt_name}_{selected_recon_folder}'
                    save_name = f'{pt_name}'
                    dict_scans_pt[save_name] = subject
                    dict_all_scans[save_name] = subject
                except Exception as e:
                    print(f'Error processing folder {selected_recon_folder}: {e}')
            except Exception as e:
                print(f'Error processing patient {pt_idx} scan {idx_scan}: {e}')

        #save dict_scans_pt as a pickle file
        dict_scans_pt_path = os.path.join(get_patient_eval_path(eval_settings, pt_idx), f'dict_scans.pkl')
        with open(dict_scans_pt_path, 'wb') as f:
            pickle.dump(dict_scans_pt, f)
            print(f'Saved dict_scans_pt for patient {pt_idx} to {dict_scans_pt_path}')

        print('successfully created dict_scans_pt for patient', pt_idx)
    dict_all_scans_path = get_path_dict_all_scans(eval_settings)
    with open(dict_all_scans_path, 'wb') as f:
        pickle.dump(dict_all_scans, f)
        print(f'Saved dict_all_scans to {dict_all_scans_path}')
    print('All patients processed successfully.')


def create_scan_rescan_plots_from_dict(eval_settings):
    for pt_idx in eval_settings.get('patients_to_process'):
        dict_scans_pt_path = os.path.join(get_patient_eval_path(eval_settings, pt_idx), f'dict_scans.pkl')
        with open(dict_scans_pt_path, 'rb') as f:
            dict_scans_pt = pickle.load(f)
        for idx_scan in [1, 2]:
            if idx_scan == 2 and eval_settings.get('study_to_eval') != 1:
                continue
            pt_id = get_patient_identifier(eval_settings.get('study_to_eval'), pt_idx, 'recon', idx_scan=idx_scan)
            subject = dict_scans_pt.get(get_patient_identifier(eval_settings.get('study_to_eval'), pt_idx, 'postprocessing', idx_scan=idx_scan))
            if not subject:
                print(f'No data found for patient {pt_id} scan {idx_scan}.')
                continue

        #here the functions executed per scan-rescan
        if eval_settings.get('plot_mean_wall_intensity_over_time', True):
            print(f'Plotting mean wall intensity over time for patient {pt_idx}...')
            plot_mean_wall_intensity_over_time(dict_scans_pt.values(), eval_settings,
                                           image_key='image', mask_key='mask', show=True, normalization=True)


def get_dict_all_scans(eval_settings):
    dict_all_scans_path = get_path_dict_all_scans(eval_settings)
    with open(dict_all_scans_path, 'rb') as f:
        dict_all_scans = pickle.load(f)
    patients_to_process = eval_settings.get('patients_to_process')
    dict_filtered_scans = {k: v for k, v in dict_all_scans.items()
                           if v.get('pt_idx') in patients_to_process}
    return dict_filtered_scans


if __name__ == "__main__":
    eval_settings = get_evaluation_settings('evaluation.yml')
    if eval_settings.get('load_and_process_recons', True):
        create_scan_rescan_dict(eval_settings)
    if eval_settings.get('create_plots_patientwise', True):
        create_scan_rescan_plots_from_dict(eval_settings)
    combine_all_scans_dicts(eval_settings)

