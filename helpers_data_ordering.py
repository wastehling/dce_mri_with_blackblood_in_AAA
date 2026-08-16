import pandas as pd
from utilities import load_df_all_data


def extract_param_to_eval(df):
    cols = df.columns
    #drop everything that contains posas or thrombus
    cols_wall_only = [c for c in cols if 'posas' not in c.lower() and 'thrombus' not in c.lower()]
    #drop everythin that contains modality, intervals, first_scan, last_scan, path
    str_to_drop = ['modality', 'intervals', 'first_scan', 'last_scan', 'path', 'date', 'recon', 'idx']
    cols_select = [c for c in cols_wall_only if all(x not in c.lower() for x in str_to_drop)]

    df_param_to_eval = pd.DataFrame({'full_name': cols_select})
    df_param_to_eval['short_name'] = df_param_to_eval['full_name'].apply(lambda x: x.split('.')[-1])
    df_param_to_eval['unit'] = ''  # Placeholder for unit, can be filled as needed
    units_dict = {
    'avg_growth': 'mm/year',
    'dia_last_meas': 'mm',
    'dia': 'mm',
    'TTP': 'sec',
    'ME': None,
    'AUC': 'sec',
    'AUC_twothird': 'sec',
    'wash_in': '1/sec',
    'wash_out': '1/sec',
    'linear_fit_slope': '1/sec',
    'linear_fit_intercept': None,
    }
    name_to_shown_dict = {
        'avg_growth': 'Avg. Growth',
        'dia_last_meas': 'Diameter',
        'dia': 'Diameter',
        'TTP': 'TTP',
        'ME': 'ME',
        'AUC': 'AUC',
        'AUC_twothird': '$AUC_{2/3}$',
        'wash_in': 'Wash-in',
        'wash_out': 'Wash-out',
        'linear_fit_slope': 'Slope Signal-Enhancement',
        'linear_fit_intercept': 'Intercept$_{t=0}$ Signal-Enhancement',
    }
    df_param_to_eval['name_to_show'] = df_param_to_eval.apply(
        lambda row: name_to_shown_dict.get(row['short_name'], row['short_name']), axis=1)
    df_param_to_eval['unit'] = df_param_to_eval.apply(
        lambda row: units_dict.get(row['short_name'], row['unit']),
        axis=1
    )
    for idx, row in df_param_to_eval.iterrows():
        if '_diff' in row['name_to_show']:
            selected_parameter = row['name_to_show'].replace('_diff', '')
            unit_selected_parameter = units_dict.get(selected_parameter, '')
            df_param_to_eval.at[idx, 'name_to_show'] = selected_parameter
            df_param_to_eval.at[idx, 'unit'] = unit_selected_parameter
            df_param_to_eval.at[idx, 'name_to_show'] = '$\Delta$' + df_param_to_eval.at[idx, 'name_to_show']



    return df_param_to_eval


def get_prestudy_param_to_eval(df):
    prestudy_cols = [c for c in df.columns if c.startswith('prestudy.')]
    #create a pd with only subpart of df which is prestudy cols
    df_prestudy = df[prestudy_cols]
    return df_prestudy


def get_dce_param_to_eval(df):
    #loop over full_name and only keep those rows that start with scan1.phaCokMod
    mask = df['full_name'].str.contains('dce_param')
    df_dce = df[mask]
    return df_dce


def get_dce_param_list(df):
    df_dce = get_dce_param_to_eval(df)
    dce_list = df_dce['full_name'].tolist()
    #drop everything before phaCokMod
    dce_list = ['phaCokMod.' + x.split('phaCokMod.', 1)[1] for x in dce_list]
    dce_list = list(set(dce_list))
    return dce_list


def get_base_param_list(df_param_to_eval):
    """
    Keeps only the parameter names with 'scan1.' prefix. and removes the scan1.
    Args:
        df_param_to_eval:
    Returns:
        pd.DataFrame with base parameters based on scan 1(without scan1. prefix)

    """
    #select data where full_name starts with scan1.
    scan1_params = df_param_to_eval.loc[df_param_to_eval['full_name'].str.startswith('scan1.')].copy()
    scan1_params['base_param'] = scan1_params['full_name'].apply(lambda x: x.split('scan1.', 1)[1])
    #remove the scan1. prefix
    return scan1_params


def get_expanded_flat_df(eval_settings):
    df_combined_data = load_df_all_data(eval_settings)

    cols_to_extract = ["prestudy", "scan1", "scan2", "changes_during_study"]
    expanded = df_combined_data.apply(
        lambda row: extract_all_info(row, cols_to_extract),
        axis=1
    )
    expanded_df = pd.DataFrame(expanded.tolist()).set_index('patient_id')
    # Remove columns which only contain nan but print warning for each deleted col
    col_to_drop = expanded_df.columns[expanded_df.isna().all()].tolist()
    for col in col_to_drop:
        print(f"Warning: Dropping column '{col}' because it contains only NaN values.")
    expanded_df = expanded_df.drop(columns=col_to_drop)
    
    return expanded_df


def extract_all_info(row, cols):
    flat = {}
    flat["patient_id"] = row.name
    for col in cols:
        value = row[col]
        if isinstance(value, dict):
            flat.update(flatten_dict(value, parent_key=col))
    return flat


def flatten_dict(d, parent_key="", sep="."):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items



def test_implementation():
    from utilities import get_evaluation_settings
    eval_settings = get_evaluation_settings('evaluation.yml')
    df_all_scans_flat = get_expanded_flat_df(eval_settings)
    df_param_to_eval = extract_param_to_eval(df_all_scans_flat)
    prestudy_df = get_prestudy_param_to_eval(df_all_scans_flat)
    print('success')


if __name__ == "__main__":
    test_implementation()
