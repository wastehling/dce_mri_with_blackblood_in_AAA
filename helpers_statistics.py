from scipy.stats import shapiro, pearsonr, spearmanr, ttest_rel
from scipy.stats import wilcoxon
# from pingouin import intraclass_corr
import pandas as pd
import numpy as np


def get_correlation_statistics(var1, var2, silent=False):
    """
    Association between two variables
    :param var1:
    :param var2:
    :return:
    """
    var1 = np.array(var1)
    var2 = np.array(var2)

    # Normality tests
    shapiro_var1 = shapiro(var1)
    shapiro_var2 = shapiro(var2)

    if not silent:
        print("Shapiro-Wilk Test (p < 0.05 indicates non-normality):")
        print(f"Variable 1: W = {shapiro_var1.statistic:.4f}, p = {shapiro_var1.pvalue:.4f}")
        print(f"Variable 2: W = {shapiro_var2.statistic:.4f}, p = {shapiro_var2.pvalue:.4f}")

    # Choose correlation method
    if shapiro_var1.pvalue < 0.05 or shapiro_var2.pvalue < 0.05:
        if not silent:
            print("Using Spearman correlation (non-parametric). So one of the parameters is not normally distributed.")
        corr_stat, p_value = spearmanr(var1, var2)
        method = "Spearman"
    else:
        if not silent:
            print("Using Pearson correlation (parametric, both parameters are normally distributed).")
        corr_stat, p_value = pearsonr(var1, var2)
        method = "Pearson"

    return {
        "correlation_method": method,
        "correlation_coefficient": corr_stat,
        "p_value": p_value,
        "shapiro_var1": shapiro_var1,
        "shapiro_var2": shapiro_var2
    }


def get_paired_sample_statistics(var_scan1, var_scan2):
    s1 = np.array(var_scan1)
    s2 = np.array(var_scan2)
    differences = s1 - s2

    # 1. Normality of DIFFERENCES (Correct for paired tests)
    shapiro_p = shapiro(differences).pvalue

    # 2. Within-Subject CV (Correct for repeatability)
    # Using the Root Mean Square approach is also common in MRI
    means = (s1 + s2) / 2
    cv = np.mean((np.abs(differences) / np.sqrt(2)) / means) * 100
    sem = np.std(differences) / np.sqrt(len(differences))  # Standard Error of the Mean
    # 3. Choose test
    if shapiro_p < 0.05:
        stat, p = wilcoxon(s1, s2)
        test = "Wilcoxon"
    else:
        stat, p = ttest_rel(s1, s2)
        test = "Paired t-test"

    return {'test': test, 'p_value': p, 'cv_ws': cv, 'stat': stat, 'sem': sem}


def calc_spearman_table(data, name_first_meas, name_second_meas):
    rows = []
    for x in name_first_meas:
        for y in name_second_meas:
            # print(f'Calculating Spearman correlation between {x} and {y}; '
            #       f'length of input data is {len(data[x].dropna())} and {len(data[y].dropna())}')
            rho, p = spearmanr(data[x], data[y], nan_policy='omit')
            rows.append({
                'x': x,
                'y': y,
                'rho': rho,
                'p_value': p
            })
    df = pd.DataFrame(rows)
    return df


def create_spearman_table(data, name_first_meas, name_second_meas, par_info_table):
    """
    Spearman correlation analysis between clinical and DCE parameters.
    """
    spearman_table = calc_spearman_table(data, name_first_meas, name_second_meas)
    spearman_table = spearman_table.merge(
        par_info_table,
        left_on='y',
        right_on='full_name',
        how='left'
    ).rename(columns={'short_name': 'y_short_name', 'unit': 'y_unit', 'name_to_show': 'y_name_to_show' }).drop(columns=['full_name'])
    spearman_table = spearman_table.merge(
        par_info_table,
        left_on='x',
        right_on='full_name',
        how='left'
    ).rename(columns={'short_name': 'x_short_name', 'unit': 'x_unit', 'name_to_show': 'x_name_to_show' }).drop(columns=['full_name'])
    spearman_table['y_scan_idx'] = spearman_table['y'].apply(lambda s: s.split('.')[0] if '.' in s else '')
    spearman_table['x_scan_idx'] = spearman_table['x'].apply(lambda s: s.split('.')[0] if '.' in s else '')
    return spearman_table



def test_retest_analysis(data, base_param, patient_col='patient_id'):
    """
    Compute Wilcoxon + ICC for wide-format data:
    columns are scan1.xxx, scan2.xxx
    """


    results = {}
    for param in base_param['base_param'].to_list():
        col1 = f'scan1.{param}'
        col2 = f'scan2.{param}'

        if col1 not in data.columns or col2 not in data.columns:
            print(f"Skipping {param}: missing scan columns")
            continue

        x1 = data[col1].dropna()
        x2 = data[col2].dropna()

        # keep only patients with both scans
        common_idx = x1.index.intersection(x2.index)
        x1 = x1.loc[common_idx]
        x2 = x2.loc[common_idx]

        # Wilcoxon
        try:
            stat, p = wilcoxon(x1, x2)
        except ValueError:
            stat, p = np.nan, np.nan

        # ICC
        icc_df = pd.DataFrame({
            'subject': np.repeat(common_idx, 2),
            'rater': np.tile(['scan1', 'scan2'], len(common_idx)),
            'score': np.concatenate([x1.values, x2.values])
        })

        # icc = intraclass_corr(icc_df, targets='subject', raters='rater', ratings='score')
        # icc_row = icc.loc[icc['Type'] == 'ICC3'].iloc[0]

        # --- within-subject coefficient of variation (root-mean-square method) ---
        diff_sq = (x1 - x2) ** 2 / 2  # within-subject variance
        sub_mean = (x1 + x2) / 2  # subject mean
        s2_over_m2 = diff_sq / sub_mean ** 2  # normalized
        wcv = np.sqrt(np.mean(s2_over_m2))

        # --- within-subject standard deviation (wSD) ---
        wsd = np.sqrt(np.mean(diff_sq))

        # --- between-subject SD and CV ---
        bsd = sub_mean.std()
        bcv = bsd / sub_mean.mean()
        # abs_pct_diff = (np.abs(x1-x2) / sub_mean*100)
        abs_pct_diff = (x1-x2) / sub_mean/100
        # abs_pct_diff = np.abs(x1-x2)
        # abs_pct_diff = x1-x2

        #get short_name of param
        param_shot_name = base_param.loc[base_param['base_param'] == param, 'short_name'].values[0]
        unit = base_param.loc[base_param['base_param'] == param, 'unit'].values[0]
        results[param_shot_name] = {
            'unit': unit,
            'param': param,
            'wilcoxon_p': round(p,2),
            'wCV': round(wcv*100),
            'wSD': round(wsd, 2),
            'bSD': round(bsd, 2),
            'bCV': round(bcv * 100),
            'wilcoxon_stat': stat,
            'abs_pct_diff': abs_pct_diff,
        }

    return results
