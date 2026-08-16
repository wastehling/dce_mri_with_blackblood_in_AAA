from helpers_data_ordering import get_base_param_list, get_dce_param_to_eval
from helpers_plotting import plot_associations
from helpers_statistics import test_retest_analysis, create_spearman_table
from utilities import *
import pandas as pd
from helpers_plotting import plot_correlation_table
from helpers_plotting import plot_bland_altman_test_retest


def snr_cnr_repeatability_correlation(data, repeatability_results):
    """
    Correlate per-subject SNR/CNR with absolute % difference between scans
    for each DCE parameter, using Spearman's rho.
    """
    from scipy.stats import spearmanr
    snr = data[list(('scan1.snr_mean', 'scan2.snr_mean'))].min(axis=1)
    cnr = abs(data[list(('scan1.cnr_mean', 'scan2.cnr_mean'))].min(axis=1))

    corr_results = {}
    for param, metrics in repeatability_results.items():
        abs_pct_diff = metrics.get('abs_pct_diff')
        if abs_pct_diff is None:
            continue

        # align on index
        common_idx = abs_pct_diff.index.intersection(snr.dropna().index)
        apd = abs_pct_diff.loc[common_idx]

        rho_snr, p_snr = spearmanr(snr.loc[common_idx], apd)
        rho_cnr, p_cnr = spearmanr(cnr.loc[common_idx], apd)
        #if p_snr or p_cnr is lower than 0.05 print warning

        # # scatter snr and cnr against apd
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 5))
        # plt.suptitle('Correlation of SNR/CNR with Absolute % Difference (APD) for ' + param)
        # plt.subplot(1, 2, 1)
        # plt.scatter(snr.loc[common_idx], apd, label='SNR', alpha=0.7)
        # plt.xlabel('SNR')
        #
        # plt.subplot(1, 2, 2)
        # plt.scatter(cnr.loc[common_idx], apd, label='CNR', alpha=0.7)
        # plt.xlabel('CNR')
        # plt.ylabel('Absolute % Difference (APD)')
        # plt.show()

        p_thres = 0.05
        if p_snr < p_thres or p_cnr < p_thres:
            print(f'SNR/CNR correlation results for {param} are significant!!!!:')
            print(f'rho_snr, rho_cnr, p_snr, p_cnr: {rho_snr, rho_cnr, p_snr, p_cnr}')
        corr_results[param] = {
            'snr_rho': round(rho_snr, 3),
            'snr_p': round(p_snr, 3),
            'cnr_rho': round(rho_cnr, 3),
            'cnr_p': round(p_cnr, 3),
        }
    print("\n".join("{}\t{}".format(k, v) for k, v in corr_results.items()))
    return corr_results


def statistics_test_retest(eval_settings, df_all_scans_flat, par_info_table):
    """
    Compute and visualize test–retest statistics for DCE-derived parameters.

    Parameters
    ----------
    df_all_scans_flat : pandas.DataFrame
        DataFrame containing all measurements.
        Must have columns: subject_id, scan_id, and DCE parameters.
    par_info_table :
    """
    df_base_param = get_base_param_list(par_info_table)
    results_stats = test_retest_analysis(df_all_scans_flat, df_base_param, patient_col='patient_id')
    snr_cnr_corr = snr_cnr_repeatability_correlation(df_all_scans_flat, results_stats)

    df_results_stats = pd.DataFrame.from_dict(results_stats, orient='index')
    df_results_stats.to_csv(f'{get_test_retest_save_path(eval_settings)}/test_retest_results.csv', index=True)

    plot_bland_altman_test_retest(eval_settings, df_all_scans_flat, df_base_param, df_results_stats, patient_col='patient_id')
    print('Final test-retest statistics saved.')


def statistics_association(eval_settings, df_all_scans_flat, par_info_table):
    ## Compute associations
    dce_derived_params = get_dce_param_to_eval(par_info_table)
    spearman_df = create_spearman_table(
        data=df_all_scans_flat,
        name_first_meas=['prestudy.avg_growth', 'prestudy.dia_last_meas',],
        # name_first_meas=['scan1.snr_mean', 'scan2.snr_mean', ],
        # name_first_meas=['scan1.snr_mean', 'scan1.cnr_mean', ],
        # name_first_meas=['scan2.snr_mean', 'scan2.cnr_mean', ],

        # name_first_meas=['prestudy.avg_growth', 'scan1.dia','changes_during_study.dia',],
        # name_first_meas=['prestudy.avg_growth', 'scan1.dia',],
        # 'scan1.dia', 'scan2.dia'],
        name_second_meas=dce_derived_params['full_name'].tolist(),
        par_info_table=par_info_table
    )
    spearman_df = spearman_df[~spearman_df['y'].str.contains('changes')]
    plot_correlation_table(eval_settings, spearman_df, p_thresh=0.05)

    spearman_df = spearman_df.sort_values(by='p_value')
    #round rho to 2 decimal places and p_value to 4 decimal places
    spearman_df['rho'] = spearman_df['rho'].round(2)
    spearman_df['p_value'] = spearman_df['p_value'].round(4)

    spearman_df.to_csv(f'{get_assoc_save_path(eval_settings)}/associations.csv', index=False)
    plot_associations(eval_settings, df_all_scans_flat, spearman_df, par_info_table)
    print('Association analysis completed and results saved.')

