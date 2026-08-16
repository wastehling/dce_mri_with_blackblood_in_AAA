from helpers_analysis import  statistics_test_retest, statistics_association
from utilities import get_evaluation_settings
from helpers_data_ordering import extract_param_to_eval, get_expanded_flat_df


def main_statistics():
    eval_settings = get_evaluation_settings('evaluation.yml')
    data_all_scans_flat = get_expanded_flat_df(eval_settings)
    par_info_table = extract_param_to_eval(data_all_scans_flat)

    statistics_association(eval_settings, data_all_scans_flat, par_info_table)
    statistics_test_retest(eval_settings, data_all_scans_flat, par_info_table)


if __name__ == "__main__":
    main_statistics()
