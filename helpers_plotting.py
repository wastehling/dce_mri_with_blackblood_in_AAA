import pandas as pd
import seaborn as sns
from utilities import get_assoc_save_path, get_test_retest_save_path, get_path_to_results
import matplotlib.pyplot as plt
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


def plot_bland_altman_test_retest(eval_settings, data, param_to_eval, df_results_stats, patient_col='patient_id'):
    """
    Create Bland–Altman plots for wide-format test–retest data
    where columns are scan1.param and scan2.param
    """

    for sel_base_param in param_to_eval['base_param'].to_list():
        sel_base_param_short = param_to_eval[param_to_eval['base_param'] == sel_base_param]['short_name'].values[0]
        col1 = f'scan1.{sel_base_param}'
        col2 = f'scan2.{sel_base_param}'

        if col1 not in data.columns or col2 not in data.columns:
            print(f"Skipping {sel_base_param}: missing scan columns")
            continue

        x1 = data[col1].dropna()
        x2 = data[col2].dropna()

        # keep only patients with both scans
        common_idx = x1.index.intersection(x2.index)
        x1 = x1.loc[common_idx]
        x2 = x2.loc[common_idx]

        mean = (x1 + x2) / 2
        diff = x2 - x1

        bias = diff.mean()
        loa = 1.96 * diff.std()

        #get icc form df_results_stats
        # icc = df_results_stats.loc[sel_base_param, 'icc'] if sel_base_param in df_results_stats.index else np.nan
        wilcoxon_p = df_results_stats.loc[sel_base_param_short, 'wilcoxon_p']
        wCV = df_results_stats.loc[sel_base_param_short, 'wCV']

        plt.figure( figsize=(FIGWIDTH_SINGLE, FIGHEIGHT_SINGLE))
        plt.scatter(mean, diff, s=15, marker='x', c='black')
        plt.axhline(bias, linestyle='--', color='gray')
        plt.axhline(bias + loa, linestyle=':', color='red')
        plt.axhline(bias - loa, linestyle=':', color='red')

        #remove everything before last .
        par_name_to_show = param_to_eval[param_to_eval['base_param'] == sel_base_param]['name_to_show'].values[0]
        #if $ in par_name_to_show, remove so that i tcan be used as safename
        name_to_save = par_name_to_show.replace('$', '').replace(' ', '_').replace('[', '').replace(']', '').replace('/', '-').replace('{', '').replace('}', '')
        unit = param_to_eval[param_to_eval['base_param'] == sel_base_param]['unit'].values[0]
        if unit is not None:
            unit = f'[{unit}]'
        else:
            unit = ''

        #string wCV in percent rounded
        if wilcoxon_p < 0.05:
            str_p = 'p<0.05'
        else:
            str_p = f'p={wilcoxon_p:.2f}'
        str_wCV = f'{wCV:.0f}%'
        subtitle = f'wCV= {str_wCV}; {str_p}' #LOA: [{bias - loa:.2f}, {bias + loa:.2f}]'
        plt.xlabel(f'Mean {par_name_to_show} {unit}')
        #ylabel with delta symbol and par name
        plt.ylabel(f'$\Delta$ {par_name_to_show} {unit}')

        #set ylimet to be 1.5 times the max of diff and 0, and 1.5 times the min of diff and 0
        diff_max = diff.max()
        diff_min = diff.min()
        abs_diff_max = max(abs(diff_max), abs(diff_min))
        delta_mean = mean.max() - mean.min()
        plt.ylim(-2*abs_diff_max, 2*abs_diff_max)

        plt.title(subtitle, fontsize=10)
        plt.tight_layout()
        plt.grid(True)

        dict_annot = {'Diameter': 'a)', 'Wash-in': 'b)', 'AUC': 'c)', 'ME': 'd)'}
        if par_name_to_show in dict_annot:
            plt.annotate(dict_annot[par_name_to_show], xy=(10,150), xycoords='figure points', fontsize=12, fontweight='bold')

        path_to_save = get_test_retest_save_path(eval_settings)
        plt.savefig(f'{path_to_save}/bland_altman_{name_to_save}.tiff', dpi=300)
        if eval_settings.get('show_plots', False):
            plt.show()
        else:
            plt.close()




def plot_associations(eval_settings, data, assoc_df, par_info_table, alpha=0.05):
    """
    Generate scatter plots for all significant associations.
    """
    # sig_df = assoc_df[assoc_df['p_value'] < alpha]
    for _, row in assoc_df.iterrows():
        x = row['x']
        y = row['y']
        rho = row['rho']
        p = row['p_value']

        #check if any nan in rho or p, if so, skip
        if pd.isna(rho) or pd.isna(p):
            continue

        x_param = x.split('.')[-1]
        y_param = y.split('.')[-1]
        x_param_to_display = par_info_table[par_info_table['short_name'] == x_param]['name_to_show'].values[0]
        y_param_to_display = par_info_table[par_info_table['short_name'] == y_param]['name_to_show'].values[0]
        x_unit = par_info_table[par_info_table['short_name'] == x_param]['unit'].values[0]
        y_unit = par_info_table[par_info_table['short_name'] == y_param]['unit'].values[0]
        if x_unit is not None:
            x_param_to_display += f' [{x_unit}]'
        if y_unit is not None:
            y_param_to_display += f' [{y_unit}]'

        plt.figure(figsize=(FIGWIDTH_SINGLE, FIGHEIGHT_SINGLE))
        plt.scatter(data[x], data[y], marker='x', color='black')
        plt.xlabel(x_param_to_display)
        plt.ylabel(y_param_to_display)

        #set ylimet to be 1.5 times the max of y and 0, and 1.5 times the min of y and 0
        y_max = data[y].max()
        y_min = data[y].min()
        delta_y = y_max - y_min
        plt.ylim(y_min - delta_y *1, y_max + delta_y *1)

        if any(keyword in y_param_to_display for keyword in ['TTP', 'AUC', "ME"]):
            plt.ylim(0, 2*y_max)

        title = f'ρ = {rho:.2f}, p = {p:.2f}'
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.grid(True)

        dict_annot = {'Diameter': 'a)', 'Wash-in': 'b)', 'AUC': 'c)', 'ME': 'd)'}
        if y == 'scan1.dce_param.TTP' and x == 'prestudy.avg_growth':
            plt.annotate('a)', xy=(4, 150), xycoords='figure points', fontsize=12,
                         fontweight='bold')
        elif y == 'scan2.dce_param.AUC' and x == 'prestudy.dia_last_meas':
            plt.annotate('b)', xy=(4, 150), xycoords='figure points', fontsize=12,
                         fontweight='bold')

        path_ass_res = get_assoc_save_path(eval_settings)
        plt.savefig(f'{path_ass_res}/association_{x}_vs_{y}_{row["y_scan_idx"]}.tiff', dpi=300)
        plt.savefig(f'{path_ass_res}/association_{x}_vs_{y}_{row["y_scan_idx"]}.pdf', dpi=300)
        if eval_settings.get('show_plots', False):
            plt.show()
        else:
            plt.close()


def plot_correlation_table(eval_settings, df, p_thresh=0.05):
    """
    Plots a correlation matrix-style table with rho as color and significant p-values displayed.

    Parameters:
        df : pd.DataFrame
            Dataframe containing x, y, rho, p columns
        rho_col : str
            Column name for correlation coefficient
        p_col : str
            Column name for p-value
        p_thresh : float
            Significance threshold to display p-values
    """
    #df sort values
    df = df.sort_values(by=['y_scan_idx', 'x_scan_idx'])

    # Create unique labels for x and y axes, so that no confusion arises for same param from different scans
    df['x_internal'] = df['x_scan_idx'] + ": " + df['x_short_name']
    df['y_internal'] = df['y_scan_idx'] + ": " + df['y_short_name']
    # Create a mapping from internal labels to display names
    x_label_map = dict(zip(df['x_internal'], df['x_name_to_show']))
    y_label_map = dict(zip(df['y_internal'], df['y_name_to_show']))

    # Define the desired order for y_internal
    y_internal_values = df['y_internal'].unique()
    desired_order = sorted(
        y_internal_values,
        key=lambda x: (
            0 if x.startswith('scan1:') else
            1 if x.startswith('scan2:') else
            2
        )
    )

    # Pivot and reindex
    rho_pivot = df.pivot(index='y_internal', columns='x_internal', values='rho').reindex(desired_order).T
    p_pivot = df.pivot(index='y_internal', columns='x_internal', values='p_value').reindex(desired_order).T

    # Mask for significant p-values
    significant = p_pivot < p_thresh

    print(f'Min max of rho: {rho_pivot.min().min():.2f} to {rho_pivot.max().max():.2f}')
    print(f"min max of p: {p_pivot.min().min():.3g} to {p_pivot.max().max():.3g}")

    plt.figure(figsize=(FIGWIDTH_DOUBLE, FIGHEIGHT_SINGLE))
    ax = sns.heatmap(rho_pivot, annot=False, fmt=".2f", cmap="coolwarm", center=0, square=True)

    #Annotate colorbar with Speerma's rho
    cbar = ax.collections[0].colorbar
    cbar.set_label("Spearman's ρ", rotation=90, labelpad=15)

    # Annotate significant p-values
    for i in range(rho_pivot.shape[0]):
        for j in range(rho_pivot.shape[1]):
            if significant.iloc[i, j]:
                ax.text(j + 0.5, i + 0.5, f"{p_pivot.iloc[i, j]:.2f}", ha="center", va="center", color="black",
                        fontsize=8, fontweight="bold")

    # Make x and y ticks more readable by replacing with name_to_show
    x_ticks = [y_label_map[tick.get_text()] for tick in ax.get_xticklabels()]
    ax.set_xticklabels(x_ticks, rotation=45, ha="right")

    # Replace y-axis tick labels with y_name_to_show
    y_ticks = [x_label_map[tick.get_text()] for tick in ax.get_yticklabels()]
    ax.set_yticklabels(y_ticks, rotation=0)

    # Draw rectancles around group
    from matplotlib.patches import Rectangle
    size_group = rho_pivot.shape[1] // 2
    scan_groups = {
        "scan1": (0, size_group),
        "scan2": (size_group, 2 * size_group),
    }

    n_cols = rho_pivot.shape[0]  # after .T, shape[0] is the column count of original

    for group, (row_start, row_end) in scan_groups.items():
        row_span = row_end - row_start
        ax.add_patch(
            Rectangle(
                (row_start, 0),  # x=row_start (now on x-axis after .T), y=0
                row_span,  # width: spans the group
                n_cols,  # height: spans all columns
                fill=False,
                edgecolor="black",
                lw=2,
                linestyle="--",
                clip_on=False
            )
        )
        # Add label above the rectangle
        label = {"scan1": "Baseline", "scan2": "Follow-up"}.get(group, group)
        ax.text(
            row_start + row_span / 2,  # x: center of the group
            -0.5,  # y: just above the heatmap (negative = above row 0)
            label,
            ha="center",
            va="bottom",
            # fontsize=8,
            # fontweight="bold",
            clip_on=False
        )

    #do x and y labels rotation
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor'
    )
    ax.tick_params(axis='x', pad=6)
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor'
    )
    ax.tick_params(axis='y', pad=6)

    ax.set_xlabel('')
    ax.set_ylabel('')

    path_to_save = get_assoc_save_path(eval_settings)
    plt.savefig(f'{path_to_save}/correlation_table.pdf', dpi=300)
    plt.savefig(f'{path_to_save}/correlation_table.tiff', dpi=300)
    plt.show()
