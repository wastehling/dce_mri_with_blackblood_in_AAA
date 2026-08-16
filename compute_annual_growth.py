import pandas as pd
from matplotlib import pyplot as plt
from helpers_plotting import FIGWIDTH_DOUBLE, FIGHEIGHT_DOUBLE
from utilities import get_evaluation_settings
from scipy import stats
import os

def create_annual_growth_plot():
    # eval_settings = get_evaluation_settings('evaluation_B1.yml')
    path_to_csv ="data_predefined/diameter_prestudy.csv"
    prestudy_dia, marvy_dia = load_prestudy_data(path_to_csv)
    prestudy_growth = compute_growth_rates(prestudy_dia)
    #make dir data_processesed if it doesn't exist
    if not os.path.exists("data_processed"):
        os.makedirs("data_processed")
    prestudy_growth.to_pickle("data_processed/prestudy_growth_rates.pkl")
    marvy_dia.to_pickle("data_processed/marvy_mri_diameters.pkl")
    plot_diameter_over_time_for_paper(prestudy_dia, prestudy_growth, marvy_dia)


def compute_growth_rates(long_df):
    results = []

    # Ensure correct dtype and sorting
    df = long_df.sort_values(["patient_id", "scan_date"])

    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("scan_date")
        modality = group["modality"].iloc[0]

        if len(group) < 2:
            results.append({
                "patient_id": pid,
                "modality": modality,
                "avg_growth_mm_per_year": float("nan"),
                "intervals": 0,
                "first_scan": group["scan_date"].iloc[0],
                "last_scan": group["scan_date"].iloc[-1]
            })
            continue

        # Convert dates to years relative to first scan
        t_years = (group["scan_date"] - group["scan_date"].iloc[0]).dt.days / 365.25
        d = group["measurement"]

        slope, intercept, r_value, p_value, std_err = stats.linregress(t_years, d)

        results.append({
            "patient_id": pid,
            "modality": modality,
            "avg_growth_mm_per_year": slope,
            "intervals": len(group) - 1,
            "first_scan": group["scan_date"].iloc[0],
            "last_scan": group["scan_date"].iloc[-1],
            "dia_last_meas": group["measurement"].iloc[-1]
        })

    growth_df = pd.DataFrame(results)
    return growth_df


def load_prestudy_data(file_path):
    df = pd.read_csv(file_path)

    # -----------------------------
    # 1) Identify numeric year columns
    # -----------------------------
    year_cols = [
        col for col in df.columns
        if isinstance(col, str) and col.isdigit()
    ]

    # Build list of (date_col, measurement_col) pairs for dynamic years
    year_pairs = []
    for year_col in year_cols:
        val_col = df.columns[df.columns.get_loc(year_col) + 1]
        year_pairs.append((year_col, val_col))

    # -----------------------------
    # 2) Build long_df = regular data only
    # -----------------------------
    long_records = []
    for _, row in df.iterrows():
        patient_id = row["MARVY_num"]
        modality = row["Modality"]

        for date_col, val_col in year_pairs:
            scan_date = row[date_col]
            measurement = row[val_col]

            if pd.notna(scan_date) and pd.notna(measurement):
                long_records.append({
                    "patient_id": patient_id,
                    "modality": modality,
                    "scan_date": pd.to_datetime(scan_date),
                    "measurement": measurement
                })

    long_df = pd.DataFrame(long_records).sort_values(["patient_id", "scan_date"])

    # -----------------------------
    # 3) Build mri_df = only MRI dates (first + second)
    # -----------------------------
    fixed_pairs = [
        ("MARVY_scan1", "diameter_1", 1),
        ("MARVY_scan2", "diameter_2", 2),
    ]

    mri_records = []
    for _, row in df.iterrows():
        patient_id = row["MARVY_num"]

        for date_col, val_col, scan_idx in fixed_pairs:
            if date_col in df.columns and val_col in df.columns:
                scan_date = row[date_col]
                measurement = row[val_col]

                if pd.notna(scan_date) and pd.notna(measurement):
                    mri_records.append({
                        "patient_id": patient_id,
                        "scan_date": pd.to_datetime(scan_date),
                        "scan_type": date_col,
                        "scan_idx": scan_idx,
                        "measurement": measurement
                    })

    mri_df = pd.DataFrame(mri_records).sort_values(["patient_id", "scan_idx"])

    # -----------------------------
    # RETURN TWO CLEAN DATAFRAMES
    # -----------------------------
    return long_df, mri_df


def plot_diameter_over_time_for_paper(long_df, growth_df, mri_df=None):
    modality_styles = {
        "mri": "dotted",
        "us": "dashdot",
        "cta": "dashed"
    }
    modality_colors = {
        "mri": "blue",
        "us": "red",
        "cta": "green"
    }
    # Assign a consistent color per patient
    patients = sorted(long_df["patient_id"].unique())
    #sort patients by modality


    plt.figure(figsize=(FIGWIDTH_DOUBLE, FIGHEIGHT_DOUBLE))

    # -------------------------------------------------
    # Plot NON-MRI measurements (main timeline)
    # -------------------------------------------------
    for pid, group in long_df.groupby("patient_id"):
        group = group.sort_values("scan_date")
        modality = group["modality"].iloc[0]

        # Lookup precomputed growth
        growth_row = growth_df[growth_df["patient_id"] == pid]
        if not growth_row.empty:
            avg_growth = growth_row["avg_growth_mm_per_year"].iloc[0]
            growth_str = f"{avg_growth:+.2f} mm/yr" if pd.notna(avg_growth) else "n/a"
        else:
            growth_str = "n/a"

        linestyle = modality_styles.get(str(modality).lower(), "-")

        plt.plot(
            group["scan_date"],
            group["measurement"],
            marker="o",
            linestyle=linestyle,
            color=modality_colors.get(str(modality).lower(), "black"),
        )


    #add legend for modalities only
    handles = []
    for modality, style in modality_styles.items():
        handle = plt.Line2D([], [], color=modality_colors.get(modality, "black"), linestyle=style, marker='o', label=modality.upper())
        handles.append(handle)
    plt.legend(handles=handles, title="Modality")

    plt.xlabel("Scan Date")
    plt.ylabel("Diameter [mm]")
    plt.title("AAA Diameter Over Time per Patient")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    eval_settings = get_evaluation_settings('evaluation.yml')
    create_annual_growth_plot()
