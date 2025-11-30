import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.load_config import get_config


def plot_aligned_comparison(clean_anom, backdoor_anom, index, sensor_columns, dataset_name):
    plt.figure(figsize=(20, 8))
    # Get sample data
    clean_sample = clean_anom.loc[index, sensor_columns]
    backdoor_sample = backdoor_anom.loc[index, sensor_columns]

    # Filter out sensor columns with values > 10000
    filtered_cols = [col for col, val in backdoor_sample.items() if val <= 10000]

    clean_sample = clean_sample[filtered_cols]
    backdoor_sample = backdoor_sample[filtered_cols]

    # Generate sensor indices
    x = np.arange(len(filtered_cols))

    # Plot comparison curves
    plt.plot(x, clean_sample, 'g-', linewidth=1.5, marker='o', markersize=4, label='Original Data')
    plt.plot(x, backdoor_sample, color='r',linestyle='--', linewidth=1.5, marker='s', markersize=4, label='Backdoored Data')

    # Visualization settings
    plt.yticks(fontsize=16)
    # Remove x-axis tick labels
    plt.xticks([])
    # plt.xticks(x, filtered_cols, rotation=45, ha='right', fontsize=16)
    plt.xlabel('Sensor Channels', fontsize=22)
    plt.ylabel('Sensor Readings', fontsize=22)

    plt.grid(True, linestyle='--', alpha=0.6, axis='y')

    # Increase legend font size
    plt.legend(fontsize=22, loc='upper right')
    # Adjust the left margin
    plt.subplots_adjust(left=0.08)

    # Save as PDF vector image
    output_path = f'../images/{dataset_name}.pdf'
    plt.savefig(
        output_path,
        format='pdf',
        dpi=1200,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
    )
    plt.tight_layout()
    plt.show()


def get_aligned_anomalies(clean_df, backdoor_df, target_col):
    clean_anom_idx = clean_df[clean_df[target_col] == 1].index
    backdoor_anom_idx = backdoor_df[backdoor_df[target_col] == 1].index
    common_idx = clean_anom_idx.intersection(backdoor_anom_idx)
    return clean_df.loc[common_idx], backdoor_df.loc[common_idx]


def visualize_data(dataset_name, num_samples_to_plot: int = 1):
    config = get_config(dataset_name)
    # --- Data Loading ---
    try:
        clean_data = pd.read_csv(config.test_data_path)
        backdoor_data = pd.read_csv(config.backdoor_data_path)
    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during CSV loading: {e}")
        return

    # --- Preprocessing ---
    cols = [col for col in clean_data.columns if col not in config.remove_list]
    sensor_cols = cols[:config.sensor_dim]

    clean_anom, backdoor_anom = get_aligned_anomalies(clean_data, backdoor_data, config.target_col)

    # Get common indices
    common_idx = clean_anom.index

    if len(common_idx) == 0:
        print("No common anomalies found to plot.")
        return

    # Randomly select indices for visualization
    if num_samples_to_plot > len(common_idx):
        num_samples_to_plot = len(common_idx)

    # Randomly select num_samples_to_plot indices from common_idx
    selected_indices = random.sample(list(common_idx), num_samples_to_plot)

    print(f"Plotting {len(selected_indices)} random samples from common anomalies.")

    for index in selected_indices:
        plot_aligned_comparison(clean_anom, backdoor_anom, index, sensor_cols, dataset_name)


# --- Example Usage ---
if __name__ == "__main__":
    visualize_data("HAI")