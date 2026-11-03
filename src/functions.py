# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

import os
import pandas as pd
import numpy as np

def load_mab_datasets(data_dir="datasets"):
    df_ic = pd.read_csv(os.path.join(data_dir, "initial_conditions.csv"))

    batch_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith("batch_") and filename.endswith(".csv"):
            batch_files.append(os.path.join(data_dir, filename))

    df_list = []
    for file in batch_files:
        batch_id = int(os.path.basename(file).split('_')[1].split('.')[0])
        df_temp = pd.read_csv(file)
        df_temp.insert(0, "Batch ID", batch_id)
        df_list.append(df_temp)

    df_ts = pd.concat(df_list, ignore_index=True)
    return df_ic, df_ts

def generate_x_lag_cur_var(p, batch_ids, df, features):
    """
    Generates the current and lagged feature matrices for a VAR/NVAR model.
    """
    x_lag_list = []
    x_cur_list = []
    t_cur_list = []
    b_cur_list = []

    for batch_id in batch_ids:
        # Isolate the specific batch
        mask = df["Batch ID"] == batch_id
        df_batch = df.loc[mask].reset_index(drop=True)

        t_batch = df_batch["Time [h]"].to_numpy()
        x_batch = df_batch[features].to_numpy()

        n_samples = x_batch.shape[0]

        # Start at index p to ensure enough previous time points exist
        for i in range(p, n_samples):
            x_cur_list.append(x_batch[i])
            t_cur_list.append(t_batch[i])
            b_cur_list.append(batch_id)

            # Extract x_(t-1), x_(t-2), ..., x_(t-p) and concatenate horizontally
            lag_vector = []
            for j in range(1, p + 1):
                lag_vector.append(x_batch[i - j])

            x_lag_list.append(np.concatenate(lag_vector))

    return np.array(x_lag_list), np.array(x_cur_list), np.array(t_cur_list), np.array(b_cur_list)