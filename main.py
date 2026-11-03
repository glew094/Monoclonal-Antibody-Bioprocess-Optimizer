import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from src.functions import load_mab_datasets, generate_x_lag_cur_var
from src.classes import StandardScalerVAR


def main():
    # 1. Data Ingestion
    df_ic, df_ts = load_mab_datasets(data_dir="datasets")

    features = [
        "Glucose [mM]", "Glutamine [mM]", "Lactate [mM]",
        "Ammonia [mM]", "VCD [cells/mL]", "TCD [cells/mL]", "mAb [mg/L]"
    ]
    batch_ids = np.unique(df_ts["Batch ID"])

    # 2. Lag Generation (Testing p=3)
    p = 3
    x_lag, x_cur, t_cur, b_cur = generate_x_lag_cur_var(
        p=p, batch_ids=batch_ids, df=df_ts, features=features
    )

    print(f"Lagged Matrix Shape (x_lag): {x_lag.shape}")
    print(f"Target Matrix Shape (x_cur): {x_cur.shape}")

    # 3. Model Pipeline Initialization
    model = make_pipeline(
        StandardScalerVAR(p=p),
        KNeighborsRegressor(n_neighbors=5, weights="uniform")
    )

    # Test the fit on all data
    model.fit(x_lag, x_cur)
    print("k-NN NVAR Pipeline successfully fitted.")


if __name__ == "__main__":
    main()