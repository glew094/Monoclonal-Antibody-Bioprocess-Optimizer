import os
import numpy as np
from src.functions import load_mab_datasets, tune_nvar_model, optimize_bioprocess_economics, generate_analysis_figures


def main():
    # 1. Data Ingestion
    df_ic, df_ts = load_mab_datasets(data_dir="datasets")

    features = [
        "Glucose [mM]", "Glutamine [mM]", "Lactate [mM]",
        "Ammonia [mM]", "VCD [cells/mL]", "TCD [cells/mL]", "mAb [mg/L]"
    ]
    batch_ids = np.unique(df_ts["Batch ID"])

    # 2. Hyperparameter Tuning (LOBOCV)
    best_p, best_k = tune_nvar_model(df_ts, features, batch_ids)

    # 3. Economic Monte Carlo Simulations (p=2, k=5)
    #print("Running NVAR Economic Monte Carlo Simulations (p=2, k=5)...")
    #df_results = optimize_bioprocess_economics(df_ic, df_ts, features, batch_ids, p=2, k=5, n_sims=1000)
    #df_results = df_results.sort_values(by="Max STP [$/(L*d)]", ascending=False)
    #df_results.to_csv(os.path.join("tables", "optimized_economic_results.csv"), index=False)
    #best_batch = df_results.iloc[0]
    #print(f"\n--- GLOBAL OPTIMUM FOUND (Batch {best_batch['Batch ID']}) ---")


    # 4. Generate Portfolio Figures for the Global Optimum (Batch 127)
    generate_analysis_figures(df_ic, df_ts, features, batch_ids, target_batch=127, p=2, k=5)

if __name__ == "__main__":
    main()
