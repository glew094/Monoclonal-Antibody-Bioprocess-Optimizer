import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from src.classes import StandardScalerVAR
import src.constants as const


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

def tune_nvar_model(df_ts, features, batch_ids):
    print("Starting LOBOCV Grid Search for k-NN NVAR...")

    p_values = [2, 3, 4]
    k_values = [2, 3, 4, 5]

    best_score = float('-inf')
    best_params = {}
    best_p = None

    for p in p_values:
        x_lag, x_cur, t_cur, b_cur = generate_x_lag_cur_var(
            p=p, batch_ids=batch_ids, df=df_ts, features=features
        )

        model = make_pipeline(
            StandardScalerVAR(p=p),
            KNeighborsRegressor(weights="uniform")
        )

        param_grid = {"kneighborsregressor__n_neighbors": k_values}
        gkf = GroupKFold(n_splits=3)

        model_gscv = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=gkf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        model_gscv.fit(x_lag, x_cur, groups=b_cur)
        print(
            f"p={p} | Best k: {model_gscv.best_params_['kneighborsregressor__n_neighbors']} | RMSE: {-model_gscv.best_score_:.3f}")

        if model_gscv.best_score_ > best_score:
            best_score = model_gscv.best_score_
            best_params = model_gscv.best_params_
            best_p = p

    print(f"\nOptimal Architecture: p={best_p}, k={best_params['kneighborsregressor__n_neighbors']}")
    return best_p, best_params['kneighborsregressor__n_neighbors']




def optimize_bioprocess_economics(df_ic, df_ts, features, batch_ids, p=2, k=5, n_sims=1000):
    idx_mab = features.index("mAb [mg/L]")
    results = []
    rng = np.random.default_rng(0)

    for test_batch in batch_ids:
        # 1. LOBOCV Split and Model Training
        train_batches = batch_ids[batch_ids != test_batch]
        x_lag_train, x_cur_train, _, _ = generate_x_lag_cur_var(p, train_batches, df_ts, features)

        model = make_pipeline(StandardScalerVAR(p=p), KNeighborsRegressor(n_neighbors=k, weights="uniform"))
        model.fit(x_lag_train, x_cur_train)

        # 2. Extract Training Residuals for White Noise
        x_cur_train_pred = model.predict(x_lag_train)
        residuals = x_cur_train_pred - x_cur_train

        # 3. Retrieve Initial Conditions
        ic_row = df_ic[df_ic["Batch ID"] == test_batch].iloc[0]
        glc_0 = ic_row["Glucose [mM]"]
        gln_0 = ic_row["Glutamine [mM]"]
        vcd_0 = ic_row["VCD [cells/mL]"]

        test_mask = df_ts["Batch ID"] == test_batch
        t_test = df_ts.loc[test_mask, "Time [h]"].to_numpy()
        x_initial = df_ts.loc[test_mask, features].to_numpy()[0]

        # 4. Run Monte Carlo Forecast Simulations
        mab_forecasts = []
        for _ in range(n_sims):
            x_forecast = [x_initial]

            while len(x_forecast) < len(t_test):
                x_lag = x_forecast[::-1][:p].copy()
                while len(x_lag) < p:
                    x_lag.append(x_lag[-1])

                x_lag_concat = np.concatenate(x_lag, axis=0).reshape(1, -1)
                noise = np.array([[rng.choice(residuals[:, j]) for j in range(len(features))]])

                x_cur = model.predict(x_lag_concat) + noise
                x_cur = np.clip(x_cur[0], a_min=0, a_max=None)
                x_forecast.append(x_cur)

            mab_forecasts.append(np.array(x_forecast)[:, idx_mab])

        mab_forecasts = np.array(mab_forecasts)

        # 5. Economic Vector Calculations
        t_days = t_test / 24.0
        total_time = t_days + const.DOWNTIME_DAYS
        cost = (const.COST_GLUCOSE_MM * glc_0) + (const.COST_GLUTAMINE_MM * gln_0) + (const.COST_VCD_CELLS_ML * vcd_0)

        value_forecasts = const.MAB_VALUE_PER_MG_L * mab_forecasts
        profit_forecasts = value_forecasts - cost
        stp_forecasts = profit_forecasts / total_time
        sty_forecasts = mab_forecasts / total_time

        # 6. Extract 95% Prediction Intervals & Optimal Harvest
        stp_median = np.quantile(stp_forecasts, 0.5, axis=0)
        stp_lower = np.quantile(stp_forecasts, 0.025, axis=0)
        stp_upper = np.quantile(stp_forecasts, 0.975, axis=0)

        best_idx = np.argmax(stp_median)

        results.append({
            "Batch ID": test_batch,
            "Optimal Harvest Time [d]": t_days[best_idx],
            "GLC_0 [mM]": glc_0,
            "GLN_0 [mM]": gln_0,
            "VCD_0 [cells/mL]": vcd_0,
            "Culture Cost [$/L]": cost,
            "Max STP [$/(L*d)]": stp_median[best_idx],
            "STP 95% PI Lower": stp_lower[best_idx],
            "STP 95% PI Upper": stp_upper[best_idx],
            "Value at Harvest [$/L]": np.quantile(value_forecasts, 0.5, axis=0)[best_idx],
            "STY at Harvest [mg/(L*d)]": np.quantile(sty_forecasts, 0.5, axis=0)[best_idx]
        })
        print(f"Evaluated Batch {test_batch}: Max STP = ${stp_median[best_idx]:.2f}")

    return pd.DataFrame(results)



def generate_analysis_figures(df_ic, df_ts, features, batch_ids, target_batch, p=2, k=5, n_sims=1000):
    print(f"\nGenerating portfolio figures for optimal Batch {target_batch}...")
    idx_mab = features.index("mAb [mg/L]")
    rng = np.random.default_rng(0)

    # 1. Train Model
    train_batches = batch_ids[batch_ids != target_batch]
    x_lag_train, x_cur_train, t_cur_train, _ = generate_x_lag_cur_var(p, train_batches, df_ts, features)
    model = make_pipeline(StandardScalerVAR(p=p), KNeighborsRegressor(n_neighbors=k, weights="uniform"))
    model.fit(x_lag_train, x_cur_train)

    # 2. Extract Training Residuals for White Noise
    x_cur_train_pred = model.predict(x_lag_train)
    residuals = x_cur_train_pred - x_cur_train

    # --- FIGURE 1: White Noise Residuals (Professor's Methodology) ---
    t_unique = np.unique(t_cur_train)
    res_mean = []
    res_pi_upper = []
    res_pi_lower = []
    alpha = 0.05

    for t in t_unique:
        mask = t_cur_train == t
        res_t = residuals[mask, idx_mab]
        n_mean = np.mean(res_t)
        n_sd = np.std(res_t)

        # Student's t-distribution for 95% PI
        dof = res_t.shape[0] - 1
        pi_sd = stats.t.isf(q=alpha / 2, df=dof)

        res_mean.append(n_mean)
        res_pi_upper.append(n_mean + pi_sd * n_sd)
        res_pi_lower.append(n_mean - pi_sd * n_sd)

    res_mean = np.array(res_mean)
    res_pi_upper = np.array(res_pi_upper)
    res_pi_lower = np.array(res_pi_lower)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(t_cur_train, residuals[:, idx_mab], alpha=0.7, s=12, color="tab:orange", label="Noise",
                edgecolors="black", linewidth=0.5, zorder=3)
    ax1.plot(t_unique, res_mean, color="black", label="Mean", zorder=4)
    ax1.fill_between(t_unique, res_pi_lower, res_pi_upper, color="tab:blue", alpha=0.3, label="95% PI",
                     edgecolors="none", zorder=2)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("mAb [mg/L] (Residuals)")
    ax1.legend(loc="upper left")
    fig1.savefig(os.path.join("figures", "fig_01_white_noise.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 3. Setup Test Data (Batch 127)
    test_mask = df_ts["Batch ID"] == target_batch
    t_test = df_ts.loc[test_mask, "Time [h]"].to_numpy()
    x_test = df_ts.loc[test_mask, features].to_numpy()
    x_initial = x_test[0]

    ic_row = df_ic[df_ic["Batch ID"] == target_batch].iloc[0]
    cost = (const.COST_GLUCOSE_MM * ic_row["Glucose [mM]"]) + (const.COST_GLUTAMINE_MM * ic_row["Glutamine [mM]"]) + (
                const.COST_VCD_CELLS_ML * ic_row["VCD [cells/mL]"])

    # 4. Simulate Forecasts
    mab_forecasts = []
    for _ in range(n_sims):
        x_forecast = [x_initial]
        while len(x_forecast) < len(t_test):
            x_lag = x_forecast[::-1][:p].copy()
            while len(x_lag) < p: x_lag.append(x_lag[-1])
            x_lag_concat = np.concatenate(x_lag, axis=0).reshape(1, -1)
            noise = np.array([[rng.choice(residuals[:, j]) for j in range(len(features))]])
            x_cur = np.clip(model.predict(x_lag_concat) + noise, a_min=0, a_max=None)[0]
            x_forecast.append(x_cur)
        mab_forecasts.append(np.array(x_forecast)[:, idx_mab])

    mab_forecasts = np.array(mab_forecasts)
    mab_median = np.quantile(mab_forecasts, 0.5, axis=0)
    mab_lower = np.quantile(mab_forecasts, 0.025, axis=0)
    mab_upper = np.quantile(mab_forecasts, 0.975, axis=0)

    # --- FIGURE 2: Forecast Comparison ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(t_test, x_test[:, idx_mab], color="tab:orange", s=20, label="Actual Test Data", edgecolors="black",
                zorder=3)
    ax2.plot(t_test, mab_median, color="black", linewidth=2, label="k-NN NVAR Forecast", zorder=4)
    ax2.fill_between(t_test, mab_lower, mab_upper, color="tab:blue", alpha=0.3, label="95% PI", zorder=2)
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("mAb [mg/L]")
    ax2.legend(loc="upper left")
    fig2.savefig(os.path.join("figures", "fig_02_forecast_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # --- FIGURE 3: Space-Time Profit ---
    t_days = t_test / 24.0
    total_time = t_days + const.DOWNTIME_DAYS
    profit_forecasts = (const.MAB_VALUE_PER_MG_L * mab_forecasts) - cost
    stp_forecasts = profit_forecasts / total_time

    stp_median = np.quantile(stp_forecasts, 0.5, axis=0)
    stp_lower = np.quantile(stp_forecasts, 0.025, axis=0)
    stp_upper = np.quantile(stp_forecasts, 0.975, axis=0)
    best_idx = np.argmax(stp_median)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(t_days, stp_median, color="black", linewidth=2, label="Median STP")
    ax3.fill_between(t_days, stp_lower, stp_upper, color="tab:green", alpha=0.3, label="95% PI")
    ax3.axvline(x=t_days[best_idx], color="tab:red", linestyle="--", linewidth=2,
                label=f"Optimum Harvest: Day {t_days[best_idx]:.2f}")
    ax3.set_xlabel("Time [d]")
    ax3.set_ylabel("Space-Time Profit [$/(L*d)]")
    ax3.legend(loc="lower right")
    fig3.savefig(os.path.join("figures", "fig_03_space_time_profit.png"), dpi=300, bbox_inches="tight")
    plt.close(fig3)

    print("Figures 01, 02, and 03 successfully exported to figures/!")