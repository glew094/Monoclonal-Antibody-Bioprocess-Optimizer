import pandas as pd
import numpy as np


def generate_doe_batches():
    # Load the initial conditions mapping
    df = pd.read_csv("datasets/initial_conditions.csv")
    cols = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]

    # Identify the levels for each variable
    levels = {col: np.sort(df[col].unique()) for col in cols}
    extremes = {col: (levels[col][0], levels[col][-1]) for col in cols}
    medians = {col: levels[col][len(levels[col]) // 2] for col in cols}

    selected_batches = []

    # 1. Corner Points (8 batches): Min and Max combinations for all variables
    for glu in extremes[cols[0]]:
        for gln in extremes[cols[1]]:
            for vcd in extremes[cols[2]]:
                batch = df[(df[cols[0]] == glu) & (df[cols[1]] == gln) & (df[cols[2]] == vcd)].iloc[0]
                selected_batches.append(batch['Batch ID'])

    # 2. Center Points (3 replicates): All variables exactly at their medians to capture variance
    center_df = df[(df[cols[0]] == medians[cols[0]]) &
                   (df[cols[1]] == medians[cols[1]]) &
                   (df[cols[2]] == medians[cols[2]])]
    selected_batches.extend(center_df['Batch ID'].tolist())

    # 3. Axial Points (6 batches): One extreme variable, two median variables
    for col in cols:
        other_cols = [c for c in cols if c != col]
        for extreme_val in extremes[col]:
            batch = df[(df[col] == extreme_val) &
                       (df[other_cols[0]] == medians[other_cols[0]]) &
                       (df[other_cols[1]] == medians[other_cols[1]])].iloc[0]
            selected_batches.append(batch['Batch ID'])

    # 4. Space-Filling Points: Randomly select the remaining files from the interior parameter space
    remaining_df = df[~df['Batch ID'].isin(selected_batches)]
    interior_df = remaining_df[
        (~remaining_df[cols[0]].isin(extremes[cols[0]])) &
        (~remaining_df[cols[1]].isin(extremes[cols[1]])) &
        (~remaining_df[cols[2]].isin(extremes[cols[2]]))
        ]

    needed = 30 - len(selected_batches)
    space_filling = interior_df.sample(n=needed, random_state=42)
    selected_batches.extend(space_filling['Batch ID'].tolist())

    # Output the exact file names to download
    print("Download these 30 files from the provided_files directory:")
    for batch in selected_batches:
        print(f"batch_{str(int(batch)).zfill(4)}.csv")


if __name__ == "__main__":
    generate_doe_batches()