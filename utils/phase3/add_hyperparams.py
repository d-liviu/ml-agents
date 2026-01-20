import pandas as pd

csv1 = pd.read_csv("configs_Worm.csv")
csv2 = pd.read_csv("worm_phase3_predictions.csv")
csv2.columns = csv2.columns.str.strip()

# we need to keep all columns up to final_perf from the tensorboard_to_csv (hyperparameters of new trainings) + the likely column that conveys wether threshold is reached/not reached from new trainings phase 3 script
csv1_trim = csv1.iloc[:, :10]
csv2_trim = csv2[["run_id", "likely_by_horizon", "likely_by_horizon_conservative"]]

merged = csv1_trim.merge(csv2_trim, on="run_id", how="inner")
merged.to_csv("worm_final.csv", index=False)
print("ok")
