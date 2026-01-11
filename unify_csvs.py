import pandas as pd
import glob

# Path to teh CSVs
files = glob.glob("Csv_Pyramids1/*.csv")
dfs = [pd.read_csv(f) for f in files]

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv("merged.csv", index=False)

print("Done")