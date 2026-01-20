import pandas as pd
import sys

#script for using unique name of column of rpediction passed/not passed for both environments 
SRC = "likely_pass_like"
DST = "likely_by_horizon_conservative"
DROP_SRC = True

def main(in_path: str, out_path: str):
    df = pd.read_csv(in_path)

    if SRC not in df.columns:
        raise SystemExit(f"Column '{SRC}' not found. Columns: {list(df.columns)}")
    if DST not in df.columns:
        df[DST] = pd.NA

    df[DST] = df[DST].where(df[DST].notna(), df[SRC])

    if DROP_SRC:
        df = df.drop(columns=[SRC])

    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}. Filled '{DST}' from '{SRC}'{' and dropped SRC' if DROP_SRC else ''}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("python utils/standardize_naming_column.py phase3.csv phase3_corrected.csv")
    main(sys.argv[1], sys.argv[2])
