# Worm Phase 3 Prediction

Predict likelihood of Phase 3 Worm runs (2M budget) reaching the Worm threshold by 5M steps. The script reads TensorBoard event data from each run, performs linear extrapolation, and writes a CSV with predicted reward at 5M plus LIKELY/UNLIKELY labels (optimistic and conservative).

---

## Folder Setup

- Create a folder containing only Phase 3 Worm run subfolders (each run is a subfolder).
- Each run folder must include TensorBoard event files (e.g., events.out.tfevents...). Do not copy only .onnx files.

Example structure:

`	ext
Worm_phase3_runs/
  WormA_0050_s0/
  WormA_0051_s0/
  WormA_0052_s0/
`

---

## How to Run

Recommended oneliner:

`powershell
python utils/phase3/predict_worm_phase3_to_5m.py \
  --phase3-runs-dir "D:\\...\\Worm_phase3_runs" \
  --out "D:\\...\\worm_phase3_predictions.csv" \
  --require-positive-slope
`

After running, open the output CSV and use likely_by_horizon_conservative as your main label (recommended).

---

## Parameters

- **--phase3-runs-dir**: Path to the folder containing Phase 3 Worm run subfolders.
- **--out**: Path to the output CSV file to write predictions.
- **--require-positive-slope**: If set, only consider runs whose extrapolated slope is positive; otherwise label as unlikely.
