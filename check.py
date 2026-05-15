import json
with open('results/calibration/calibration_summary.json') as f:
    s = json.load(f)
print('ECE:', s['ece'])
for b in s['calibration_bins']:
    if b['fraction_positive'] is not None:
        print(f"  [{b['bin_lower']}-{b['bin_upper']}] pred={b['mean_pred_prob']} actual={b['fraction_positive']}")
        