# Phase 6a status check -- CPU ONLY. Turn the accelerator OFF before running this:
# it reads files and trains nothing, so there is no reason to spend GPU on it.
#
# Attach: verify-dr-phase6 and verify-dr-manifests. Paste into a fresh notebook.
import json
from pathlib import Path

import pandas as pd

INPUT = Path('/kaggle/input')
RUNS = [f'H1_{v}_s{s}' for v in ('eyepacs_full', 'eyepacs_ddr_full') for s in (42, 43, 44)]


def first(pattern):
    hits = sorted(INPUT.rglob(pattern))
    return hits[0] if hits else None


# ---- 1. Are all six runs finished -- and did any collapse? --------------------
# A directory existing proves only that a run STARTED. metrics.json is written
# once, after the last epoch; a run killed mid-way leaves checkpoint.pt alone.
done = {}
for p in sorted(INPUT.rglob('H1_*/metrics.json')):
    done.setdefault(p.parent.name, p)
started = {p.parent.name for p in INPUT.rglob('H1_*/checkpoint.pt')}

rows = []
for name in RUNS:
    if name not in done:
        rows.append({'run': name,
                     'status': 'PARTIAL - resume it' if name in started else 'MISSING'})
        continue
    m = json.loads(done[name].read_text())
    b = m.get('best_val') or {}
    rows.append({
        'run': name, 'status': 'done',
        'epochs': m.get('epochs_run'), 'best ep': m.get('best_epoch'),
        'QWK': round(b.get('qwk', float('nan')), 4),
        'macro-F1': round(b.get('macro_f1', float('nan')), 4),
        'g1-F1': round(b.get('per_class_f1', {}).get('1', float('nan')), 4),
        'distinct': b.get('distinct_predictions'),
    })

print('=' * 78)
print('1. PHASE 6a RUNS')
print('=' * 78)
table = pd.DataFrame(rows)
for col in ('epochs', 'best ep', 'distinct'):      # NaN from unfinished rows forces floats
    if col in table:
        table[col] = table[col].astype('Int64')
print(table.to_string(index=False))
n_done = sum(r['status'] == 'done' for r in rows)
collapsed = [r['run'] for r in rows if r.get('distinct') == 1]
print()
if n_done < len(RUNS):
    print(f'{len(RUNS) - n_done} run(s) not finished. Re-open 06_final_training with')
    print('verify-dr-phase6 attached; section 6 carries them forward and --resume')
    print('continues each from its last epoch.')
if collapsed:
    print(f'COLLAPSED (predicts one grade for every image): {collapsed}')
    print('These decide nothing, whatever their QWK. They must be retrained.')
if n_done == len(RUNS) and not collapsed:
    print('All six complete and none collapsed. Phase 6a training is DONE.')

# ---- 2. Which EyePACS split did the six models actually train on? -------------
print()
print('=' * 78)
print('2. THE EYEPACS SPLIT')
print('=' * 78)
plan = first('dataset_plan.json')
if plan is not None:
    policy = json.loads(plan.read_text()).get('eyepacs', {}).get('split_policy', {})
    print('dataset_plan.json records:', policy)
else:
    print('dataset_plan.json not found -- attach verify-dr-manifests')

full = first('eyepacs_full.csv')
if full is None:
    print('eyepacs_full.csv not found -- attach verify-dr-manifests')
else:
    f = pd.read_csv(full, usecols=lambda c: c in ('split', 'source_split'))
    src = (f['source_split'].fillna('').astype(str) if 'source_split' in f.columns
           else pd.Series('', index=f.index))
    if not src.ne('').any():
        print()
        print('No source_split recorded: the mirror did not say which images were the')
        print('official test set. They cannot be identified, so a protocol-matched')
        print('leaderboard number is impossible from this cache -> OPTION A.')
    else:
        print()
        print('rows = the mirror\'s own split; columns = the split we trained with')
        print(pd.crosstab(src.replace('', '(none)'), f['split'], margins=True))
        official_test = f[src == 'test']
        seen = int(official_test['split'].isin(['train', 'val', 'calibration']).sum())
        print()
        print(f'mirror marks {len(official_test):,} images as test '
              f'(the official test set is 53,576)')
        if len(official_test):
            print(f'of those, {seen:,} ({seen / len(official_test):.0%}) sit in our '
                  f'train/val/calibration splits --')
            print('the six models trained on them, early-stopped on them, or will be')
            print('calibrated on them. Scoring those models on the "official test set"')
            print('would largely score them on their own training data.')
        # Option C prices a model trained on the official TRAIN images only.
        n_train = int((src == 'train').sum())
        fr = json.loads(plan.read_text()).get('eyepacs', {}) if plan else {}
        keep = 1 - fr.get('val_frac', 0.10) - fr.get('calibration_frac', 0.05)
        h = n_train * keep * 10 / 46.0 / 3600
        print()
        print(f'OPTION C is possible: {n_train:,} official-train images, ~{n_train * keep:,.0f}')
        print(f'after carving val/calibration -> ~{h:.1f} GPU-h per seed at the frozen recipe.')
