# ---------------------------------------------------------------------------
# C1 head comparison — paste as a new cell AFTER section 6 of 04_phase4.ipynb
#
# The question: is C1's failure a representational limit (a regressed
# coordinate cannot hold the disc's two possible sides) or something else?
#
# A synthetic bimodal fixture could not answer it — both heads solved it
# identically, 0.177 vs 0.173 DD with zero flips each, because a synthetic disc
# is far more salient than a real one. Only this data discriminates.
#
# Runs as a SEPARATE experiment id, so C1_geometry's recorded 0.686 DD result is
# untouched whichever way this lands.
# ---------------------------------------------------------------------------
C1_HEATMAP = 'C1_geometry_heatmap'

if not C1_READY:
    print('SKIPPED -', C1_WHY)
else:
    run(' '.join([
        f"python {q(REPO_DIR / 'scripts/train_geometry.py')}",
        f'--manifest {q(C1_MANIFEST)}',
        f'--experiment {q(C1_HEATMAP)}',
        f'--results-dir {q(RESULTS)}',
        f'--cache-root {CACHE_FLAGS}',
        '--head heatmap',
        '--image-size 512 --batch-size 16 --epochs 40 --lr 3e-4',
        '--val-frac 0.2 --patience 10 --workers 2 --resume',
    ]))

    # ---- the comparison that decides it ----------------------------------
    import json as _json
    print()
    print('=' * 72)
    print('C1 — coordinate regression vs heatmap, same data and split')
    print('=' * 72)

    rows = []
    for label, name in (('regress', C1_EXPERIMENT), ('heatmap', C1_HEATMAP)):
        path = RESULTS / name / 'metrics.json'
        if not path.exists():
            print(f'  {label}: no metrics.json — not run')
            continue
        m = _json.loads(path.read_text())
        b = m['best_val']
        errs = RESULTS / name / 'val_errors.csv'
        flips = n = 0
        if errs.exists():
            import csv
            with open(errs) as fh:
                recs = list(csv.DictReader(fh))
            n = len(recs)
            flips = sum(int(r['side_flipped']) for r in recs)
        rows.append({
            'head': label,
            'mean DD': round(m['best_mean_error_dd'], 3),
            'OD DD': round(b['od_error_dd'], 3),
            'fovea DD': round(b['fovea_error_dd'], 3),
            'within 0.5': f"{b['within_half_dd'] * 100:.1f}%",
            'flips': f'{flips}/{n}' if n else '—',
            'gate': 'PASS' if m['best_mean_error_dd'] < 0.5 else 'FAIL',
        })

    if rows:
        import pandas as pd
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    if len(rows) == 2:
        reg, hm = rows[0], rows[1]
        d = reg['mean DD'] - hm['mean DD']
        print(f"  heatmap moves the mean by {d:+.3f} DD")
        rf = int(reg['flips'].split('/')[0]) if '/' in reg['flips'] else None
        hf = int(hm['flips'].split('/')[0]) if '/' in hm['flips'] else None
        if rf is not None and hf is not None:
            print(f"  laterality flips {rf} -> {hf}")
            if hf < rf and hm['gate'] == 'PASS':
                print('  -> The diagnosis held AND the gate passes. M3 keeps quadrant')
                print('     reasoning, so R4 (the 4-2-1 rule) can fire.')
            elif hf < rf:
                print('  -> Fewer flips, gate still fails. The diagnosis was right and')
                print('     something else also limits it; report both.')
            else:
                print('  -> Flips did NOT drop. The bimodality explanation is wrong, so')
                print('     record that and keep the regression result. M3 falls back to')
                print('     count-only rules, which docs/00_START_HERE.md names as the')
                print('     designed fallback.')
        print()
        print('  Whichever wins, C1_geometry (0.686 DD) stays the recorded first result.')
        print('  A second architecture tried after seeing a failure is a deviation to')
        print('  write down, not a replacement for what was pre-specified.')
