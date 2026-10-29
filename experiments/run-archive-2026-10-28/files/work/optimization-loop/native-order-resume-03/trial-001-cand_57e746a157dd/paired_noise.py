"""Exploratory operator A/A: reference-style twenty-pair confidence gate.

Pairing cancels shared temporal drift. Raw wall-time MAD/range stay visible;
the earlier four-wall-time MAD gate is not used as a paired uncertainty estimate.
This is a new diagnostic protocol, not a recertification of historical results.
"""
import random
import statistics as st


def screen(rows, seed=42):
    a={r['pair']:r for r in rows if r['lane']=='baseline'}
    b={r['pair']:r for r in rows if r['lane']=='baseline_repeat'}
    if set(a)!=set(range(1,21)) or set(b)!=set(a) or len(rows)!=40:
        raise ValueError('A/A requires exactly twenty complete pairs')
    if any(r['environment_issues'] for r in rows):
        raise ValueError('A/A environment invalid')
    ratios=[(a[p]['elapsed_ms']/a[p]['calls'])/(b[p]['elapsed_ms']/b[p]['calls']) for p in range(1,21)]
    rng=random.Random(seed)
    boot=sorted(st.median(rng.choices(ratios,k=20)) for _ in range(10000))
    lo,hi=boot[249],boot[9749]
    median=st.median(ratios)
    order=[st.median(ratios[0::2]),st.median(ratios[1::2])]
    times=[r['elapsed_ms']/r['calls'] for r in rows]
    center=st.median(times)
    return {
        'method':'twenty_pair_bootstrap_ci_includes_one_width_at_most_four_percentage_points',
        'ratios':ratios,'median_ratio':median,'descriptive_bootstrap_95pct':[lo,hi],
        'order_medians':order,'maximum_order_bias':max(abs(v-1) for v in order),
        'relative_mad':st.median(abs(v-center) for v in times)/center,
        'full_range':(max(times)-min(times))/center,
        'raw_mad_and_range_are_diagnostics':True,
        'passed':lo<=1<=hi and hi-lo<=.04 and abs(median-1)<=.02
                 and all(abs(v-1)<=.02 for v in order),
    }
