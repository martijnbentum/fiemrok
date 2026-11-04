'''Find and report Tukey duration outliers per key.'''

import numpy as np


RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'
CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RED = '\033[31m'


def tukey_fences(durations, k=1.5):
    '''Return the lower and upper Tukey fences for an array of durations.'''
    q1, q3 = np.percentile(durations, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def find_outliers(groups, k=1.5):
    '''Return {key: [objects whose duration falls outside Tukey's fences]}.
    groups:    dict mapping key -> iterable of objects with a .duration
    k:         IQR multiplier (1.5 = standard, 3 = only extreme outliers)
    '''
    outliers = {}
    for key, objects in groups.items():
        objects = list(objects)
        if not objects:
            outliers[key] = []
            continue
        durations = np.array([o.duration for o in objects], dtype=float)
        lower, upper = tukey_fences(durations, k)
        mask = (durations < lower) | (durations > upper)
        outliers[key] = [o for o, is_out in zip(objects, mask) if is_out]
    return outliers


def report_outliers(groups, k=1.5, precision=3):
    '''Print per-key duration stats and Tukey outliers, color coded.
    Median, mean, min and max are computed on the inliers only.
    Short outliers are blue, long outliers red.
    groups:       dict mapping key -> iterable of objects with a .duration
    k:            IQR multiplier (1.5 = standard, 3 = only extreme outliers)
    precision:    decimals shown for durations
    '''
    key_width = max([len(str(key)) for key in groups] + [3])
    col = precision + 7
    header = (f'{"key":<{key_width}}  {"n":>4}  {"median":>{col}}  '
        f'{"mean":>{col}}  {"min":>{col}}  {"max":>{col}}  outliers')
    print(BOLD + header + RESET)
    print(DIM + '-' * len(header) + RESET)
    for key, objects in groups.items():
        durations = np.array([o.duration for o in objects], dtype=float)
        print(_report_line(key, durations, k, precision, key_width, col))
    print(DIM + f'\nTukey fences, k={k}  ' + RESET + BLUE + 'too short'
        + RESET + '  ' + RED + 'too long' + RESET)


def _report_line(key, durations, k, precision, key_width, col):
    '''Format one colored report line for a key.'''
    name = f'{BOLD}{CYAN}{str(key):<{key_width}}{RESET}'
    if durations.size == 0: return f'{name}  {DIM}(empty){RESET}'
    fmt = lambda x: f'{x:.{precision}f}'
    lower, upper = tukey_fences(durations, k)
    inside = durations[(durations >= lower) & (durations <= upper)]
    low = np.sort(durations[durations < lower])
    high = np.sort(durations[durations > upper])
    out = [BLUE + fmt(d) + RESET for d in low]
    out += [RED + fmt(d) + RESET for d in high]
    n_out = len(out)
    out_str = f'({n_out}) ' + ', '.join(out) if n_out else DIM + '-' + RESET
    return (f'{name}  {len(durations):>4}  '
        f'{GREEN}{fmt(np.median(inside)):>{col}}{RESET}  '
        f'{YELLOW}{fmt(np.mean(inside)):>{col}}{RESET}  '
        f'{fmt(inside.min()):>{col}}  {fmt(inside.max()):>{col}}  {out_str}')
