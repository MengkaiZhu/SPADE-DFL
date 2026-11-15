"""Independent, explicit statistics for frozen result verification."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy import stats


def mean_sd(values, ddof: int, expected_count: int | None = None):
    array = np.asarray(tuple(values), dtype=np.float64)
    if expected_count is not None and array.size != expected_count:
        raise ValueError(f"expected seeds/count {expected_count}, found {array.size}")
    if ddof not in (0, 1) or array.ndim != 1 or array.size <= ddof or not np.all(np.isfinite(array)):
        raise ValueError("values must be finite and sufficient for ddof=0 or 1")
    return float(array.mean()), float(array.std(ddof=ddof))


def pool_fold_counts(rows):
    totals = defaultdict(lambda: [0, 0])
    for row in rows:
        correct, count = int(row['correct']), int(row.get('count', row.get('eval_count', 0)))
        if count <= 0 or not 0 <= correct <= count:
            raise ValueError('invalid fold evaluation counts')
        totals[int(row['seed'])][0] += correct
        totals[int(row['seed'])][1] += count
    if not totals:
        raise ValueError('empty fold records')
    return {seed: correct / count for seed, (correct, count) in sorted(totals.items())}


pool_folds = pool_fold_counts


def benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('p values must be a nonempty finite probability vector')
    order = np.argsort(values, kind='stable')
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1)
    return result


def mean_ci95(values):
    mean, sd = mean_sd(values, ddof=1)
    half = float(stats.t.ppf(.975, len(values) - 1)) * sd / np.sqrt(len(values))
    return mean, mean - half, mean + half
