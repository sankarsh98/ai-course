"""Time-series anomaly detection on an hourly series with labelled anomalies:
a global threshold, a seasonal robust z-score, and a forecast-residual score.
Then: how point adjustment makes even random scores look excellent."""
import numpy as np

rng = np.random.default_rng(1)
P = 24                                    # hourly data, daily cycle
n = 60 * P                                # 60 days
t = np.arange(n)
daily = 10 * np.sin(2 * np.pi * (t % P) / P - np.pi / 2)          # low at night
y = 50 + daily + 0.01 * t + rng.normal(0, 1.5, n)
label = np.zeros(n, bool)

# Four kinds of anomaly, each a labelled segment (start hour, length).
for s in [400, 700, 1100]:                       # point spikes
    y[s] += 12; label[s] = True
y[900:924] += 6; label[900:924] = True           # level shift lasting one day
night = 1203                                     # contextual: a daytime value at 3 am
y[night:night + 3] = 60 + 0.01 * t[night:night + 3] + rng.normal(0, 1.5, 3)  # daily-peak level
label[night:night + 3] = True
y[1300:1340] += rng.normal(0, 6, 40); label[1300:1340] = True      # variance burst
print(f"{n} points, {label.sum()} labelled anomalous ({label.mean():.1%}) in 6 segments")


def seasonal_robust_z(y, window_days=14):
    """Compare each point with the same hour on previous days: subtract the
    trailing median for that hour, divide by a robust scale (MAD). Uses only
    the past, so it can run online."""
    z = np.zeros(len(y))
    for i in range(window_days * P, len(y)):
        same_hour = y[i - window_days * P:i:P]          # same hour, previous days
        med = np.median(same_hour)
        mad = np.median(np.abs(same_hour - med)) * 1.4826 + 1e-9  # ~std under normality
        z[i] = abs(y[i] - med) / mad
    return z


def forecast_residual(y):
    """|y_t - y_{t-24}|: the error of a seasonal naive forecast, scaled by a
    trailing robust scale of past errors."""
    r = np.zeros(len(y))
    r[P:] = np.abs(y[P:] - y[:-P])
    score = np.zeros(len(y))
    for i in range(8 * P, len(y)):
        score[i] = r[i] / (np.median(r[i - 7 * P:i]) + 1e-9)
    return score


def segments(lab):
    """List of (start, end) index pairs of contiguous True runs."""
    d = np.diff(np.r_[0, lab.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def point_adjust(pred, lab):
    """The 'point adjustment' protocol: if any point of a true segment is
    flagged, count the whole segment as detected."""
    adj = pred.copy()
    for s, e in segments(lab):
        if pred[s:e].any():
            adj[s:e] = True
    return adj


def f1(pred, lab):
    tp = np.sum(pred & lab)
    prec = tp / max(pred.sum(), 1); rec = tp / lab.sum()
    return 0 if tp == 0 else 2 * prec * rec / (prec + rec)


def best_f1(score, lab, adjust=False):
    """Best F1 over thresholds (an optimistic, oracle-tuned number, as in many papers)."""
    best = 0
    for q in np.linspace(0.80, 0.999, 200):
        pred = score > np.quantile(score, q)
        best = max(best, f1(point_adjust(pred, lab) if adjust else pred, lab))
    return best


def segment_recall(score, lab, k):
    """Of the anomaly segments, how many contain at least one of the top-k scores?"""
    top = np.zeros(len(score), bool); top[np.argsort(score)[-k:]] = True
    return [bool(top[s:e].any()) for s, e in segments(lab)]


warm = 14 * P                         # skip the warm-up period the online scores need
scores = {"global |z| of raw value": np.abs(y - y.mean()) / y.std(),
          "seasonal robust z": seasonal_robust_z(y),
          "forecast residual": forecast_residual(y),
          "random scores": rng.random(n)}
names = ["spike", "spike", "spike", "level shift", "night value", "variance burst"]
print(f"\n{'detector':<25} {'F1':>5} {'PA-F1':>6}   segments caught by top-{label.sum()} scores")
for name, s in scores.items():
    s, lab = s[warm:], label[warm:]
    caught = segment_recall(s, lab, label.sum())
    hit = [nm for nm, c in zip(names, caught) if c]
    print(f"{name:<25} {best_f1(s, lab):5.2f} {best_f1(s, lab, adjust=True):6.2f}   {', '.join(hit) or 'none'}")
