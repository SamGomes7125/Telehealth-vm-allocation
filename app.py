import os, json, hashlib, numpy as np, pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__, static_folder=".")
CORS(app)


DAYS  = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
HOURS = [f"{h:02d}:{m:02d}" for h in range(8,18) for m in (0,30)]

# SLOTS_PER_WEEK used everywhere a "one-week lag" is needed — never hardcode 100
SLOTS_PER_WEEK = len(DAYS) * len(HOURS)   # 5 × 20 = 100

# ---------------------------------------------------------------------------
# Demand weights (Multiplicative Holt-Winters decomposition)
# Reference: Hyndman & Athanasopoulos, Forecasting: Principles and Practice
#            (3rd ed., 2021), Ch. 8
# ---------------------------------------------------------------------------
TIME_W = {
    "08:00":0.70,"08:30":0.85,"09:00":1.20,"09:30":1.35,
    "10:00":1.30,"10:30":1.25,"11:00":1.20,"11:30":1.10,
    "12:00":0.80,"12:30":0.65,"13:00":0.70,"13:30":0.85,
    "14:00":1.10,"14:30":1.15,"15:00":1.20,"15:30":1.10,
    "16:00":1.00,"16:30":0.90,"17:00":0.75,"17:30":0.60,
}
DAY_W = {"Monday":1.30,"Tuesday":1.10,"Wednesday":1.00,"Thursday":0.95,"Friday":0.85}

BASE_WEEKLY  = 105
DEMAND_SCALE = 4.0
MAX_SESSIONS = 40
CPU_SES, RAM_SES = 2, 2.0
DEFAULT_BUFFER   = 2.10
VM_CPU = 8;  VM_RAM = 16.0
MIN_VMS = 1; FLAT_VMS = 4

# Fix #7: clamp buffer to a sensible range at the constant level too
BUFFER_MIN, BUFFER_MAX = 1.0, 5.0

# Fix #9: clamp provisioning delay to half a week at most
DELAY_MAX = SLOTS_PER_WEEK // 2   # 50 slots = 25 hours



# ---------------------------------------------------------------------------
# Training data synthesis
# Demand is normalised by BASE_WEEKLY before training so the model learns
# relative slot patterns (0–1 scale), not absolute counts. This removes the
# need for a post-hoc output scaling hack and keeps lag features consistent
# with any week_total the user requests.
#
# Fix #3: normalise demand by weekly total during training so lag features
# and predictions are on the same relative scale as any user-supplied
# week_total. Post-hoc multiplication by week_total then gives correct
# absolute counts without feeding unscaled lags to a scaled output.
#
# Demand: d(w,day,t) ~ Poisson(λ)  where
#   λ = (W_weekly / SLOTS_PER_WEEK) · β_day · β_time · Δ
# Reference for Poisson count model:
#   Cameron & Trivedi, Regression Analysis of Count Data (2013)
# ---------------------------------------------------------------------------
def build_training_data(n_weeks=104):
    np.random.seed(42)
    X, y = [], []
    demand_history = []   # stores normalised demands (divided by weekly total)
    base = pd.Timestamp("2023-01-02")
    for w in range(n_weeks):
        ws  = base + pd.Timedelta(weeks=w)
        tr  = 1.0 + 0.003 * w
        sea = 1.0 + 0.15 * np.cos(2 * np.pi * (ws.month - 1) / 12)
        wt  = int(BASE_WEEKLY * tr * sea * np.random.uniform(0.90, 1.10))
        for day in DAYS:
            for ts in HOURS:
                rate = (wt / SLOTS_PER_WEEK) * DAY_W[day] * TIME_W[ts] * DEMAND_SCALE
                dem  = np.random.poisson(max(rate, 0.5))

                # Occasional surge event (3 % probability)
                if np.random.rand() < 0.03:
                    dem = int(dem * np.random.uniform(1.3, 1.8))

                dem = min(dem, MAX_SESSIONS)

                # Normalise by actual weekly total so the model learns
                # relative demand share per slot, not absolute counts.
                dem_norm = dem / max(wt, 1)

                # Lag features on the same normalised scale
                lag1     = demand_history[-1]              if len(demand_history) >= 1              else dem_norm
                lag2     = demand_history[-2]              if len(demand_history) >= 2              else dem_norm
                lag_week = demand_history[-SLOTS_PER_WEEK] if len(demand_history) >= SLOTS_PER_WEEK else dem_norm

                X.append([lag1, lag2, lag_week, DAY_W[day], TIME_W[ts], tr, sea])
                y.append(dem_norm)
                demand_history.append(dem_norm)

    return np.array(X), np.array(y), demand_history


# ---------------------------------------------------------------------------
# Model training
# Ridge regression minimises: L(β) = ‖y − Xβ‖² + α‖β‖²
# Reference: Hoerl & Kennard (1970), "Ridge Regression: Biased Estimation for
#            Nonorthogonal Problems", Technometrics, 12(1).
# RidgeCV selects α via leave-one-out CV — more robust than a fixed α=1.0
# ---------------------------------------------------------------------------
X, y, _training_history = build_training_data()

# Chronological split — no shuffle to prevent temporal leakage.
# Reference: Bergmeir & Benítez, "On the use of cross-validation for time
#            series predictor evaluation", Information Sciences (2012).
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

# RidgeCV evaluates α over a log-spaced grid and picks the best via LOO-CV
regressor = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
regressor.fit(X_train_scaled, y_train)

test_preds = regressor.predict(X_test_scaled)
MODEL_MAE  = round(mean_absolute_error(y_test, test_preds), 2)
MODEL_RMSE = round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 2)


# ---------------------------------------------------------------------------
# Inference — predict one week of 100 slots.
# Predictions are in normalised units (demand / weekly_total); multiplying
# by week_total at the end gives correct absolute session counts for any
# requested weekly booking volume. Lag features stay on the same normalised
# scale throughout, so extrapolation beyond BASE_WEEKLY is consistent.
# ---------------------------------------------------------------------------
def linear_regression_predict_week(week_total):
    preds_norm = []
    week_idx   = 104   # one week beyond the training window

    base    = pd.Timestamp("2023-01-02")
    current = base + pd.Timedelta(weeks=week_idx)
    tr      = 1.0 + 0.003 * week_idx
    sea     = 1.0 + 0.15 * np.cos(2 * np.pi * (current.month - 1) / 12)

    # Seed with real normalised training tail so lag_week is meaningful
    recent_norm = list(_training_history[-SLOTS_PER_WEEK:])

    for day in DAYS:
        for ts in HOURS:
            lag1     = recent_norm[-1]
            lag2     = recent_norm[-2]
            lag_week = recent_norm[-SLOTS_PER_WEEK] if len(recent_norm) >= SLOTS_PER_WEEK else recent_norm[-1]

            features        = np.array([[lag1, lag2, lag_week, DAY_W[day], TIME_W[ts], tr, sea]])
            features_scaled = scaler_X.transform(features)

            pred_norm = float(regressor.predict(features_scaled)[0])
            pred_norm = max(pred_norm, 0)

            preds_norm.append(pred_norm)
            recent_norm.append(pred_norm)

    # Convert from relative share back to absolute sessions for this week_total
    return [p * week_total for p in preds_norm]


def formula_pred(week_total, day, time_str):
    return (week_total / SLOTS_PER_WEEK) * DAY_W[day] * TIME_W[time_str] * DEMAND_SCALE


# ---------------------------------------------------------------------------
# VM allocation — bin-packing lower bound
# vms = max( ⌈sessions·CPU_SES/VM_CPU⌉, ⌈sessions·RAM_SES/VM_RAM⌉, MIN_VMS )
# Both CPU and RAM constraints are binding; whichever requires more VMs wins.
# Reference: Coffman et al., "Approximation Algorithms for Bin Packing" (1997)
# Cloud auto-scaling: Lorido-Botran et al., J. Grid Computing (2014)
# ---------------------------------------------------------------------------
def allocate(pred, buf):
    buffered = int(np.ceil(pred * buf))
    vms_cpu  = int(np.ceil(buffered * CPU_SES / VM_CPU))
    vms_ram  = int(np.ceil(buffered * RAM_SES / VM_RAM))
    vms      = max(vms_cpu, vms_ram, MIN_VMS)
    return {"buffered": buffered, "vms": vms, "cpu": vms * VM_CPU, "ram": vms * VM_RAM}


# ---------------------------------------------------------------------------
# Response time — pure M/M/1 queue model
#
# The M/M/1 queue models a single-server queue with:
#   Poisson arrivals at rate  lambda = sessions arriving per slot
#   Exponential service rate  mu     = capacity (max sessions VMs can serve)
#   Server utilisation        rho    = lambda / mu  (must be < 1 for stability)
#
# Mean sojourn time (wait in queue + service time):
#   W = S / (1 - rho)   where S = 1/mu = BASE_SERVICE_MS (mean service time)
#
# This is the textbook M/M/1 result. The old formula used an empirical
# damping constant k=0.72 which has no theoretical grounding. This version
# applies the formula directly with no adjustments.
# rho is clamped to 0.99 to keep the model in its stable regime (rho < 1).
# This is a documented modelling boundary, not a physical claim.
#
# Reference: Kleinrock, L. (1975). Queueing Systems Vol. 1: Theory.
#            Wiley-Interscience. (M/M/1 mean sojourn time, Ch. 2)
#            Gross, D. & Harris, C.M. (2008). Fundamentals of Queueing
#            Theory, 4th ed. Wiley. (M/M/1 derivation, Ch. 3)
# ---------------------------------------------------------------------------
BASE_SERVICE_MS = 120   # S = mean service time per request in milliseconds

def response_time(sessions, capacity):
    rho = sessions / max(capacity, 1)    # utilisation: rho = lambda / mu
    rho = min(rho, 0.99)                 # clamp to stable regime (rho < 1)
    W   = BASE_SERVICE_MS / (1.0 - rho) # M/M/1 mean sojourn time: W = S/(1-rho)
    return int(round(W))


# ---------------------------------------------------------------------------
# Setup latency — cold-start provisioning model
# L = L_base + max(ΔVMS, 0) × L_per_vm  (scale-down incurs no latency)
# Reference: Mohan et al., "Agile Cold Starts for Scalable Serverless",
#            USENIX HotCloud (2019)
# ---------------------------------------------------------------------------
def setup_latency(vms, vms_prev):
    return int(round(80 + max(vms - vms_prev, 0) * 420))


def queue_wait(sessions, capacity):
    if sessions <= capacity:
        return 0
    return int(round(((sessions - capacity) / max(capacity, 1)) * 30))



@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    week_total = max(50, min(int(data.get("week_total", 420)), 900))

    # Fix #9: clamp provisioning delay so it can never exceed half a week
    provisioning_delay = max(0, min(int(data.get("delay", 4)), DELAY_MAX))

    # Fix #7: clamp buffer so it cannot silently under-provision
    buffer = max(BUFFER_MIN, min(float(data.get("buffer", DEFAULT_BUFFER)), BUFFER_MAX))

    preds = linear_regression_predict_week(week_total)
    slots = [(d, t) for d in DAYS for t in HOURS]

    # ---------------------------------------------------------------------------
    # Fix #4: uncertainty σ estimated from test-set residuals rather than a
    # 5-slot rolling window. The rolling window gave σ≈0 for the first 4 slots
    # (Monday morning — the highest demand period) and was far too narrow to
    # capture day-level forecast variance.
    #
    # We use the model's held-out RMSE (in normalised units, rescaled to
    # absolute sessions) as a stable, data-driven σ for every slot.
    # This is equivalent to assuming forecast errors are i.i.d. with the
    # empirical test-set standard deviation — a standard assumption in
    # prediction-interval construction.
    # Reference: Hyndman & Athanasopoulos, Forecasting: P&P (2021), Ch. 5.5
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Fix #5: actuals are deterministic given the same week_total.
    # Seed with a stable hash of week_total so results are reproducible across
    # runs but still vary meaningfully between different week_total inputs.
    # ---------------------------------------------------------------------------
    seed = int(hashlib.md5(str(week_total).encode()).hexdigest(), 16) % (2**31)
    rng  = np.random.default_rng(seed)

    # ---------------------------------------------------------------------------
    # Actuals generation with structural spike scenario
    # Base demand: formula × uniform noise (±20%)
    # Structural spike: 5% probability — 2–3.5× surge modelling flu outbreaks,
    # post-holiday rushes, or incidents the smooth model cannot anticipate.
    # Reference: Rostami-Tabar & Syntetos, "Demand forecasting in healthcare",
    #            Int. J. Forecasting (2022)
    # ---------------------------------------------------------------------------
    actuals = []
    for (d, t) in slots:
        base_demand = formula_pred(week_total, d, t) * rng.uniform(0.82, 1.20)
        if rng.random() < 0.05:
            base_demand *= rng.uniform(2.0, 3.5)   # structural spike
        actuals.append(max(int(round(base_demand)), 0))

    # ---------------------------------------------------------------------------
    # Fix #6: compute shared quantities once outside the strategy loop.
    # stdevs is no longer used (replaced by Fix #4 sigma_absolute), but
    # actuals, slots and flat_cap are all constant across strategies.
    # ---------------------------------------------------------------------------
    flat_cap = FLAT_VMS * VM_CPU // CPU_SES

    results = {}
    # ---------------------------------------------------------------------------
    # Safety buffer: Z x sigma  (used by delay_uncertainty_aware)
    # sigma_absolute: forecast error std in session units (MODEL_RMSE rescaled)
    # Z_SCORE = norm.ppf(0.95) = 1.645 => 95% one-sided coverage
    # Reference: Silver, Pyke & Peterson (1998), Ch. 7  SS = z * sigma_L
    # ---------------------------------------------------------------------------
    from scipy.stats import norm as _norm
    Z_SCORE        = _norm.ppf(0.95)               # 1.645 for 95th percentile
    sigma_absolute = MODEL_RMSE * week_total        # sigma in session units
    safety_margin  = Z_SCORE * sigma_absolute       # Z x sigma safety buffer

    for strategy in ["delay_unaware", "delay_aware", "delay_uncertainty_aware"]:
        out      = []
        prev_vms = FLAT_VMS

        for idx, (day, ts) in enumerate(slots):

            if strategy == "delay_unaware":
                idx_pred = idx
                cap      = allocate(preds[idx_pred], buffer)

            elif strategy == "delay_aware":
                # Provisioning committed `delay` slots ago — use that old pred.
                # Reference: Liu et al., "Automated and Scalable QoS Control
                #            for Network Virtualization", USENIX (2017)
                idx_pred = max(idx - provisioning_delay, 0)
                cap      = allocate(preds[idx_pred], buffer)

            elif strategy == "delay_uncertainty_aware":
                # Past-committed prediction + Z x sigma safety buffer.
                # Most conservative: stale prediction AND uncertainty hedge.
                # Safety buffer = Z_SCORE x sigma_absolute (Z x sigma formula)
                # where Z = norm.ppf(0.95) = 1.645 for 95% one-sided coverage.
                # Reference: Silver, Pyke & Peterson (1998), Ch. 7
                idx_pred = max(idx - provisioning_delay, 0)
                cap      = allocate(preds[idx_pred] + safety_margin, buffer)


            actual = actuals[idx]
            rt     = response_time(actual, cap["buffered"])
            ssl    = setup_latency(cap["vms"], prev_vms)
            qw     = queue_wait(actual, cap["buffered"])

            out.append({
                "day": day, "time": ts,
                "predicted": round(preds[idx_pred], 2),
                "actual":    actual,
                "buffered":  cap["buffered"], "vms": cap["vms"],
                "cpu":       cap["cpu"],      "ram": cap["ram"],
                "rt_ms":     rt, "ssl_ms": ssl, "qw_sec": qw,
                "sla_ok":    actual <= cap["buffered"],
                "flat_rt":   response_time(actual, flat_cap),
            })
            prev_vms = cap["vms"]

        # --- Core technical metrics ---
        n        = len(out)
        sla      = round(sum(1 for r in out if r["sla_ok"]) / n * 100, 2)
        avg_rt   = round(sum(r["rt_ms"]  for r in out) / n, 1)
        avg_ssl  = round(sum(r["ssl_ms"] for r in out) / n, 1)
        over     = [r for r in out if r["qw_sec"] > 0]
        avg_qw   = round(sum(r["qw_sec"] for r in over) / len(over), 1) if over else 0
        ml_vms   = sum(r["vms"] for r in out)
        fl_vms   = n * FLAT_VMS
        saving   = round((fl_vms - ml_vms) / fl_vms * 100, 1)
        flat_rt  = round(sum(r["flat_rt"] for r in out) / n, 1)
        flat_sla = round(sum(1 for r in out if r["actual"] <= flat_cap) / n * 100, 2)

        # --- Hospital / clinical metrics ---
        total_patients    = sum(r["actual"]  for r in out)
        patients_served   = sum(min(r["actual"], r["buffered"]) for r in out)
        patients_unserved = total_patients - patients_served
        throughput_pct    = round(patients_served / max(total_patients, 1) * 100, 1)
        overload_count    = sum(1 for r in out if r["actual"] > r["buffered"])
        overload_pct      = round(overload_count / n * 100, 1)
        peak              = max(out, key=lambda r: r["actual"])
        peak_label        = f"{peak['day']} {peak['time']}"
        avg_headroom      = round(sum(r["buffered"] - r["actual"] for r in out) / n, 1)

        # Average Utilisation = average of (Actual Demand / Allocated Capacity) x 100
        # Formula from doc: Average Utilisation = average of (Actual Demand / Allocated Capacity) x 100
        avg_utilisation   = round(
            sum(r["actual"] / max(r["buffered"], 1) for r in out) / n * 100, 1
        )

        # Per-day breakdown
        day_stats = {}
        for d in DAYS:
            ds  = [r for r in out if r["day"] == d]
            d_n = len(ds)
            d_peak = max(ds, key=lambda r: r["actual"])
            day_stats[d] = {
                "total_patients":  sum(r["actual"] for r in ds),
                "patients_served": sum(min(r["actual"], r["buffered"]) for r in ds),
                "overload_slots":  sum(1 for r in ds if r["actual"] > r["buffered"]),
                "sla_pct":         round(sum(1 for r in ds if r["sla_ok"]) / d_n * 100, 1),
                "peak_time":       d_peak["time"],
                "peak_demand":     d_peak["actual"],
                "avg_headroom":    round(sum(r["buffered"] - r["actual"] for r in ds) / d_n, 1),
            }

        # Per-time-slot pattern (averaged across all days)
        hour_stats = {}
        for ts in HOURS:
            hs  = [r for r in out if r["time"] == ts]
            h_n = len(hs)
            hour_stats[ts] = {
                "avg_demand":    round(sum(r["actual"]   for r in hs) / h_n, 1),
                "avg_capacity":  round(sum(r["buffered"] for r in hs) / h_n, 1),
                "overload_days": sum(1 for r in hs if r["actual"] > r["buffered"]),
            }

        # Infrastructure cost proxy (VM-hours × spot/on-demand rate)
        USD_TO_AUD       = 1.55                          # approximate USD→AUD exchange rate
        VM_COST_PER_HOUR = 0.096 * USD_TO_AUD             # AUD — ~t3.large equivalent
        SLOT_HOURS       = 0.5
        total_vm_hours   = sum(r["vms"] for r in out) * SLOT_HOURS
        flat_vm_hours    = fl_vms * SLOT_HOURS
        infra_cost_aud   = round(total_vm_hours * VM_COST_PER_HOUR, 2)
        flat_cost_aud    = round(flat_vm_hours  * VM_COST_PER_HOUR, 2)
        cost_saving_aud  = round(flat_cost_aud  - infra_cost_aud, 2)
        cost_per_patient = round(infra_cost_aud  / max(patients_served, 1), 4)

        # Slot-level MAE and RMSE — measure per-slot prediction accuracy
        # MAE  = average of |Actual Demand - Predicted Demand|
        # RMSE = sqrt( average of (Actual Demand - Predicted Demand)^2 )
        # These use the per-slot predicted vs actual values (not the model's
        # held-out training error), so they reflect live prediction quality.
        slot_mae  = round(
            sum(abs(r["actual"] - r["predicted"]) for r in out) / n, 2
        )
        slot_rmse = round(
            float(np.sqrt(sum((r["actual"] - r["predicted"])**2 for r in out) / n)), 2
        )

        if sla >= 98 and overload_pct < 5:
            status_label = "Excellent – system well within capacity"
        elif sla >= 92 and overload_pct < 15:
            status_label = "Good – minor overload in some slots"
        elif sla >= 80:
            status_label = "Fair – moderate overload, review peak slots"
        else:
            status_label = "At Risk – frequent overload, consider increasing resources"

        results[strategy] = {
            "slots":      out,
            "day_stats":  day_stats,
            "hour_stats": hour_stats,
            "summary": {
                # --- Model Quality ---
                # MAE  = average |Actual - Predicted| per slot
                # RMSE = sqrt(average (Actual - Predicted)^2) per slot
                "mae_per_slot":    slot_mae,
                "rmse_per_slot":   slot_rmse,
                "forecast_mae":    MODEL_MAE,
                "forecast_rmse":   MODEL_RMSE,
                "ridge_alpha":     regressor.alpha_,
                "sigma_sessions":  round(sigma_absolute, 2),
                # Technical SLA / infra
                "sla_pct":         sla,
                "avg_rt_ms":       avg_rt,
                "avg_ssl_ms":      avg_ssl,
                "avg_qw_sec":      avg_qw,
                "vm_saving_pct":   saving,
                "ml_vms_total":    ml_vms,
                "flat_vms_total":  fl_vms,
                "flat_avg_rt_ms":  flat_rt,
                "flat_sla_pct":    flat_sla,
                "model_used":      strategy,
                "buffer_used":     buffer,
                "week_total":      week_total,
                # Clinical
                "total_patients":    total_patients,
                "patients_served":   patients_served,
                "patients_unserved": patients_unserved,
                "throughput_pct":    throughput_pct,
                "overload_slots":    overload_count,
                "overload_pct":      overload_pct,
                "avg_headroom":      avg_headroom,
                "peak_slot":         peak_label,
                "peak_demand":       peak["actual"],
                "infra_cost_aud":    infra_cost_aud,
                "flat_cost_aud":     flat_cost_aud,
                "cost_saving_aud":   cost_saving_aud,
                "cost_per_patient":  cost_per_patient,
                # --- Additional Metrics ---
                # Total VM-Hours = sum(VMs per slot) x slot duration (0.5 hr)
                "total_vm_hours":    round(total_vm_hours, 2),
                # Average Utilisation = average of (Actual / Capacity) x 100
                "avg_utilisation_pct": avg_utilisation,
                "status_label":      status_label,
            }
        }

    return jsonify(results)



if __name__ == "__main__":
    app.run(debug=True, port=8080)