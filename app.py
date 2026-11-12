"""
DeccanSwift Logistics Pvt. Ltd. (Pune E-Commerce Courier Network)
AI Use-Case Prototype: Proactive SLA Breach & RTO Risk Predictor with Statutory Guardrails
=========================================================================================
Features:
1. Data Cleaning Pipeline: Deduplicates AWB numbers, resolves out-of-order scan sequences
   via state-machine ordering, and standardizes mixed time zones (UTC, GST, EST) to IST.
2. DPDP Act 2023 / DPDP Rules 2025 Privacy Layer: SHA-256 PII masking & data minimization.
3. Calibrated Gradient-Boosted Risk Model + Explainable Reason Codes.
4. Guardrails & Fallback Rules:
   - GST E-Way Bill Rule 138 Guardrail (Inter-state > Rs 50,000 / MH Intra-state > Rs 1,00,000)
   - Telemetry Completeness Fallback Rule (degraded scanner lag > 30 mins)
5. 60-Case Realistic Benchmark Runner & Interactive Streamlit Operations Dashboard.

Usage:
  Run CLI benchmark:  python deccanswift_sla_rto_prototype.py
  Run Streamlit app:  streamlit run deccanswift_sla_rto_prototype.py
"""

import hashlib
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------------------------------------------------------
# 1. CONSTANTS & PUNE 12-HUB NETWORK CONFIGURATION
# ---------------------------------------------------------------------------
PUNE_HUBS = [
    "Hinjawadi-H1", "Hadapsar-H2", "Chakan-H3", "Wagholi-H4",
    "Pimpri-H5", "Katraj-H6", "Baner-H7", "Kothrud-H8",
    "VimanNagar-H9", "Talegaon-H10", "Ranjangaon-H11", "Baramati-H12"
]

SCAN_STATE_ORDER = {
    "PICKED_UP": 1,
    "ORIGIN_HUB_IN": 2,
    "BAGGED_LINEHAUL": 3,
    "DEST_HUB_IN": 4,
    "OUT_FOR_DELIVERY": 5,
    "DELIVERED_OR_NDR": 6
}

TZ_OFFSETS_HOURS = {
    "IST": 5.5,
    "UTC": 0.0,
    "GST": 4.0,   # Gulf Standard Time misconfigured on legacy hub scanner
    "EST": -5.0   # US Eastern misconfigured on cloud gateway
}

IST_TZ = timezone(timedelta(hours=5, minutes=30))
COST_PER_INFERENCE_INR = 0.0042  # Rs 0.0042 per request on cloud CPU container


# ---------------------------------------------------------------------------
# 2. DPDP ACT 2023 PRIVACY & MULTI-HUB DATA CLEANING PIPELINE
# ---------------------------------------------------------------------------
def anonymize_pii_dpdp(raw_value: str, salt: str = "DECCANSWIFT_DPDP_2026") -> str:
    """Hashes customer names/phones per DPDP Act 2023 data minimization & Rule 6 safeguards."""
    return hashlib.sha256(f"{salt}:{raw_value}".encode("utf-8")).hexdigest()[:16]


def normalize_timestamp_to_ist(ts_str: str, tz_label: str) -> datetime:
    """Converts mixed hub server timestamps (IST, UTC, GST, EST) into canonical IST."""
    dt_naive = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    offset = TZ_OFFSETS_HOURS.get(tz_label, 5.5)
    src_tz = timezone(timedelta(hours=offset))
    dt_aware = dt_naive.replace(tzinfo=src_tz)
    return dt_aware.astimezone(IST_TZ)


def clean_scan_records(df_scans: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw multi-hub scan logs:
    1. Converts all timestamps to canonical Asia/Kolkata (IST).
    2. Removes duplicate (awb_number, event_type, hub_id) entries keeping earliest valid scan.
    3. Re-orders out-of-sequence scan events using logical state-machine rank.
    """
    df = df_scans.copy()
    df["scan_ts_ist"] = [
        normalize_timestamp_to_ist(ts, tz)
        for ts, tz in zip(df["raw_timestamp"], df["server_tz"])
    ]
    df["state_rank"] = df["event_type"].map(SCAN_STATE_ORDER).fillna(99)

    df = df.sort_values(["awb_number", "state_rank", "scan_ts_ist"])
    df = df.drop_duplicates(subset=["awb_number", "event_type", "hub_id"], keep="first")
    df["repaired_ts_ist"] = df.groupby("awb_number")["scan_ts_ist"].cummax()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. SYNTHETIC TRAINING DATA & CALIBRATED ML MODEL
# ---------------------------------------------------------------------------
def build_training_dataset(n_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hub_congestion = rng.uniform(0.2, 0.98, n_samples)
    dwell_hours = rng.exponential(4.5, n_samples) + 0.5
    transit_lag_hours = rng.normal(1.2, 2.4, n_samples)
    is_cod = rng.choice([0, 1], size=n_samples, p=[0.42, 0.58])
    pin_hist_rto = rng.beta(2.0, 14.0, n_samples)
    monsoon_rain_mm = rng.exponential(8.0, n_samples)
    eway_hours_left = rng.uniform(2.0, 36.0, n_samples)
    consignment_inr = rng.choice([850, 2400, 14500, 62000, 125000], size=n_samples, p=[0.45, 0.30, 0.15, 0.06, 0.04])

    logit = (
        -3.1
        + 2.3 * (hub_congestion - 0.5)
        + 0.22 * (dwell_hours - 4.0)
        + 0.35 * np.maximum(0, transit_lag_hours)
        + 0.85 * is_cod
        + 6.5 * pin_hist_rto
        + 0.035 * monsoon_rain_mm
        + 1.4 * (eway_hours_left < 6.0).astype(float)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = rng.binomial(1, np.clip(prob, 0.02, 0.98))

    return pd.DataFrame({
        "hub_congestion": hub_congestion,
        "dwell_hours": dwell_hours,
        "transit_lag_hours": transit_lag_hours,
        "is_cod": is_cod,
        "pin_hist_rto": pin_hist_rto,
        "monsoon_rain_mm": monsoon_rain_mm,
        "eway_hours_left": eway_hours_left,
        "consignment_inr": consignment_inr,
        "target_breach_rto": target
    })


FEATURE_COLS = [
    "hub_congestion", "dwell_hours", "transit_lag_hours",
    "is_cod", "pin_hist_rto", "monsoon_rain_mm",
    "eway_hours_left", "consignment_inr"
]


def train_risk_model():
    df_train = build_training_dataset(n_samples=4000, seed=42)
    base_clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42)
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=3, method="isotonic")
    calibrated_clf.fit(df_train[FEATURE_COLS], df_train["target_breach_rto"])
    return calibrated_clf


# ---------------------------------------------------------------------------
# 4. PREDICTION ENGINE WITH GUARDRAILS & FALLBACK RULES
# ---------------------------------------------------------------------------
def evaluate_shipment(model, record: dict) -> dict:
    """
    Evaluates a single shipment with:
    1. GST E-Way Bill Rule 138 statutory guardrail (always runs first).
    2. Telemetry freshness fallback rule (if scan_lag_mins > 30 or missing_ratio > 0.20).
    3. Calibrated ML probability scoring + explainable drivers.
    """
    t0 = time.perf_counter()

    # Guardrail 1: GST E-Way Bill Compliance (CGST Sec 68 & Rule 138)
    is_interstate = record.get("is_interstate", False)
    val_inr = record.get("consignment_inr", 0)
    eway_left = record.get("eway_hours_left", 24.0)
    eway_required = (is_interstate and val_inr > 50000) or (not is_interstate and val_inr > 100000)

    if eway_required and eway_left < 6.0:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "decision_mode": "GUARDRAIL_EWAY_BILL",
            "risk_probability": 0.96,
            "action": "ESCALATE_EWAY_PART_B_EXTENSION",
            "reason": f"Consignment Rs {val_inr:,.0f} exceeds GST threshold with only {eway_left:.1f}h E-Way validity left (<6h window).",
            "latency_ms": round(latency_ms, 3),
            "cost_inr": COST_PER_INFERENCE_INR
        }

    # Guardrail 2: Fallback Rule when live hub telemetry is delayed or incomplete
    scan_lag_mins = record.get("scan_lag_mins", 5)
    missing_ratio = record.get("missing_feature_ratio", 0.0)
    if scan_lag_mins > 30 or missing_ratio > 0.20:
        fallback_score = min(0.95, 0.35 + 2.2 * record.get("pin_hist_rto", 0.10) + 0.22 * record.get("is_cod", 0))
        action = "HIGH_RISK_INTERVENTION" if fallback_score >= 0.72 else (
            "WATCHLIST_MONITOR" if fallback_score >= 0.45 else "STANDARD_DISPATCH"
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "decision_mode": "FALLBACK_HEURISTIC",
            "risk_probability": round(float(fallback_score), 3),
            "action": action,
            "reason": f"Live scanner lag {scan_lag_mins}m (>30m limit); used PIN RTO & COD historical fallback.",
            "latency_ms": round(latency_ms, 3),
            "cost_inr": COST_PER_INFERENCE_INR
        }

    # Primary Path: Calibrated ML Risk Prediction
    x_df = pd.DataFrame([{col: record[col] for col in FEATURE_COLS}])
    prob = float(model.predict_proba(x_df)[0, 1])

    reasons = []
    if record["hub_congestion"] > 0.75:
        reasons.append(f"High hub congestion ({record['hub_congestion']:.0%})")
    if record["dwell_hours"] > 7.0:
        reasons.append(f"Excess hub dwell ({record['dwell_hours']:.1f}h)")
    if record["is_cod"] == 1 and record["pin_hist_rto"] > 0.16:
        reasons.append(f"High-RTO COD PIN ({record['pin_hist_rto']:.0%})")
    if record["monsoon_rain_mm"] > 18.0:
        reasons.append(f"Heavy Pune rain ({record['monsoon_rain_mm']:.0f}mm)")
    if not reasons:
        reasons.append("Nominal hub dwell & corridor transit")

    if prob >= 0.72:
        action = "HIGH_RISK_INTERVENTION"
    elif prob >= 0.45:
        action = "WATCHLIST_MONITOR"
    else:
        action = "STANDARD_DISPATCH"

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "decision_mode": "CALIBRATED_ML",
        "risk_probability": round(prob, 3),
        "action": action,
        "reason": "; ".join(reasons[:3]),
        "latency_ms": round(latency_ms, 3),
        "cost_inr": COST_PER_INFERENCE_INR
    }


# ---------------------------------------------------------------------------
# 5. 60-CASE REALISTIC BENCHMARK SUITE
# ---------------------------------------------------------------------------
def generate_60_test_cases(seed: int = 101) -> list:
    rng = np.random.default_rng(seed)
    cases = []

    # 1. Normal Pune Intra-City Deliveries (18 cases)
    for i in range(18):
        high = (i % 6 == 0)
        cases.append({
            "case_id": f"TC-NORM-{i+1:02d}",
            "category": "Normal Pune Intra-City",
            "hub_congestion": 0.88 if high else float(rng.uniform(0.25, 0.55)),
            "dwell_hours": 10.5 if high else float(rng.uniform(1.5, 4.2)),
            "transit_lag_hours": 3.5 if high else float(rng.uniform(-0.5, 0.8)),
            "is_cod": 1 if high else int(rng.choice([0, 1])),
            "pin_hist_rto": 0.24 if high else float(rng.uniform(0.04, 0.11)),
            "monsoon_rain_mm": float(rng.uniform(0.0, 5.0)),
            "eway_hours_left": 24.0,
            "consignment_inr": 1499,
            "is_interstate": False,
            "scan_lag_mins": 4,
            "missing_feature_ratio": 0.0,
            "expected_action": "HIGH_RISK_INTERVENTION" if high else "STANDARD_DISPATCH"
        })

    # 2. Monsoon / Chakan-Hinjawadi Traffic Delays (12 cases)
    for i in range(12):
        severe = (i < 8)
        cases.append({
            "case_id": f"TC-RAIN-{i+1:02d}",
            "category": "Monsoon / Traffic Delay",
            "hub_congestion": 0.91 if severe else 0.62,
            "dwell_hours": 9.8 if severe else 5.2,
            "transit_lag_hours": 4.6 if severe else 1.5,
            "is_cod": 1 if severe else 0,
            "pin_hist_rto": 0.21 if severe else 0.10,
            "monsoon_rain_mm": 34.0 if severe else 14.0,
            "eway_hours_left": 18.0,
            "consignment_inr": 2200,
            "is_interstate": False,
            "scan_lag_mins": 6,
            "missing_feature_ratio": 0.0,
            "expected_action": "HIGH_RISK_INTERVENTION" if severe else "WATCHLIST_MONITOR"
        })

    # 3. Out-of-Order Scan Burst Scenarios (10 cases)
    for i in range(10):
        delayed = (i < 6)
        cases.append({
            "case_id": f"TC-SCAN-{i+1:02d}",
            "category": "Out-of-Order Scan Bursts",
            "hub_congestion": 0.86 if delayed else 0.42,
            "dwell_hours": 11.2 if delayed else 2.8,
            "transit_lag_hours": 4.2 if delayed else 0.3,
            "is_cod": 1 if delayed else 0,
            "pin_hist_rto": 0.22 if delayed else 0.07,
            "monsoon_rain_mm": 4.0,
            "eway_hours_left": 20.0,
            "consignment_inr": 3100,
            "is_interstate": False,
            "scan_lag_mins": 8,
            "missing_feature_ratio": 0.0,
            "expected_action": "HIGH_RISK_INTERVENTION" if delayed else "STANDARD_DISPATCH"
        })

    # 4. High-Value GST E-Way Bill Compliance Cases (10 cases)
    for i in range(10):
        expiring = (i < 7)
        cases.append({
            "case_id": f"TC-EWAY-{i+1:02d}",
            "category": "E-Way Bill Compliance",
            "hub_congestion": 0.45,
            "dwell_hours": 3.0,
            "transit_lag_hours": 0.5,
            "is_cod": 0,
            "pin_hist_rto": 0.06,
            "monsoon_rain_mm": 2.0,
            "eway_hours_left": 3.5 if expiring else 16.0,
            "consignment_inr": 78000,
            "is_interstate": True,
            "scan_lag_mins": 5,
            "missing_feature_ratio": 0.0,
            "expected_action": "ESCALATE_EWAY_PART_B_EXTENSION" if expiring else "STANDARD_DISPATCH"
        })

    # 5. Degraded Telemetry Fallback Cases (10 cases)
    for i in range(10):
        high_rto = (i < 5)
        cases.append({
            "case_id": f"TC-FALL-{i+1:02d}",
            "category": "Degraded Fallback",
            "hub_congestion": 0.50,
            "dwell_hours": 4.0,
            "transit_lag_hours": 0.0,
            "is_cod": 1 if high_rto else 0,
            "pin_hist_rto": 0.24 if high_rto else 0.04,
            "monsoon_rain_mm": 0.0,
            "eway_hours_left": 24.0,
            "consignment_inr": 1800,
            "is_interstate": False,
            "scan_lag_mins": 48,  # Triggers >30 min fallback rule
            "missing_feature_ratio": 0.25,
            "expected_action": "HIGH_RISK_INTERVENTION" if high_rto else "STANDARD_DISPATCH"
        })

    return cases


def run_benchmark():
    model = train_risk_model()
    cases = generate_60_test_cases()
    results = []
    for c in cases:
        out = evaluate_shipment(model, c)
        passed = (out["action"] == c["expected_action"])
        results.append({
            "case_id": c["case_id"],
            "category": c["category"],
            "expected": c["expected_action"],
            "predicted": out["action"],
            "prob": out["risk_probability"],
            "mode": out["decision_mode"],
            "latency_ms": out["latency_ms"],
            "cost_inr": out["cost_inr"],
            "passed": passed
        })
    df_res = pd.DataFrame(results)
    print("=== DeccanSwift Logistics AI Prototype: 60-Case Benchmark ===")
    print(f"Overall Accuracy : {df_res['passed'].mean()*100:.1f}% ({df_res['passed'].sum()}/{len(df_res)})")
    print(f"Median Latency   : {df_res['latency_ms'].median():.2f} ms (P95: {np.percentile(df_res['latency_ms'], 95):.2f} ms)")
    print(f"Cost per Request : Rs {df_res['cost_inr'].mean():.4f}")
    print("\nBreakdown by Scenario:")
    print(df_res.groupby("category")["passed"].agg(["count", "sum", lambda s: round(s.mean()*100, 1)]))
    return df_res


if __name__ == "__main__":
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            st.set_page_config(page_title="DeccanSwift Pune AI Risk Console", layout="wide")
            st.title("DeccanSwift Logistics — Pune SLA Breach & RTO Risk Prototype")
            model = train_risk_model()
            col1, col2 = st.columns(2)
            with col1:
                hub = st.selectbox("Active Pune Hub", PUNE_HUBS)
                cong = st.slider("Hub Congestion Index", 0.1, 1.0, 0.78)
                dwell = st.slider("Current Hub Dwell (Hours)", 0.5, 24.0, 8.5)
                rain = st.slider("Pune Corridor Rainfall (mm)", 0.0, 60.0, 12.0)
            with col2:
                cod = st.selectbox("Payment Mode", ["Prepaid", "Cash on Delivery (COD)"])
                pin_rto = st.slider("Destination PIN Historical RTO Rate", 0.02, 0.35, 0.18)
                val_inr = st.number_input("Consignment Value (INR)", 100, 500000, 2500)
                eway_hrs = st.slider("GST E-Way Bill Validity Remaining (Hours)", 1.0, 48.0, 18.0)
                lag_mins = st.slider("Scanner Telemetry Lag (Minutes)", 1, 90, 5)
            sample = {
                "hub_congestion": cong, "dwell_hours": dwell, "transit_lag_hours": 1.8,
                "is_cod": 1 if "COD" in cod else 0, "pin_hist_rto": pin_rto,
                "monsoon_rain_mm": rain, "eway_hours_left": eway_hrs,
                "consignment_inr": val_inr, "is_interstate": val_inr > 50000,
                "scan_lag_mins": lag_mins, "missing_feature_ratio": 0.0
            }
            res = evaluate_shipment(model, sample)
            st.subheader(f"Decision: {res['action']} (Risk: {res['risk_probability']:.1%})")
            st.info(f"Mode: {res['decision_mode']} | Reason: {res['reason']} | Latency: {res['latency_ms']} ms")
        else:
            run_benchmark()
    except ImportError:
        run_benchmark()
