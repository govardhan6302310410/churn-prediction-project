"""
helpers.py — Analytics helper functions for 15 advanced ChurnAI features.
This module contains pure functions that query the DB / models and return
data ready for rendering in templates.  Nothing in app.py is imported here
except the shared DB path.
"""

import sqlite3
import os
import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime

DB_PATH = "history.db"
MODEL_PATH = "Models/model.pkl"
DATA_PATH = "uploads/data.csv"

FEATURES = [
    'International plan',
    'Voice mail plan',
    'Number vmail messages',
    'Total day minutes',
    'Total eve minutes',
    'Total night minutes',
    'Customer service calls'
]


# ═══════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════

def _get_conn():
    return sqlite3.connect(DB_PATH)


def _all_predictions():
    """Return all predictions as list of dicts."""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════
# F1 — Customer Segmentation
# ═══════════════════════════════════════════════════

def get_segmentation_data():
    rows = _all_predictions()
    segments = {"Low Risk": [], "Medium Risk": [], "High Risk": [], "VIP": []}
    for r in rows:
        p = r.get("probability", 0) or 0
        day = r.get("day_minutes", 0) or 0
        loyalty = _quick_loyalty(r)
        if p < 30 and loyalty > 60:
            segments["VIP"].append(r)
        elif p >= 70:
            segments["High Risk"].append(r)
        elif p >= 49:
            segments["Medium Risk"].append(r)
        else:
            segments["Low Risk"].append(r)
    summary = {k: len(v) for k, v in segments.items()}
    return segments, summary


def _quick_loyalty(r):
    day = r.get("day_minutes", 0) or 0
    eve = r.get("eve_minutes", 0) or 0
    night = r.get("night_minutes", 0) or 0
    vm = r.get("voicemail", 0) or 0
    intl = r.get("international", 0) or 0
    sc = r.get("service_calls", 0) or 0
    score = min(day / 4, 25) + min(eve / 5, 20) + min(night / 6, 20)
    if vm == 1:
        score += 10
    if intl == 1:
        score += 10
    score -= sc * 5
    return max(0, min(100, round(score, 2)))


# ═══════════════════════════════════════════════════
# F2 — Churn Risk Heatmap
# ═══════════════════════════════════════════════════

def get_heatmap_data():
    rows = _all_predictions()
    if not rows:
        return [], [], []

    day_bins = ["0-100", "100-200", "200-300", "300+"]
    service_bins = ["0-1", "2-3", "4-5", "6+"]

    grid = [[[] for _ in range(len(service_bins))] for _ in range(len(day_bins))]

    for r in rows:
        day = r.get("day_minutes", 0) or 0
        sc = r.get("service_calls", 0) or 0
        prob = r.get("probability", 0) or 0

        di = min(int(day // 100), 3)
        si = 0 if sc <= 1 else (1 if sc <= 3 else (2 if sc <= 5 else 3))
        grid[di][si].append(prob)

    heatmap = []
    for i, row in enumerate(grid):
        heatmap_row = []
        for cell in row:
            heatmap_row.append(round(sum(cell) / len(cell), 1) if cell else 0)
        heatmap.append(heatmap_row)

    return heatmap, day_bins, service_bins


# ═══════════════════════════════════════════════════
# F3 — Model Performance Comparison
# ═══════════════════════════════════════════════════

def get_model_metrics():
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)
    for col in ['International plan', 'Voice mail plan', 'Churn']:
        df[col] = df[col].astype(str).str.strip().str.lower()
    mapping = {'yes': 1, 'no': 0, '1': 1, '0': 0}
    df['International plan'] = df['International plan'].map(mapping)
    df['Voice mail plan'] = df['Voice mail plan'].map(mapping)
    df['Churn'] = df['Churn'].map(mapping)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    if df.empty:
        return None

    X = df[FEATURES]
    y = df['Churn'].astype(int)

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    svm = SVC(probability=True, class_weight="balanced")
    svm.fit(X, y)
    svm_pred = svm.predict(X)

    rf = RandomForestClassifier()
    rf.fit(X, y)
    rf_pred = rf.predict(X)

    metrics = {
        "svm": {
            "accuracy": round(accuracy_score(y, svm_pred) * 100, 2),
            "precision": round(precision_score(y, svm_pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(y, svm_pred, zero_division=0) * 100, 2),
            "f1": round(f1_score(y, svm_pred, zero_division=0) * 100, 2),
        },
        "rf": {
            "accuracy": round(accuracy_score(y, rf_pred) * 100, 2),
            "precision": round(precision_score(y, rf_pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(y, rf_pred, zero_division=0) * 100, 2),
            "f1": round(f1_score(y, rf_pred, zero_division=0) * 100, 2),
        },
        "rf_model": rf
    }
    return metrics


# ═══════════════════════════════════════════════════
# F4 — Feature Importance
# ═══════════════════════════════════════════════════

def get_feature_importance():
    metrics = get_model_metrics()
    if not metrics:
        return None, None
    rf = metrics["rf_model"]
    importances = rf.feature_importances_
    paired = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in paired]
    values = [round(p[1] * 100, 2) for p in paired]
    return labels, values


# ═══════════════════════════════════════════════════
# F5 — Risk Timeline
# ═══════════════════════════════════════════════════

def get_risk_timeline():
    conn = _get_conn()
    rows = conn.execute("""
        SELECT substr(date,1,10) as d, 
               AVG(probability) as avg_prob,
               COUNT(*) as cnt
        FROM predictions GROUP BY d ORDER BY d
    """).fetchall()
    conn.close()
    dates = [r[0] for r in rows if r[0]]
    avg_probs = [round(r[1], 1) if r[1] is not None else 0 for r in rows if r[0]]
    counts = [r[2] if r[2] is not None else 0 for r in rows if r[0]]
    return dates, avg_probs, counts


# ═══════════════════════════════════════════════════
# F6 — Prediction Simulation
# ═══════════════════════════════════════════════════

def simulate_churn(international, voicemail, vmail, day, eve, night, service):
    if not os.path.exists(MODEL_PATH):
        return None
    model = pickle.load(open(MODEL_PATH, 'rb'))
    values = [international, voicemail, vmail, day, eve, night, service]
    input_df = pd.DataFrame([values], columns=FEATURES)
    proba = model.predict_proba(input_df)[0][1] * 100

    risk_boost = 0
    if day < 100: risk_boost += 8
    elif day < 200: risk_boost += 4
    else: risk_boost -= 3

    if eve < 100: risk_boost += 6
    elif eve < 200: risk_boost += 3
    else: risk_boost -= 2

    if night < 100: risk_boost += 5
    elif night < 200: risk_boost += 2
    else: risk_boost -= 2

    if voicemail == 0: risk_boost += 10
    if international == 0: risk_boost += 6
    if service >= 4: risk_boost += 8

    final = max(0, min(100, proba + risk_boost))
    return round(final, 2)


# ═══════════════════════════════════════════════════
# F7 — AI Explanation
# ═══════════════════════════════════════════════════

def get_ai_explanation(record):
    if not record:
        return [], []

    prob = record.get("probability", 0) or 0
    day = record.get("day_minutes", 0) or 0
    eve = record.get("eve_minutes", 0) or 0
    night = record.get("night_minutes", 0) or 0
    sc = record.get("service_calls", 0) or 0
    vm = record.get("voicemail", 0) or 0
    intl = record.get("international", 0) or 0

    reasons = []
    factors = []

    if sc >= 4:
        reasons.append("High number of customer service calls indicates dissatisfaction")
        factors.append(("Service Calls", min(sc * 12, 100)))
    if day < 100:
        reasons.append("Very low daytime usage suggests disengagement")
        factors.append(("Low Day Usage", 70))
    elif day > 300:
        reasons.append("Extremely high day usage may lead to bill shock")
        factors.append(("High Day Usage", 60))
    if vm == 0:
        reasons.append("No voicemail plan — missing engagement feature")
        factors.append(("No Voicemail", 50))
    if intl == 0:
        reasons.append("No international plan — limited service adoption")
        factors.append(("No Intl Plan", 40))
    if night < 100:
        reasons.append("Low night usage indicates reduced overall engagement")
        factors.append(("Low Night Usage", 45))
    if eve < 100:
        reasons.append("Low evening usage signals declining interest")
        factors.append(("Low Eve Usage", 40))
    if prob >= 70:
        reasons.append("Overall churn probability exceeds high-risk threshold")
        factors.append(("High Risk Score", 90))
    elif prob >= 49:
        reasons.append("Churn probability is in the medium-risk zone")
        factors.append(("Medium Risk", 60))

    if not reasons:
        reasons.append("Customer shows healthy engagement patterns")
        factors.append(("Good Health", 20))

    factors.sort(key=lambda x: x[1], reverse=True)
    return reasons, factors


# ═══════════════════════════════════════════════════
# F8 — Retention Strategy
# ═══════════════════════════════════════════════════

def get_retention_strategy(record):
    if not record:
        return None

    prob = record.get("probability", 0) or 0
    sc = record.get("service_calls", 0) or 0
    day = record.get("day_minutes", 0) or 0

    if prob >= 70:
        urgency = "CRITICAL"
        color = "red"
        actions = [
            {"icon": "🚨", "title": "Immediate Outreach", "desc": "Call customer within 24 hours with a dedicated account manager", "timeline": "Day 1"},
            {"icon": "💰", "title": "Emergency Discount", "desc": "Offer 30% discount on next 3 months billing", "timeline": "Day 1-2"},
            {"icon": "🎁", "title": "Premium Bundle", "desc": "Upgrade to premium plan at current rate with added OTT services", "timeline": "Week 1"},
            {"icon": "📞", "title": "Service Recovery", "desc": "Assign priority support queue and resolve all pending complaints", "timeline": "Week 1"},
            {"icon": "📊", "title": "Monthly Check-in", "desc": "Schedule monthly satisfaction surveys and proactive engagement", "timeline": "Ongoing"},
        ]
    elif prob >= 49:
        urgency = "HIGH"
        color = "orange"
        actions = [
            {"icon": "📧", "title": "Personalized Email", "desc": "Send targeted email highlighting value-added services", "timeline": "Day 1-3"},
            {"icon": "💸", "title": "Loyalty Discount", "desc": "Offer 15% discount or bonus data pack", "timeline": "Week 1"},
            {"icon": "📱", "title": "App Engagement", "desc": "Push in-app notifications with personalized recommendations", "timeline": "Week 1-2"},
            {"icon": "🎯", "title": "Usage Incentive", "desc": "Reward increased usage with bonus minutes or cashback", "timeline": "Week 2"},
        ]
    else:
        urgency = "MODERATE"
        color = "green"
        actions = [
            {"icon": "🏆", "title": "Loyalty Reward", "desc": "Send thank-you message with a small loyalty bonus", "timeline": "Week 1"},
            {"icon": "📊", "title": "Quarterly Review", "desc": "Schedule quarterly satisfaction check-in", "timeline": "Quarterly"},
            {"icon": "🎁", "title": "Referral Program", "desc": "Invite to referral program with mutual benefits", "timeline": "Month 1"},
        ]

    return {"urgency": urgency, "color": color, "actions": actions, "probability": prob}


# ═══════════════════════════════════════════════════
# F9 — Business Impact Analyzer
# ═══════════════════════════════════════════════════

def get_business_impact():
    rows = _all_predictions()
    if not rows:
        return None

    total = len(rows)
    churn_rows = [r for r in rows if (r.get("result", "") or "").lower() == "churn"]
    no_churn_rows = [r for r in rows if (r.get("result", "") or "").lower() != "churn"]

    probs = [r.get("probability", 0) or 0 for r in rows]
    avg_prob = round(sum(probs) / len(probs), 1) if probs else 0

    # Estimate revenue at risk (assume avg monthly bill ~500 if not stored)
    avg_bill = 500
    total_revenue_risk = round(len(churn_rows) * avg_bill * 12, 2)
    retention_cost = round(total_revenue_risk * 0.15, 2)
    net_savings = round(total_revenue_risk - retention_cost, 2)

    high_risk = len([r for r in rows if (r.get("probability", 0) or 0) >= 70])
    medium_risk = len([r for r in rows if 49 <= (r.get("probability", 0) or 0) < 70])
    low_risk = total - high_risk - medium_risk

    return {
        "total": total,
        "churn_count": len(churn_rows),
        "no_churn_count": len(no_churn_rows),
        "avg_probability": avg_prob,
        "total_revenue_risk": total_revenue_risk,
        "retention_cost": retention_cost,
        "net_savings": net_savings,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
    }


# ═══════════════════════════════════════════════════
# F10 — Customer Persona
# ═══════════════════════════════════════════════════

def get_persona_data():
    rows = _all_predictions()
    personas = {
        "High Value": [],
        "Loyal Customer": [],
        "At Risk": [],
        "Discount Seeker": [],
        "New User": []
    }
    for r in rows:
        p = r.get("probability", 0) or 0
        day = r.get("day_minutes", 0) or 0
        loyalty = _quick_loyalty(r)
        sc = r.get("service_calls", 0) or 0

        if day > 250 and loyalty > 50 and p < 40:
            personas["High Value"].append(r)
        elif loyalty > 60 and p < 50:
            personas["Loyal Customer"].append(r)
        elif p >= 60:
            personas["At Risk"].append(r)
        elif sc >= 3 and p >= 40:
            personas["Discount Seeker"].append(r)
        else:
            personas["New User"].append(r)

    summary = {k: len(v) for k, v in personas.items()}
    return personas, summary


# ═══════════════════════════════════════════════════
# F11 — Risk Alerts
# ═══════════════════════════════════════════════════

def get_high_risk_alerts():
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions WHERE probability >= 70 ORDER BY probability DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════
# F12 — Dataset Insights
# ═══════════════════════════════════════════════════

def _safe(val):
    """Convert NaN/inf/None to 0 so Jinja2 renders valid JS numbers."""
    if val is None:
        return 0
    try:
        if math.isnan(val) or math.isinf(val):
            return 0
    except (TypeError, ValueError):
        pass
    return val


def get_dataset_insights():
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)
    for col in ['International plan', 'Voice mail plan', 'Churn']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    mapping = {'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0}
    for col in ['International plan', 'Voice mail plan', 'Churn']:
        if col in df.columns:
            df[col] = df[col].map(mapping)
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=FEATURES + ['Churn'], inplace=True)

    if df.empty:
        return None

    stats = {}
    for f in FEATURES:
        if f in df.columns:
            stats[f] = {
                "mean": _safe(round(float(df[f].mean()), 2)),
                "median": _safe(round(float(df[f].median()), 2)),
                "std": _safe(round(float(df[f].std()), 2)),
                "min": _safe(round(float(df[f].min()), 2)),
                "max": _safe(round(float(df[f].max()), 2)),
            }

    churn_rate = _safe(round(float(df['Churn'].mean()) * 100, 2)) if 'Churn' in df.columns else 0
    total_rows = len(df)

    # Churn vs feature means
    churn_means = {}
    no_churn_means = {}
    for f in FEATURES:
        if f in df.columns:
            cm = df[df['Churn'] == 1][f].mean()
            ncm = df[df['Churn'] == 0][f].mean()
            churn_means[f] = _safe(round(float(cm), 2)) if not pd.isna(cm) else 0
            no_churn_means[f] = _safe(round(float(ncm), 2)) if not pd.isna(ncm) else 0

    return {
        "stats": stats,
        "churn_rate": churn_rate,
        "total_rows": total_rows,
        "features": FEATURES,
        "churn_means": churn_means,
        "no_churn_means": no_churn_means,
    }


# ═══════════════════════════════════════════════════
# F13 — Confidence Indicator
# ═══════════════════════════════════════════════════

def get_confidence_data():
    rows = _all_predictions()
    results = []
    for r in rows:
        p = r.get("probability", 0) or 0
        distance = abs(p - 50)
        if distance >= 30:
            conf_label = "Very High"
            conf_pct = min(95, 70 + distance)
            conf_color = "#06d6a0"
        elif distance >= 15:
            conf_label = "High"
            conf_pct = 60 + distance
            conf_color = "#4cc9f0"
        elif distance >= 5:
            conf_label = "Moderate"
            conf_pct = 40 + distance
            conf_color = "#fb8500"
        else:
            conf_label = "Low"
            conf_pct = 20 + distance * 2
            conf_color = "#ef233c"
        results.append({
            **r,
            "conf_label": conf_label,
            "conf_pct": round(conf_pct, 1),
            "conf_color": conf_color
        })
    return results


# ═══════════════════════════════════════════════════
# F14 — Interactive Charts
# ═══════════════════════════════════════════════════

def get_interactive_chart_data():
    rows = _all_predictions()
    if not rows:
        return None

    # Risk distribution
    risk_dist = {"High": 0, "Medium": 0, "Low": 0}
    for r in rows:
        p = r.get("probability", 0) or 0
        if p >= 70:
            risk_dist["High"] += 1
        elif p >= 49:
            risk_dist["Medium"] += 1
        else:
            risk_dist["Low"] += 1

    # Revenue impact buckets
    revenue_buckets = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}
    for r in rows:
        p = r.get("probability", 0) or 0
        if p < 25:
            revenue_buckets["0-25%"] += 1
        elif p < 50:
            revenue_buckets["25-50%"] += 1
        elif p < 75:
            revenue_buckets["50-75%"] += 1
        else:
            revenue_buckets["75-100%"] += 1

    # Behavior patterns: avg day/eve/night for churn vs no-churn
    churn_usage = {"day": [], "eve": [], "night": []}
    nochurn_usage = {"day": [], "eve": [], "night": []}
    for r in rows:
        res = (r.get("result", "") or "").lower()
        target = churn_usage if res == "churn" else nochurn_usage
        target["day"].append(r.get("day_minutes", 0) or 0)
        target["eve"].append(r.get("eve_minutes", 0) or 0)
        target["night"].append(r.get("night_minutes", 0) or 0)

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    behavior = {
        "churn": {"day": avg(churn_usage["day"]), "eve": avg(churn_usage["eve"]), "night": avg(churn_usage["night"])},
        "nochurn": {"day": avg(nochurn_usage["day"]), "eve": avg(nochurn_usage["eve"]), "night": avg(nochurn_usage["night"])}
    }

    return {
        "risk_dist": risk_dist,
        "revenue_buckets": revenue_buckets,
        "behavior": behavior,
        "total": len(rows)
    }


# ═══════════════════════════════════════════════════
# F15 — System Activity Monitor
# ═══════════════════════════════════════════════════

def log_activity(action):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO system_activity (action, timestamp) VALUES (?, ?)",
        (action, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_activity_stats():
    conn = _get_conn()

    # prediction count
    pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    # activity log
    try:
        activities = conn.execute(
            "SELECT * FROM system_activity ORDER BY id DESC LIMIT 50"
        ).fetchall()
    except Exception:
        activities = []

    # count by action type
    try:
        action_counts = {}
        for row in conn.execute(
            "SELECT action, COUNT(*) FROM system_activity GROUP BY action"
        ).fetchall():
            action_counts[row[0]] = row[1]
    except Exception:
        action_counts = {}

    conn.close()

    return {
        "prediction_count": pred_count,
        "upload_count": action_counts.get("dataset_upload", 0),
        "training_count": action_counts.get("model_training", 0),
        "activities": activities,
        "action_counts": action_counts,
    }
