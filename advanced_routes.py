"""
advanced_routes.py — Flask Blueprint for 15 advanced ChurnAI features.
Register this blueprint in app.py with:
    from advanced_routes import advanced_bp
    app.register_blueprint(advanced_bp)
"""

from flask import Blueprint, render_template, request, jsonify
from helpers import (
    get_segmentation_data,
    get_heatmap_data,
    get_model_metrics,
    get_feature_importance,
    get_risk_timeline,
    simulate_churn,
    get_ai_explanation,
    get_retention_strategy,
    get_business_impact,
    get_persona_data,
    get_high_risk_alerts,
    get_dataset_insights,
    get_confidence_data,
    get_interactive_chart_data,
    get_activity_stats,
    log_activity,
    _all_predictions,
)

advanced_bp = Blueprint('advanced', __name__)


# ──────────────────────────────────────────
# F1 — Customer Segmentation Dashboard
# ──────────────────────────────────────────
@advanced_bp.route('/segmentation')
def segmentation():
    segments, summary = get_segmentation_data()
    return render_template('segmentation.html', segments=segments, summary=summary)


# ──────────────────────────────────────────
# F2 — Churn Risk Heatmap
# ──────────────────────────────────────────
@advanced_bp.route('/heatmap')
def heatmap():
    grid, day_bins, service_bins = get_heatmap_data()
    return render_template('heatmap.html', grid=grid, day_bins=day_bins, service_bins=service_bins)


# ──────────────────────────────────────────
# F3 — Model Performance Comparison
# ──────────────────────────────────────────
@advanced_bp.route('/model-comparison')
def model_comparison():
    metrics = get_model_metrics()
    return render_template('model_comparison.html', metrics=metrics)


# ──────────────────────────────────────────
# F4 — Feature Importance
# ──────────────────────────────────────────
@advanced_bp.route('/feature-importance')
def feature_importance():
    labels, values = get_feature_importance()
    return render_template('feature_importance.html', labels=labels, values=values)


# ──────────────────────────────────────────
# F5 — Risk Timeline
# ──────────────────────────────────────────
@advanced_bp.route('/risk-timeline')
def risk_timeline():
    dates, avg_probs, counts = get_risk_timeline()
    return render_template('risk_timeline.html', dates=dates, avg_probs=avg_probs, counts=counts)


# ──────────────────────────────────────────
# F6 — Prediction Simulation
# ──────────────────────────────────────────
@advanced_bp.route('/simulation')
def simulation():
    return render_template('simulation.html')


@advanced_bp.route('/api/simulate', methods=['POST'])
def api_simulate():
    try:
        data = request.get_json()
        prob = simulate_churn(
            int(data.get('international', 0)),
            int(data.get('voicemail', 0)),
            int(data.get('vmail', 0)),
            float(data.get('day', 0)),
            float(data.get('eve', 0)),
            float(data.get('night', 0)),
            int(data.get('service', 0))
        )
        if prob is None:
            return jsonify({"error": "Model not trained yet"}), 400
        risk = "High Risk" if prob >= 70 else ("Medium Risk" if prob >= 49 else "Low Risk")
        return jsonify({"probability": prob, "risk": risk})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────
# F7 — AI Explanation
# ──────────────────────────────────────────
@advanced_bp.route('/ai-explanation')
def ai_explanation():
    pred_id = request.args.get('id', type=int)
    record = None
    reasons = []
    factors = []
    predictions = _all_predictions()

    if pred_id:
        for p in predictions:
            if p.get('id') == pred_id:
                record = p
                break
        if record:
            reasons, factors = get_ai_explanation(record)

    return render_template('ai_explanation.html',
                           predictions=predictions, record=record,
                           reasons=reasons, factors=factors, selected_id=pred_id)


# ──────────────────────────────────────────
# F8 — Retention Strategy
# ──────────────────────────────────────────
@advanced_bp.route('/retention-strategy')
def retention_strategy():
    pred_id = request.args.get('id', type=int)
    record = None
    strategy = None
    predictions = _all_predictions()

    if pred_id:
        for p in predictions:
            if p.get('id') == pred_id:
                record = p
                break
        if record:
            strategy = get_retention_strategy(record)

    return render_template('retention_strategy.html',
                           predictions=predictions, record=record,
                           strategy=strategy, selected_id=pred_id)


# ──────────────────────────────────────────
# F9 — Business Impact Analyzer
# ──────────────────────────────────────────
@advanced_bp.route('/business-impact')
def business_impact():
    data = get_business_impact()
    return render_template('business_impact.html', data=data)


# ──────────────────────────────────────────
# F10 — Customer Persona
# ──────────────────────────────────────────
@advanced_bp.route('/customer-persona')
def customer_persona():
    personas, summary = get_persona_data()
    return render_template('customer_persona.html', personas=personas, summary=summary)


# ──────────────────────────────────────────
# F11 — Risk Alerts
# ──────────────────────────────────────────
@advanced_bp.route('/risk-alerts')
def risk_alerts():
    alerts = get_high_risk_alerts()
    return render_template('risk_alerts.html', alerts=alerts)


# ──────────────────────────────────────────
# F12 — Dataset Insights
# ──────────────────────────────────────────
@advanced_bp.route('/dataset-insights')
def dataset_insights():
    data = get_dataset_insights()
    return render_template('dataset_insights.html', data=data)


# ──────────────────────────────────────────
# F13 — Confidence Indicator
# ──────────────────────────────────────────
@advanced_bp.route('/confidence')
def confidence():
    data = get_confidence_data()
    return render_template('confidence_indicator.html', data=data)


# ──────────────────────────────────────────
# F14 — Interactive Charts
# ──────────────────────────────────────────
@advanced_bp.route('/interactive-charts')
def interactive_charts():
    data = get_interactive_chart_data()
    return render_template('interactive_charts.html', data=data)


# ──────────────────────────────────────────
# F15 — Activity Monitor
# ──────────────────────────────────────────
@advanced_bp.route('/activity-monitor')
def activity_monitor():
    stats = get_activity_stats()
    return render_template('activity_monitor.html', stats=stats)
