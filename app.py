from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import pickle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from database import init_db
from datetime import datetime
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


init_db()


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'Models/model.pkl'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('Models', exist_ok=True)

data_uploaded = False
model_trained = False

FEATURES = [
    'International plan',
    'Voice mail plan',
    'Number vmail messages',
    'Total day minutes',
    'Total eve minutes',
    'Total night minutes',
    'Customer service calls'
]

# ---------------- OFFER ENGINE (CHURN BASED) ----------------
def generate_offers(values, churn_prob):
    international, voicemail, vmail, day, eve, night, service = values
    offers = []

    if churn_prob > 70:
        offers.append("🔥 3 Months Recharge + Hotstar + Amazon Prime")
    if day > 250:
        offers.append("📺 2 Months Recharge + Hotstar Subscription")
    if night > 200:
        offers.append("🌙 Unlimited Night Data Pack")
    if service >= 4:
        offers.append("💸 10% Monthly Bill Discount for 3 Months")
    if international == 0:
        offers.append("🌍 International Calling Pack at 50% Off")
    if voicemail == 0:
        offers.append("📩 Free Voice Mail Service for 1 Month")
    if not offers:
        offers.append("🎁 Loyalty Reward: Extra 5GB Data This Month")

    return offers


# ---------------- LOYALTY SCORE ----------------
def calculate_loyalty(values):
    international, voicemail, vmail, day, eve, night, service = values
    score = 0

    score += min(day / 4, 25)
    score += min(eve / 5, 20)
    score += min(night / 6, 20)

    if voicemail == 1:
        score += 10
    if international == 1:
        score += 10

    score -= service * 5
    return max(0, min(100, round(score, 2)))


def loyalty_reward(loyalty_percent):
    slab = int(loyalty_percent // 5) * 5
    rewards = {
        0: "Welcome Message",
        5: "500MB Data",
        10: "1GB Data",
        15: "2GB Data",
        20: "5% Discount",
        25: "2GB Data + 5% Discount",
        30: "10% Discount",
        35: "3GB Data",
        40: "10% Discount + 1GB Data",
        45: "5GB Data",
        50: "15% Discount",
        55: "10GB Data",
        60: "20% Discount",
        65: "₹50 Cashback",
        70: "₹100 Cashback",
        75: "10GB Data + 20% Discount",
        80: "Free OTT Subscription",
        85: "Priority Support + 10GB",
        90: "₹150 Cashback",
        95: "₹250 Cashback + Premium Upgrade"
    }
    return rewards.get(slab, "Exclusive Loyalty Reward")


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------- UPLOAD ----------------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global data_uploaded
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            file.save(os.path.join(UPLOAD_FOLDER, 'data.csv'))
            data_uploaded = True
            return redirect(url_for('model'))
    return render_template('upload.html')


# ---------------- TRAIN ----------------
@app.route('/model', methods=['GET', 'POST'])
def model():
    global model_trained

    if not data_uploaded:
        return "Please upload dataset first."

    best_model = best_name = best_acc = None
    svm_acc = rf_acc = None

    if request.method == 'POST':
        df = pd.read_csv('uploads/data.csv')

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
            return "Dataset became empty after preprocessing."

        X = df[FEATURES]
        y = df['Churn'].astype(int)

        svm = SVC(probability=True, class_weight="balanced")
        svm.fit(X, y)
        svm_acc = accuracy_score(y, svm.predict(X))

        rf = RandomForestClassifier()
        rf.fit(X, y)
        rf_acc = accuracy_score(y, rf.predict(X))

        if rf_acc >= svm_acc:
            best_model, best_name, best_acc = rf, "Random Forest", rf_acc
        else:
            best_model, best_name, best_acc = svm, "SVM", svm_acc

        pickle.dump(best_model, open(MODEL_PATH, 'wb'))
        model_trained = True

    return render_template(
        'model.html',
        best_model=best_name,
        score=round(best_acc, 3) if best_acc else None,
        svm_acc=round(svm_acc, 3) if svm_acc else None,
        rf_acc=round(rf_acc, 3) if rf_acc else None
    )


# ---------------- PREDICTION ----------------
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if not model_trained:
        return "Please train the model first."

    result = risk = persona = probability = None
    urgency = confidence = revenue_loss = None
    reasons = []
    suggestions = []
    offers = []
    loyalty_score = None
    loyalty_offer = None

    annual_value = clv = churn_cost = retention_budget = None
    tier = priority = engagement_health = None
    offer_suitability = growth_potential = strategy = None

    if request.method == 'POST':
        international = int(request.form['international'])
        voicemail = int(request.form['voicemail'])
        vmail = int(request.form['vmail_messages'])
        day = float(request.form['day_minutes'])
        eve = float(request.form['eve_minutes'])
        night = float(request.form['night_minutes'])
        service = int(request.form['service_calls'])

        active_app = int(request.form['active_app'])
        on_time_payment = int(request.form['on_time_payment'])
        long_term_user = int(request.form['long_term_user'])
        recent_recharge = int(request.form['recent_recharge'])
        uses_data = int(request.form['uses_data'])
        no_complaints = int(request.form['no_complaints'])

        monthly_bill = float(request.form['monthly_bill'])

        values = [international, voicemail, vmail, day, eve, night, service]

        model = pickle.load(open(MODEL_PATH, 'rb'))
        input_df = pd.DataFrame([values], columns=FEATURES)
        proba = model.predict_proba(input_df)[0][1] * 100

        # ---------------- FIXED USAGE LOGIC ONLY ----------------
        risk_boost = 0

        # Day minutes
        if day < 100:
            risk_boost += 8
        elif day < 200:
            risk_boost += 4
        else:
            risk_boost -= 3

        # Evening minutes
        if eve < 100:
            risk_boost += 6
        elif eve < 200:
            risk_boost += 3
        else:
            risk_boost -= 2

        # Night minutes
        if night < 100:
            risk_boost += 5
        elif night < 200:
            risk_boost += 2
        else:
            risk_boost -= 2

        # Your original risk rules
        if voicemail == 0:
            risk_boost += 10
        if international == 0:
            risk_boost += 6
        if service >= 4:
            risk_boost += 8

        # Engagement reduction (unchanged)
        loyalty_reduction = (
            active_app * 5 +
            on_time_payment * 5 +
            long_term_user * 5 +
            recent_recharge * 5 +
            uses_data * 5 +
            no_complaints * 5
        )

        final_proba = max(0, min(100, proba + risk_boost - loyalty_reduction))
        probability = round(final_proba, 2)
      


        # Everything below stays exactly same
        # ------------------------------------

        if final_proba >= 70:
            result, risk = "Churn", "High Risk"
        elif final_proba >= 49:
            result, risk = "Churn", "Medium Risk"
        else:
            result, risk = "No Churn", "Low Risk"
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions 
        (international, voicemail, vmail_messages, day_minutes, eve_minutes, night_minutes, service_calls, result, probability, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
        international,
        voicemail,
        vmail,
        day,
        eve,
        night,
        service,
        result,
        probability,
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()
        confidence = "Very High" if abs(final_proba - 50) >= 30 else "Moderate"
        urgency = min(100, int((final_proba * 0.7) + (service * 5)))
        revenue_loss = round(monthly_bill, 2)

        loyalty_score = calculate_loyalty(values)
        loyalty_offer = loyalty_reward(loyalty_score)

        annual_value = round(monthly_bill * 12, 2)
        clv = round(annual_value * (loyalty_score / 100), 2)
        churn_cost = round(monthly_bill * 6, 2)
        retention_budget = round(churn_cost * (probability / 100), 2)

        if clv > 20000:
            tier = "Platinum"
        elif clv > 10000:
            tier = "Gold"
        elif clv > 5000:
            tier = "Silver"
        else:
            tier = "Bronze"

        priority_score = (probability * clv) / 100
        if priority_score > 10000:
            priority = "HIGH 🔴"
        elif priority_score > 4000:
            priority = "MEDIUM 🟠"
        else:
            priority = "LOW 🟢"

        engagement_health = min(100, (active_app + recent_recharge + uses_data) * 33)
        offer_suitability = min(100, int((loyalty_score + (100 - probability)) / 2))

        if probability < 40 and loyalty_score > 60:
            growth_potential = "High"
        else:
            growth_potential = "Low"

        if probability > 60 and clv > 10000:
            strategy = "Premium Retention Program"
        elif probability > 60 and clv <= 10000:
            strategy = "Low-Cost Incentive Strategy"
        elif probability <= 40 and clv > 10000:
            strategy = "Loyalty Reward Strategy"
        else:
            strategy = "Minimal Action"

        offers = generate_offers(values, probability)

    return render_template(
        'predictions.html',
        result=result,
        risk=risk,
        probability=probability,
        confidence=confidence,
        urgency=urgency,
        revenue_loss=revenue_loss,
        offers=offers,
        loyalty_score=loyalty_score,
        loyalty_offer=loyalty_offer,
        annual_value=annual_value,
        clv=clv,
        churn_cost=churn_cost,
        retention_budget=retention_budget,
        tier=tier,
        priority=priority,
        engagement_health=engagement_health,
        offer_suitability=offer_suitability,
        growth_potential=growth_potential,
        strategy=strategy
    )
@app.route('/history')
def history():

    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    result = request.args.get('result', '')
    min_prob = request.args.get('min_prob', '')
    max_prob = request.args.get('max_prob', '')
    date = request.args.get('date', '')

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if result:
        query += " AND result = ?"
        params.append(result)

    if min_prob:
        query += " AND probability >= ?"
        params.append(min_prob)

    if max_prob:
        query += " AND probability <= ?"
        params.append(max_prob)

    if date:
        query += " AND date LIKE ?"
        params.append('%' + date + '%')

    count_query = query.replace("*", "COUNT(*)")

    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([per_page, offset])

    cursor.execute(query, params)
    data = cursor.fetchall()

    conn.close()

    total_pages = (total // per_page) + (1 if total % per_page else 0)

    return render_template(
        "history.html",
        data=data,
        page=page,
        total_pages=total_pages,
        result=result,
        min_prob=min_prob,
        max_prob=max_prob,
        date=date
    )



@app.route('/export_excel')
def export_excel():
    conn = sqlite3.connect("history.db")

    df = pd.read_sql_query("SELECT * FROM predictions", conn)

    os.makedirs("static", exist_ok=True)
    file_path = os.path.join("static", "prediction_history.xlsx")

    df.to_excel(file_path, index=False)

    conn.close()

    return redirect("/" + file_path)


@app.route('/export_pdf')
def export_pdf():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, result, probability, date FROM predictions")
    data = cursor.fetchall()

    file_path = "static/churn_report.pdf"

    pdf = canvas.Canvas(file_path, pagesize=letter)
    pdf.drawString(200, 750, "Customer Churn Prediction Report")

    y = 700

    for row in data:
        text = f"ID: {row[0]} | Result: {row[1]} | Probability: {row[2]}% | Date: {row[3]}"
        pdf.drawString(50, y, text)
        y -= 20

        if y < 50:
            pdf.showPage()
            y = 750

    pdf.save()
    conn.close()

    return redirect("/" + file_path)



@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE LOWER(result)='churn'")
    churn = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE LOWER(result)='no churn'")
    nochurn = cursor.fetchone()[0]

    cursor.execute("""
        SELECT 
        CASE 
            WHEN probability >= 70 THEN 'High'
            WHEN probability >= 50 THEN 'Medium'
            ELSE 'Low'
        END as risk,
        COUNT(*)
        FROM predictions
        GROUP BY risk
    """)
    risk_data = cursor.fetchall()

    cursor.execute("SELECT substr(date,1,10) as d, COUNT(*) FROM predictions GROUP BY d")
    trend = cursor.fetchall()

    cursor.execute("SELECT probability FROM predictions")
    probabilities = cursor.fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total=total,
        churn=churn,
        nochurn=nochurn,
        risk_data=risk_data,
        trend=trend,
        probabilities=probabilities
    )



@app.route('/delete/<int:id>')
def delete_record(id):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect(url_for('history'))


@app.route('/clear_history')
def clear_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

    return redirect(url_for('history'))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

