from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import json
import hashlib
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'heartcare-ai-secret-key-2024'  # Change this in production

# File paths for persistent storage
USERS_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.json')
PREDICTIONS_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.json')

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def datetime_decoder(dct):
    for k, v in dct.items():
        if isinstance(v, str) and 'T' in v:
            try:
                dct[k] = datetime.fromisoformat(v)
            except ValueError:
                pass
    return dct

# Load data from JSON files or initialize if not exists
def load_data():
    global users_db, predictions_db
    try:
        if os.path.exists(USERS_DB_FILE):
            with open(USERS_DB_FILE, 'r') as f:
                users_db = json.load(f, object_hook=datetime_decoder)
            print(f"Loaded {len(users_db)} users from {USERS_DB_FILE}")
        else:
            users_db = {}
            print(f"No users file found at {USERS_DB_FILE}, initializing empty database")
            
        if os.path.exists(PREDICTIONS_DB_FILE):
            with open(PREDICTIONS_DB_FILE, 'r') as f:
                predictions_db = json.load(f, object_hook=datetime_decoder)
            print(f"Loaded {sum(len(preds) for preds in predictions_db.values())} predictions from {PREDICTIONS_DB_FILE}")
        else:
            predictions_db = {}
            print(f"No predictions file found at {PREDICTIONS_DB_FILE}, initializing empty database")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        users_db = {}
        predictions_db = {}
        # Try to create the files if they don't exist
        save_data()

# Save data to JSON files
def save_data():
    try:
        os.makedirs(os.path.dirname(USERS_DB_FILE), exist_ok=True)
        with open(USERS_DB_FILE, 'w') as f:
            json.dump(users_db, f, cls=DateTimeEncoder, indent=2)
        print(f"Saved {len(users_db)} users to {USERS_DB_FILE}")
        
        os.makedirs(os.path.dirname(PREDICTIONS_DB_FILE), exist_ok=True)
        with open(PREDICTIONS_DB_FILE, 'w') as f:
            json.dump(predictions_db, f, cls=DateTimeEncoder, indent=2)
        print(f"Saved predictions to {PREDICTIONS_DB_FILE}")
    except Exception as e:
        print(f"Error saving data: {e}")

# Load data on startup
load_data()

# Global variables for model and scaler
model = None
scaler = None

def create_sample_data():
    """Create realistic heart disease dataset for training based on medical research"""
    np.random.seed(42)
    n_samples = 2000  # Increased sample size for better training
    
    # Generate realistic heart disease data with medical correlations
    data = []
    
    for i in range(n_samples):
        # Age distribution: more realistic age ranges
        age = np.random.choice(
            [np.random.randint(25, 45), np.random.randint(45, 65), np.random.randint(65, 85)],
            p=[0.3, 0.5, 0.2]  # More middle-aged and elderly patients
        )
        
        # Sex: slightly more males in heart disease studies
        sex = np.random.choice([0, 1], p=[0.4, 0.6])
        
        # Chest pain type: realistic distribution
        cp = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
        
        # Blood pressure: age and sex correlated
        base_bp = 110 + (age - 25) * 0.8 + sex * 5
        trestbps = int(np.random.normal(base_bp, 15))
        trestbps = max(90, min(200, trestbps))
        
        # Cholesterol: age and lifestyle correlated
        base_chol = 180 + (age - 25) * 1.2 + np.random.choice([0, 30, 60], p=[0.6, 0.3, 0.1])
        chol = int(np.random.normal(base_chol, 25))
        chol = max(120, min(400, chol))
        
        # Fasting blood sugar: age correlated
        fbs_prob = 0.1 + (age - 25) * 0.003
        fbs = np.random.choice([0, 1], p=[1-fbs_prob, fbs_prob])
        
        # Resting ECG: age and health correlated
        restecg = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        
        # Max heart rate: age inversely correlated
        base_hr = 220 - age + np.random.randint(-10, 10)
        thalach = int(np.random.normal(base_hr * 0.85, 20))
        thalach = max(60, min(220, thalach))
        
        # Exercise induced angina: correlated with age and other factors
        exang_prob = 0.1 + (age - 25) * 0.004 + (cp == 0) * 0.3
        exang = np.random.choice([0, 1], p=[1-exang_prob, exang_prob])
        
        # ST depression: correlated with heart problems
        oldpeak_base = (age - 25) * 0.02 + exang * 1.5 + (cp == 0) * 1.0
        oldpeak = max(0, np.random.normal(oldpeak_base, 0.8))
        oldpeak = min(6, oldpeak)
        
        # ST slope: correlated with heart health
        slope = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
        
        # Major vessels: correlated with severity
        ca_prob = [0.6, 0.25, 0.1, 0.05]
        ca = np.random.choice([0, 1, 2, 3], p=ca_prob)
        
        # Thalassemia: realistic distribution
        thal = np.random.choice([0, 1, 2, 3], p=[0.6, 0.15, 0.2, 0.05])
        
        # Calculate realistic target based on medical risk factors
        risk_score = 0
        
        # Age risk (major factor)
        if age > 65: risk_score += 3
        elif age > 55: risk_score += 2
        elif age > 45: risk_score += 1
        
        # Sex risk (males higher risk)
        if sex == 1: risk_score += 1.5
        
        # Chest pain (asymptomatic is highest risk)
        if cp == 0: risk_score += 3  # Asymptomatic
        elif cp == 1: risk_score += 1  # Atypical angina
        elif cp == 3: risk_score += 2  # Typical angina
        
        # Blood pressure risk
        if trestbps > 160: risk_score += 2
        elif trestbps > 140: risk_score += 1
        
        # Cholesterol risk
        if chol > 280: risk_score += 2
        elif chol > 240: risk_score += 1
        
        # Diabetes risk
        if fbs == 1: risk_score += 1.5
        
        # ECG abnormalities
        if restecg == 2: risk_score += 2  # LVH
        elif restecg == 1: risk_score += 1  # ST-T abnormality
        
        # Low max heart rate (poor fitness)
        expected_hr = 220 - age
        if thalach < expected_hr * 0.6: risk_score += 2
        elif thalach < expected_hr * 0.75: risk_score += 1
        
        # Exercise induced angina
        if exang == 1: risk_score += 2
        
        # ST depression
        if oldpeak > 3: risk_score += 3
        elif oldpeak > 2: risk_score += 2
        elif oldpeak > 1: risk_score += 1
        
        # ST slope
        if slope == 0: risk_score += 2  # Downsloping
        elif slope == 1: risk_score += 1  # Flat
        
        # Major vessels
        risk_score += ca * 1.5
        
        # Thalassemia
        if thal == 1: risk_score += 2  # Fixed defect
        elif thal == 2: risk_score += 3  # Reversible defect
        
        # Add some controlled randomness for realistic variation
        risk_score += np.random.normal(0, 1)
        
        # Determine target with balanced distribution
        # Adjust threshold to get roughly 50-50 distribution
        target = 1 if risk_score > 6.5 else 0
        
        data.append({
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': round(oldpeak, 1),
            'slope': slope,
            'ca': ca,
            'thal': thal,
            'target': target
        })
    
    df = pd.DataFrame(data)
    
    # Ensure balanced dataset
    positive_cases = df[df['target'] == 1]
    negative_cases = df[df['target'] == 0]
    
    # Balance the dataset
    min_cases = min(len(positive_cases), len(negative_cases))
    balanced_df = pd.concat([
        positive_cases.sample(n=min_cases, random_state=42),
        negative_cases.sample(n=min_cases, random_state=42)
    ]).reset_index(drop=True)
    
    return balanced_df

def train_model():
    """Train the heart disease prediction model with enhanced accuracy"""
    global model, scaler
    
    # Force retrain for better model (remove this condition to always retrain)
    # if os.path.exists('heart_disease_model.joblib') and os.path.exists('scaler.joblib'):
    #     model = joblib.load('heart_disease_model.joblib')
    #     scaler = joblib.load('scaler.joblib')
    #     print("Loaded existing model and scaler")
    #     return
    
    print("Training enhanced heart disease prediction model...")
    
    # Create and prepare realistic data
    df = create_sample_data()
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Dataset created with {len(df)} samples")
    print(f"Positive cases (heart disease): {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"Negative cases (no heart disease): {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    # Split the data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train an enhanced Random Forest model with better parameters
    model = RandomForestClassifier(
        n_estimators=200,           # More trees for better accuracy
        max_depth=10,               # Prevent overfitting
        min_samples_split=5,        # Minimum samples to split
        min_samples_leaf=2,         # Minimum samples in leaf
        max_features='sqrt',        # Feature selection strategy
        bootstrap=True,             # Bootstrap sampling
        class_weight='balanced',    # Handle class imbalance
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    # Get predictions for detailed metrics
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Testing Accuracy: {test_accuracy:.3f}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # Feature importance
    feature_names = X.columns
    importances = model.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:5]):
        print(f"{i+1}. {feature}: {importance:.3f}")
    
    # Save the model and scaler
    joblib.dump(model, 'heart_disease_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    print(f"\nModel saved successfully!")
    return model, scaler

def calculate_individual_risk_factors(data):
    """Calculate individual risk factor contributions"""
    risk_factors = {}
    
    # Age risk
    age = float(data['age'])
    if age > 65:
        risk_factors['Age'] = {'score': 85, 'level': 'High', 'message': 'Age over 65 significantly increases heart disease risk'}
    elif age > 55:
        risk_factors['Age'] = {'score': 60, 'level': 'Moderate', 'message': 'Age over 55 moderately increases risk'}
    elif age > 45:
        risk_factors['Age'] = {'score': 35, 'level': 'Low', 'message': 'Age over 45 slightly increases risk'}
    else:
        risk_factors['Age'] = {'score': 15, 'level': 'Low', 'message': 'Age is not a significant risk factor'}
    
    # Sex risk
    sex = int(data['sex'])
    if sex == 1:
        risk_factors['Sex'] = {'score': 45, 'level': 'Moderate', 'message': 'Males have higher heart disease risk'}
    else:
        risk_factors['Sex'] = {'score': 25, 'level': 'Low', 'message': 'Female sex provides some protection'}
    
    # Blood pressure risk
    bp = float(data['trestbps'])
    if bp > 160:
        risk_factors['Blood Pressure'] = {'score': 80, 'level': 'High', 'message': 'High blood pressure (>160) is a major risk factor'}
    elif bp > 140:
        risk_factors['Blood Pressure'] = {'score': 55, 'level': 'Moderate', 'message': 'Elevated blood pressure (>140) increases risk'}
    elif bp > 120:
        risk_factors['Blood Pressure'] = {'score': 30, 'level': 'Low', 'message': 'Blood pressure is slightly elevated'}
    else:
        risk_factors['Blood Pressure'] = {'score': 15, 'level': 'Low', 'message': 'Blood pressure is in healthy range'}
    
    # Cholesterol risk
    chol = float(data['chol'])
    if chol > 280:
        risk_factors['Cholesterol'] = {'score': 75, 'level': 'High', 'message': 'Very high cholesterol (>280) significantly increases risk'}
    elif chol > 240:
        risk_factors['Cholesterol'] = {'score': 50, 'level': 'Moderate', 'message': 'High cholesterol (>240) increases risk'}
    elif chol > 200:
        risk_factors['Cholesterol'] = {'score': 30, 'level': 'Low', 'message': 'Cholesterol is borderline high'}
    else:
        risk_factors['Cholesterol'] = {'score': 15, 'level': 'Low', 'message': 'Cholesterol is in healthy range'}
    
    # Exercise capacity (max heart rate)
    age = float(data['age'])
    max_hr = float(data['thalach'])
    expected_hr = 220 - age
    hr_percentage = (max_hr / expected_hr) * 100
    
    if hr_percentage < 60:
        risk_factors['Exercise Capacity'] = {'score': 70, 'level': 'High', 'message': 'Poor exercise capacity indicates higher risk'}
    elif hr_percentage < 75:
        risk_factors['Exercise Capacity'] = {'score': 45, 'level': 'Moderate', 'message': 'Below average exercise capacity'}
    elif hr_percentage < 90:
        risk_factors['Exercise Capacity'] = {'score': 25, 'level': 'Low', 'message': 'Good exercise capacity'}
    else:
        risk_factors['Exercise Capacity'] = {'score': 10, 'level': 'Low', 'message': 'Excellent exercise capacity'}
    
    # Chest pain type
    cp = int(data['cp'])
    if cp == 0:
        risk_factors['Chest Pain'] = {'score': 85, 'level': 'High', 'message': 'Asymptomatic chest pain is highest risk type'}
    elif cp == 3:
        risk_factors['Chest Pain'] = {'score': 60, 'level': 'Moderate', 'message': 'Typical angina indicates coronary artery disease'}
    elif cp == 1:
        risk_factors['Chest Pain'] = {'score': 35, 'level': 'Low', 'message': 'Atypical angina may indicate heart issues'}
    else:
        risk_factors['Chest Pain'] = {'score': 20, 'level': 'Low', 'message': 'Non-anginal pain is lower risk'}
    
    return risk_factors

def generate_recommendations(data, risk_level):
    """Generate personalized health recommendations based on risk status"""
    recommendations = []
    
    age = int(data['age'])
    sex = int(data['sex'])
    cp = int(data['cp'])
    trestbps = int(data['trestbps'])
    chol = int(data['chol'])
    fbs = int(data['fbs'])
    thalach = int(data['thalach'])
    exang = int(data['exang'])
    oldpeak = float(data['oldpeak'])
    
    # Risk-specific recommendations based on medical parameters
    if risk_level in ["Very Low", "Low"]:
        # GREEN - Risk Not Detected: Safety Maintenance Recommendations
        recommendations.extend([
            "🎉 SAFE STATUS: Your heart health parameters indicate low risk - excellent work!",
            "💚 MAINTAIN SAFETY: Continue your current healthy lifestyle to keep your heart safe",
            "🏃‍♂️ KEEP ACTIVE: Maintain regular exercise (150+ minutes moderate activity/week)",
            "🥗 HEART-HEALTHY DIET: Continue Mediterranean-style diet rich in fruits, vegetables, whole grains",
            "📊 MONITOR REGULARLY: Annual health screenings to maintain your safe status",
            "💪 CARDIOVASCULAR FITNESS: Keep your heart strong with aerobic and strength training",
            "🧘 STRESS CONTROL: Continue effective stress management - it's protecting your heart",
            "😴 QUALITY SLEEP: Maintain 7-9 hours nightly for optimal heart recovery",
            "🚭 STAY SMOKE-FREE: Continue avoiding tobacco - major factor in your safety",
            "🍷 MODERATE ALCOHOL: If you drink, keep it moderate (1-2 drinks max per day)",
            "🏆 ROLE MODEL: Share your healthy habits with family and friends",
            "📚 STAY INFORMED: Keep learning about heart health prevention strategies"
        ])
    else:
        # RED - Risk Detected: Medical Intervention and Precautions
        recommendations.extend([
            "🚨 RISK DETECTED: Your parameters indicate elevated heart disease risk - action needed",
            "❤️ URGENT CARE: Schedule immediate consultation with healthcare provider",
            "💊 MEDICATION REVIEW: Discuss blood pressure, cholesterol, or diabetes medications",
            "📊 VITAL MONITORING: Daily blood pressure and heart rate tracking required",
            "⚠️ WARNING SIGNS: Watch for chest pain, shortness of breath, dizziness, fatigue",
            "🏥 EMERGENCY PLAN: Know when to call 911 - don't delay with cardiac symptoms",
            "🚑 EMERGENCY INFO: Keep medical history and emergency contacts readily available",
            "👨‍⚕️ SPECIALIST REFERRAL: Consider cardiology consultation for comprehensive evaluation",
            "📋 TREATMENT PLAN: Work with healthcare team to develop risk reduction strategy",
            "🩺 FREQUENT CHECKUPS: More frequent monitoring until risk factors are controlled"
        ])
    
    # Specific parameter-based recommendations
    if trestbps > 140:
        if risk_level in ["Very Low", "Low"]:
            recommendations.append("📈 BLOOD PRESSURE: Monitor regularly and maintain healthy weight")
        else:
            recommendations.append("🚨 HIGH BP: Urgent - your blood pressure needs immediate medical attention")
            recommendations.append("🧂 REDUCE SODIUM: Limit salt intake to less than 2,300mg daily")
    
    if chol > 240:
        if risk_level in ["Very Low", "Low"]:
            recommendations.append("🥑 CHOLESTEROL: Include healthy fats like avocados and nuts in diet")
        else:
            recommendations.append("⚠️ HIGH CHOLESTEROL: Discuss statin therapy with your cardiologist")
            recommendations.append("🐟 OMEGA-3: Increase fatty fish consumption (salmon, mackerel)")
    
    if fbs == 1:
        if risk_level in ["Very Low", "Low"]:
            recommendations.append("🍎 DIABETES CARE: Maintain good blood sugar control with diet and exercise")
        else:
            recommendations.append("🚨 DIABETES ALERT: Strict blood glucose monitoring required")
            recommendations.append("🥗 DIABETIC DIET: Work with nutritionist for meal planning")
    
    if exang == 1:
        recommendations.append("⛔ EXERCISE CAUTION: Stop activity if you experience chest pain")
        recommendations.append("🏥 CARDIAC EVALUATION: Stress test evaluation recommended")
    
    if oldpeak > 2:
        recommendations.append("❤️ CARDIAC MONITORING: EKG changes require cardiology follow-up")
    
    # Age and sex-specific recommendations
    if age > 65:
        if risk_level in ["Very Low", "Low"]:
            recommendations.append("👴 SENIOR HEALTH: Maintain bone health with calcium and vitamin D")
        else:
            recommendations.append("🏥 SENIOR CARE: More frequent cardiac monitoring needed at your age")
    
    if sex == 1 and age > 45:  # Male over 45
        if risk_level not in ["Very Low", "Low"]:
            recommendations.append("👨 MALE RISK: Men have higher cardiac risk - extra vigilance needed")
    elif sex == 0 and age > 55:  # Female over 55
        if risk_level not in ["Very Low", "Low"]:
            recommendations.append("👩 POSTMENOPAUSAL: Consider hormone effects on heart health")
    
    # Exercise recommendations based on risk
    expected_hr = 220 - age
    if thalach < expected_hr * 0.6:
        if risk_level in ["Very Low", "Low"]:
            recommendations.append("🚶‍♀️ GENTLE EXERCISE: Start with walking and gradually increase intensity")
        else:
            recommendations.append("⚠️ LIMITED EXERCISE: Only light activities until medical clearance")
    
    # Final recommendations based on risk status
    if risk_level in ["Very Low", "Low"]:
        recommendations.extend([
            "🌟 PREVENTION FOCUS: You're on the right track - keep up the excellent work!",
            "📱 HEALTH APPS: Consider using fitness trackers to maintain activity levels",
            "🏆 ROLE MODEL: Share your healthy habits with family and friends"
        ])
    else:
        recommendations.extend([
            "🚨 URGENT CARE: Don't delay medical consultation - your heart health is at risk",
            "📋 TREATMENT PLAN: Work with healthcare team to develop comprehensive care plan",
            "👨‍⚕️ SPECIALIST CARE: Cardiology referral may be necessary for optimal management",
            "📞 SUPPORT SYSTEM: Inform family members about your condition and emergency plans"
        ])
    
    return recommendations[:15]  # Return top 15 most relevant recommendations

# Authentication helper functions
def login_required(f):
    """Decorator to require login for certain routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed

# Public routes (no login required)
@app.route('/welcome')
def welcome():
    """Public welcome/landing page"""
    return render_template('welcome.html')

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and authentication"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Debug information
        print(f"Login attempt for username: {username}")
        print(f"Available users in database: {list(users_db.keys())}")
        
        # Validate input
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')
        
        # Check if user exists and handle case sensitivity
        user_key = None
        for existing_username in users_db:
            if existing_username.lower() == username.lower():
                user_key = existing_username
                break
        
        if not user_key:
            print(f"User {username} not found in database")
            flash('Invalid username or password.', 'error')
            return render_template('login.html')
        
        # Verify password
        user = users_db[user_key]
        stored_hash = user['password']
        provided_hash = hash_password(password)
        print(f"Password verification for {username}: {stored_hash == provided_hash}")
        
        if not verify_password(password, stored_hash):
            print(f"Password verification failed for {username}")
            flash('Invalid username or password.', 'error')
            return render_template('login.html')
        
        # Login successful
        print(f"Successful login for {username}")
        session['user_id'] = user_key
        session['user_name'] = user['fullname']
        flash(f'Welcome back, {user["fullname"]}!', 'success')
        
        # Reload data to ensure we have the latest
        load_data()
        
        # Redirect to home page
        return redirect(url_for('index'))
    
    # GET request - show login form
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page and user creation"""
    if request.method == 'POST':
        # Get form data
        fullname = request.form.get('fullname', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()  # Store email in lowercase
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        age = request.form.get('age', '')
        
        print(f"Registration attempt - Username: {username}, Email: {email}")
        
        # Validate input
        errors = []
        
        if len(fullname) < 2:
            errors.append('Full name must be at least 2 characters long.')
        
        if len(username) < 3 or len(username) > 20:
            errors.append('Username must be 3-20 characters long.')
        
        if not username.replace('_', '').isalnum():
            errors.append('Username can only contain letters, numbers, and underscores.')
        
        # Check username case-insensitively
        username_exists = any(existing.lower() == username.lower() for existing in users_db)
        if username_exists:
            errors.append('Username already exists. Please choose a different one.')
            print(f"Username {username} already exists")
        
        if '@' not in email or '.' not in email:
            errors.append('Please enter a valid email address.')
        
        # Check if email already exists (case-insensitive)
        for user_data in users_db.values():
            if user_data['email'].lower() == email.lower():
                errors.append('Email address already registered.')
                print(f"Email {email} already registered")
                break
        
        if len(password) < 6:
            errors.append('Password must be at least 6 characters long.')
        
        if password != confirm_password:
            errors.append('Passwords do not match.')
        
        try:
            age_int = int(age)
            if age_int < 13 or age_int > 120:
                errors.append('Age must be between 13 and 120.')
        except ValueError:
            errors.append('Please enter a valid age.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('register.html')
        
        try:
            # Create new user with explicit datetime object
            new_user = {
                'password': hash_password(password),
                'fullname': fullname,
                'email': email.lower(),  # Store email in lowercase
                'age': int(age),
                'created_at': datetime.now()
            }
            
            # Add to database
            users_db[username] = new_user
            
            # Save to persistent storage immediately
            save_data()
            
            print(f"Successfully created new user: {username}")
            print(f"Current users in database: {list(users_db.keys())}")
            
            # Verify the data was saved by reloading
            load_data()
            if username in users_db:
                print(f"Verified user {username} was saved successfully")
            else:
                print(f"Warning: User {username} not found after save!")
            
            # Auto-login after registration
            session['user_id'] = username
            session['user_name'] = fullname
            
            flash(f'Account created successfully! Welcome to HeartCare, {fullname}!', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            print(f"Error during user registration: {e}")
            flash('An error occurred during registration. Please try again.', 'error')
            return render_template('register.html')
    
    # GET request - show registration form
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user and clear session"""
    user_name = session.get('user_name', 'User')
    session.clear()
    flash(f'Goodbye, {user_name}! You have been logged out successfully.', 'info')
    return redirect(url_for('welcome'))

# Main application routes - All require login
@app.route('/')
@login_required
def index():
    """Home page - requires login"""
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard - requires login"""
    user_id = session['user_id']
    user_predictions = predictions_db.get(user_id, [])
    return render_template('dashboard.html', predictions=user_predictions)

@app.route('/predict')
@login_required
def predict_page():
    """Prediction page - requires login"""
    return render_template('predict.html')

@app.route('/settings')
@login_required
def settings():
    """Settings page - requires login"""
    return render_template('settings.html')

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    """API endpoint for heart disease prediction - requires login"""
    try:
        # Get data from request
        data = request.json
        
        # Extract features in the correct order
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction with confidence assessment
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Calculate risk percentage
        risk_percentage = probability[1] * 100
        confidence = max(probability) * 100  # Confidence in the prediction
        if risk_percentage < 20:
            risk_level = "Very Low"
            risk_color = "success"
            risk_status = "Risk Not Detected"
            message = "Excellent! Your heart disease risk is very low. Your cardiovascular health appears to be in great condition."
        elif risk_percentage < 35:
            risk_level = "Low"
            risk_color = "success"
            risk_status = "Risk Not Detected"
            message = "Good news! Your heart disease risk is low. Continue maintaining your healthy lifestyle."
        elif risk_percentage < 50:
            risk_level = "Low-Moderate"
            risk_color = "warning"
            risk_status = "Risk Detected"
            message = "Your heart disease risk is low-moderate. Some lifestyle improvements may be beneficial."
        elif risk_percentage < 65:
            risk_level = "Moderate"
            risk_color = "warning"
            risk_status = "Risk Detected"
            message = "Your heart disease risk is moderate. Consider consulting with a healthcare professional."
        elif risk_percentage < 80:
            risk_level = "High"
            risk_color = "danger"
            risk_status = "Risk Detected"
            message = "Your heart disease risk is high. Please consult with a cardiologist for proper evaluation."
        else:
            risk_level = "Very High"
            risk_color = "danger"
            risk_status = "Risk Detected"
            message = "Your heart disease risk is very high. Immediate medical consultation is strongly recommended."
        
        # Calculate individual risk factors for detailed analysis
        risk_factors = calculate_individual_risk_factors(data)
        
        # Generate personalized recommendations
        recommendations = generate_recommendations(data, risk_level)
        
        # Save prediction to user-specific history (requires login)
        user_id = session['user_id']
        prediction_data = {
            'id': len([p for preds in predictions_db.values() for p in preds]) + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'risk_percentage': int(risk_percentage),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_status': risk_status,
            'message': message,
            'confidence': int(confidence),
            'input_data': data,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'user_id': user_id
        }
        
        # Store in user-specific predictions list and persist to storage
        if user_id not in predictions_db:
            predictions_db[user_id] = []
        predictions_db[user_id].append(prediction_data)
        save_data()
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'risk_percentage': round(risk_percentage, 2),
            'confidence': round(confidence, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_status': risk_status,
            'message': message,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'prediction_id': prediction_data['id']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/results/<int:prediction_id>')
@login_required
def results(prediction_id):
    """Results page for a specific prediction - requires login"""
    user_id = session['user_id']
    user_predictions = predictions_db.get(user_id, [])
    
    # Find prediction belonging to current user
    prediction = None
    for pred in user_predictions:
        if pred['id'] == prediction_id:
            prediction = pred
            break
    
    if not prediction:
        flash('Prediction not found or access denied.', 'error')
        return redirect(url_for('dashboard'))
    
    return render_template('results.html', prediction=prediction)

@app.route('/about')
@login_required
def about():
    """About page - requires login"""
    return render_template('about.html')

@app.route('/health-tips')
@login_required
def health_tips():
    """Health tips page for heart health - requires login"""
    return render_template('health_tips.html')

@app.route('/delete-prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    """Delete a specific prediction - requires login"""
    try:
        user_id = session['user_id']
        user_predictions = predictions_db.get(user_id, [])
        
        # Find and delete prediction belonging to current user
        deleted = False
        for i, pred in enumerate(user_predictions):
            if pred['id'] == prediction_id:
                del user_predictions[i]
                deleted = True
                break
        
        if deleted:
            # Save changes to persistent storage
            save_data()
            return jsonify({'success': True, 'message': 'Prediction deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Prediction not found or access denied'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/delete-all-predictions', methods=['POST'])
@login_required
def delete_all_predictions():
    """Delete all user predictions - requires login"""
    try:
        user_id = session['user_id']
        if user_id in predictions_db:
            predictions_db[user_id] = []
            # Save changes to persistent storage
            save_data()
        return jsonify({'success': True, 'message': 'All your predictions deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    # Train the model on startup
    train_model()
    # Use localhost for local development
    app.run(debug=True, host='127.0.0.1', port=5000)
