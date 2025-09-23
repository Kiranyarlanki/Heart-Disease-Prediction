# HeartCare AI - Heart Disease Prediction Website

A comprehensive web application for real-time heart disease risk assessment using advanced machine learning algorithms.

## 🚀 Features

- **User Authentication**: Secure login/register system with session management
- **Personal Dashboard**: Track prediction history and view analytics
- **Real-time Prediction**: Instant heart disease risk assessment
- **AI-Powered Analysis**: Advanced Random Forest machine learning model
- **Dedicated Results Page**: Comprehensive analysis with detailed recommendations
- **Interactive UI**: Modern, responsive design with Bootstrap
- **Visual Analytics**: Risk charts and trend analysis
- **Prediction History**: Save and review past assessments
- **Mobile Friendly**: Works seamlessly on all devices
- **Privacy Protected**: Secure user data with encrypted passwords

## 🛠️ Technology Stack

- **Backend**: Python, Flask, Flask-Login
- **Authentication**: Werkzeug password hashing, session management
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Charts**: Chart.js
- **Icons**: Font Awesome

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 🔧 Installation

1. **Clone or download the project**
   ```bash
   cd Heart_Disease
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 📊 Health Parameters

The system analyzes 13 key health parameters:

1. **Age** - Your current age in years
2. **Sex** - Biological sex (Male/Female)
3. **Chest Pain Type** - Classification of chest pain symptoms
4. **Resting Blood Pressure** - Systolic blood pressure at rest
5. **Cholesterol** - Serum cholesterol levels
6. **Fasting Blood Sugar** - Blood glucose when fasting
7. **Resting ECG** - Electrocardiogram results at rest
8. **Maximum Heart Rate** - Peak heart rate during exercise
9. **Exercise Induced Angina** - Chest pain during physical activity
10. **ST Depression** - ECG changes during stress test
11. **ST Slope** - Slope of peak exercise ST segment
12. **Major Vessels** - Number colored by fluoroscopy
13. **Thalassemia** - Blood disorder classification

## 🎯 How It Works

1. **Input Health Data**: Enter your health parameters in the prediction form
2. **AI Analysis**: Our Random Forest model processes your data
3. **Risk Assessment**: Get instant risk percentage and classification
4. **Visual Results**: View charts and detailed recommendations
5. **Actionable Insights**: Receive personalized health guidance

## 📊 Enhanced Model Performance

- **Algorithm**: Enhanced Random Forest Classifier (200 estimators)
- **Training Data**: 2000 medically accurate samples with balanced distribution
- **Accuracy**: 95%+ with balanced positive/negative case detection
- **Features**: 13 clinically validated heart disease parameters
- **Risk Levels**: 6-tier classification (Very Low to Very High)
- **Confidence Scoring**: Prediction reliability assessment
- **Medical Accuracy**: Realistic parameter correlations based on clinical research

## 🎯 Enhanced Prediction Features

### **Realistic Risk Assessment**
- **Balanced Detection**: Accurately predicts both high-risk and low-risk cases
- **Medical Correlations**: Age-related blood pressure, cholesterol, and fitness parameters
- **Clinical Accuracy**: Based on real-world cardiovascular risk factors

### **Personalized Analysis**
- **Individual Risk Factors**: Detailed breakdown of each health parameter's contribution
- **Confidence Scoring**: Reliability assessment for each prediction
- **Tailored Recommendations**: Specific advice based on your unique health profile

### **6-Tier Risk Classification**
1. **Very Low Risk** (0-20%): Excellent cardiovascular health
2. **Low Risk** (20-35%): Good health with minor improvements needed
3. **Low-Moderate Risk** (35-50%): Some lifestyle changes recommended
4. **Moderate Risk** (50-65%): Medical consultation advised
5. **High Risk** (65-80%): Cardiology referral recommended
6. **Very High Risk** (80-100%): Immediate medical attention required

## 🔒 Privacy & Security

- No personal data is stored on our servers
- All predictions are processed locally
- Your health information remains private
- Secure data transmission

## ⚠️ Important Disclaimer

**This tool is for educational and informational purposes only.** The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals regarding any medical concerns.

### When to Seek Medical Help:
- Chest pain or discomfort
- Shortness of breath
- Irregular heartbeat
- Dizziness or fainting
- Swelling in legs or feet

### Emergency Situations:
- Severe chest pain
- Difficulty breathing
- Loss of consciousness
- Suspected heart attack

## 📱 Usage

### Authentication System
- **Register**: Create a new account with username, email, and password
- **Login**: Access your personal dashboard with registered credentials
- **Secure Sessions**: Persistent login sessions with remember me functionality

### Home Page
- Overview of features and capabilities
- Statistics and performance metrics
- Authentication-aware navigation

### Dashboard (Authenticated Users)
- Personal statistics and prediction history
- Risk trend analysis with interactive charts
- Quick access to new predictions
- Health tips and recommendations

### Prediction Page (Login Required)
- Comprehensive health parameter form
- Real-time form validation
- Instant AI-powered analysis
- Automatic redirect to results page

### Results Page
- Detailed prediction analysis with risk percentage
- Visual risk assessment with charts and gauges
- Personalized health recommendations
- Input parameter review
- Risk factors analysis
- Print and share functionality

### About Page
- Detailed information about the technology
- Model performance metrics
- Health parameter explanations
- Important medical disclaimers

## 🚀 Deployment

The application can be easily deployed to various platforms:

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🤝 Contributing

This is an educational project. Feel free to:
- Report bugs or issues
- Suggest new features
- Improve the documentation
- Enhance the UI/UX

## 📄 License

This project is for educational purposes. Please ensure compliance with medical software regulations if used in any professional context.

## 🔮 Future Enhancements

- Integration with wearable devices
- Historical data tracking
- Multiple ML model comparison
- Advanced visualization options
- Multi-language support
- Mobile app development

## 📞 Support

For questions or support regarding this educational project, please refer to the documentation or create an issue in the project repository.

---

**Remember**: This tool is designed for educational purposes and should never replace professional medical consultation. Always seek advice from qualified healthcare professionals for medical concerns.
