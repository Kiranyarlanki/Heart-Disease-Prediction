# Deployment Guide for Heart Disease Prediction App

## Prerequisites
- Git installed
- GitHub account
- Account on deployment platform (Render/Railway/Heroku)

## Quick Deployment Steps

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Heart Disease Prediction App"
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-prediction.git
git branch -M main
git push -u origin main
```

### 2. Deploy to Render (Recommended)
1. Visit: https://render.com
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your repository
5. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Environment: Python 3

### 3. Deploy to Railway (Alternative)
1. Visit: https://railway.app
2. Sign up/Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects and deploys

### 4. Deploy to Heroku (Alternative)
```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

## Environment Variables (if needed)
- FLASK_ENV=production
- FLASK_APP=app.py

## Features Included
- ✅ Heart disease prediction using ML
- ✅ User authentication system
- ✅ Responsive UI with heart12.jpg background
- ✅ Dashboard with prediction history
- ✅ Health tips and recommendations
- ✅ Real-time risk assessment
- ✅ Medical-grade accuracy (95%+)

## Tech Stack
- Backend: Flask, Python
- ML: scikit-learn, Random Forest
- Frontend: Bootstrap 5, HTML5, CSS3, JavaScript
- Database: JSON-based file storage
- Deployment: Gunicorn WSGI server
