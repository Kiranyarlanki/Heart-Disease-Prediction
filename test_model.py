#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced HeartCare AI model capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_sample_data, train_model, calculate_individual_risk_factors, generate_recommendations

def test_model_performance():
    """Test the enhanced model with sample cases"""
    print("🏥 HeartCare AI - Enhanced Model Test")
    print("=" * 50)
    
    # Train the model
    print("Training enhanced model...")
    model, scaler = train_model()
    
    print("\n" + "=" * 50)
    print("🧪 Testing Sample Cases")
    print("=" * 50)
    
    # Test Case 1: Low Risk Profile
    print("\n📊 Test Case 1: Low Risk Profile")
    print("-" * 30)
    low_risk_data = {
        'age': '35',
        'sex': '0',  # Female
        'cp': '2',   # Non-anginal pain
        'trestbps': '110',
        'chol': '180',
        'fbs': '0',
        'restecg': '0',
        'thalach': '170',
        'exang': '0',
        'oldpeak': '0.5',
        'slope': '2',
        'ca': '0',
        'thal': '0'
    }
    
    # Calculate risk factors and recommendations
    risk_factors = calculate_individual_risk_factors(low_risk_data)
    recommendations = generate_recommendations(low_risk_data, "Low")
    
    print(f"Age: {low_risk_data['age']}, Sex: Female, BP: {low_risk_data['trestbps']}")
    print(f"Cholesterol: {low_risk_data['chol']}, Max HR: {low_risk_data['thalach']}")
    print("\nRisk Factors Analysis:")
    for factor, data in risk_factors.items():
        print(f"  • {factor}: {data['score']}% ({data['level']}) - {data['message']}")
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    # Test Case 2: High Risk Profile
    print("\n📊 Test Case 2: High Risk Profile")
    print("-" * 30)
    high_risk_data = {
        'age': '68',
        'sex': '1',  # Male
        'cp': '0',   # Asymptomatic
        'trestbps': '165',
        'chol': '290',
        'fbs': '1',
        'restecg': '1',
        'thalach': '95',
        'exang': '1',
        'oldpeak': '3.2',
        'slope': '0',
        'ca': '2',
        'thal': '2'
    }
    
    # Calculate risk factors and recommendations
    risk_factors = calculate_individual_risk_factors(high_risk_data)
    recommendations = generate_recommendations(high_risk_data, "High")
    
    print(f"Age: {high_risk_data['age']}, Sex: Male, BP: {high_risk_data['trestbps']}")
    print(f"Cholesterol: {high_risk_data['chol']}, Max HR: {high_risk_data['thalach']}")
    print("\nRisk Factors Analysis:")
    for factor, data in risk_factors.items():
        print(f"  • {factor}: {data['score']}% ({data['level']}) - {data['message']}")
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 50)
    print("✅ Model testing completed successfully!")
    print("🌐 Start the web application with: python app.py")
    print("📱 Access at: http://localhost:5000")
    print("🔐 Create an account to access the prediction system")

if __name__ == "__main__":
    test_model_performance()
