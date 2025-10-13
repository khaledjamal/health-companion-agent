#test_analyzer.py

import pytest
from datetime import date
from src.data.analyzer import HealthDataAnalyzer
from src.models import HealthMetrics, AnomalySeverity


def test_health_score_calculation():
    """Test health score calculation."""
    analyzer = HealthDataAnalyzer()
    
    # Good metrics
    metrics = HealthMetrics(
        date=date(2025, 10, 11),
        user_id=1,
        sleep_hours=8.0,
        sleep_quality=9,
        hrv=75,
        resting_heart_rate=55,
        steps=10000,
        active_minutes=50,
        stress_score=2
    )
    
    analysis = analyzer.analyze(metrics)
    
    assert analysis.overall_score >= 80
    assert analysis.overall_score <= 100


def test_anomaly_detection():
    """Test anomaly detection."""
    analyzer = HealthDataAnalyzer()
    
    # Poor metrics
    metrics = HealthMetrics(
        date=date(2025, 10, 11),
        user_id=1,
        sleep_hours=5.0,  # Below range
        sleep_quality=4,
        hrv=35,  # Below range
        resting_heart_rate=85,  # Above range
        steps=3000,  # Below range
        active_minutes=10,
        stress_score=9  # High stress
    )
    
    analysis = analyzer.analyze(metrics)
    
    assert len(analysis.anomalies) > 0
    assert any(a.metric == "sleep_hours" for a in analysis.anomalies)
    assert any(a.metric == "hrv" for a in analysis.anomalies)


def test_trend_detection():
    """Test trend detection with historical data."""
    analyzer = HealthDataAnalyzer()
    
    # Create improving trend in HRV
    historical = [
        HealthMetrics(
            date=date(2025, 10, i),
            user_id=1,
            sleep_hours=7.5,
            sleep_quality=8,
            hrv=50 + i,  # Improving trend
            resting_heart_rate=60,
            steps=8000,
            active_minutes=30,
            stress_score=5
        )
        for i in range(1, 8)
    ]
    
    current = HealthMetrics(
        date=date(2025, 10, 11),
        user_id=1,
        sleep_hours=7.5,
        sleep_quality=8,
        hrv=58,
        resting_heart_rate=60,
        steps=8000,
        active_minutes=30,
        stress_score=5
    )
    
    analysis = analyzer.analyze(current, historical)
    
    # Should detect improving HRV trend
    hrv_trends = [t for t in analysis.trends if t.metric == "hrv"]
    assert len(hrv_trends) > 0


def test_empty_historical_data():
    """Test analysis with no historical data."""
    analyzer = HealthDataAnalyzer()
    
    metrics = HealthMetrics(
        date=date(2025, 10, 11),
        user_id=1,
        sleep_hours=7.5,
        sleep_quality=8,
        hrv=65,
        resting_heart_rate=58,
        steps=10000,
        active_minutes=45,
        stress_score=4
    )
    
    # Should not raise error
    analysis = analyzer.analyze(metrics, [])
    
    assert analysis is not None
    assert len(analysis.trends) == 0  # No trends without history