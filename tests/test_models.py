#test_models.py

import pytest
from datetime import date
from src.models import (
    HealthMetrics,
    HealthAnalysis,
    Anomaly,
    AnomalySeverity,
    Recommendation,
    RecommendationPriority,
    RecommendationCategory
)


def test_health_metrics_valid():
    """Test creating valid health metrics."""
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
    
    assert metrics.sleep_hours == 7.5
    assert metrics.user_id == 1
    assert metrics.hrv == 65


def test_health_metrics_validation():
    """Test health metrics validation."""
    # Invalid sleep hours
    with pytest.raises(ValueError):
        HealthMetrics(
            date=date(2025, 10, 11),
            user_id=1,
            sleep_hours=15.0,  # Too high
            sleep_quality=8,
            hrv=65,
            resting_heart_rate=58,
            steps=10000,
            active_minutes=45,
            stress_score=4
        )


def test_anomaly_creation():
    """Test creating anomaly."""
    anomaly = Anomaly(
        metric="hrv",
        current_value=35.0,
        expected_range="50-100",
        severity=AnomalySeverity.HIGH,
        description="HRV is significantly below normal range"
    )
    
    assert anomaly.metric == "hrv"
    assert anomaly.severity == AnomalySeverity.HIGH


def test_recommendation_creation():
    """Test creating recommendation."""
    rec = Recommendation(
        title="Improve Sleep",
        description="Get 8 hours of sleep tonight",
        category=RecommendationCategory.SLEEP,
        priority=RecommendationPriority.HIGH,
        confidence=0.9,
        rationale="Low HRV indicates poor recovery",
        expected_impact="HRV should improve by 15-20%"
    )
    
    assert rec.title == "Improve Sleep"
    assert rec.confidence == 0.9
    assert rec.priority == RecommendationPriority.HIGH


def test_health_analysis_creation():
    """Test creating health analysis."""
    analysis = HealthAnalysis(
        overall_score=75,
        anomalies=[],
        trends=[],
        correlations=[],
        summary="Health is good overall"
    )
    
    assert analysis.overall_score == 75
    assert len(analysis.anomalies) == 0