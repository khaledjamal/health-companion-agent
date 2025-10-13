#test_database.py

import pytest
from datetime import date
from src.memory.database import DatabaseManager
from src.models import HealthMetrics
from src.config import settings


@pytest.fixture
def db():
    """Create test database with PostgreSQL."""
    # Use test database URL (you can set this in .env as POSTGRES_TEST_DB)
    test_db_url = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/health_companion_test"
        f"?client_encoding=utf8"
    )
    
    db = DatabaseManager(test_db_url)
    db.create_tables()
    
    yield db
    
    # Cleanup: Drop all data after each test
    with db.get_session() as session:
        session.execute("TRUNCATE TABLE agent_executions, recommendations, health_records, user_profiles RESTART IDENTITY CASCADE")
        session.commit()


def test_create_user(db):
    """Test creating a user."""
    user = db.create_user("Test User", age=30, goals="Test goals")
    
    assert user.id is not None
    assert user.name == "Test User"
    assert user.age == 30


def test_save_and_retrieve_health_record(db):
    """Test saving and retrieving health records."""
    # Create user
    user = db.create_user("Test User")
    
    # Create metrics
    metrics = HealthMetrics(
        date=date(2025, 10, 11),
        user_id=user.id,
        sleep_hours=7.5,
        sleep_quality=8,
        hrv=65,
        resting_heart_rate=58,
        steps=10000,
        active_minutes=45,
        stress_score=4
    )
    
    # Save
    record = db.save_health_record(metrics)
    assert record.id is not None
    
    # Retrieve
    retrieved = db.get_health_record(user.id, date(2025, 10, 11))
    assert retrieved is not None
    assert retrieved.hrv == 65
    assert retrieved.sleep_hours == 7.5


def test_get_historical_records(db):
    """Test retrieving historical records."""
    user = db.create_user("Test User")
    
    # Create multiple records
    for i in range(1, 6):
        metrics = HealthMetrics(
            date=date(2025, 10, i),
            user_id=user.id,
            sleep_hours=7.0,
            sleep_quality=8,
            hrv=60,
            resting_heart_rate=58,
            steps=8000,
            active_minutes=30,
            stress_score=5
        )
        db.save_health_record(metrics)
    
    # Get historical
    records = db.get_historical_records(user.id, days=10)
    
    assert len(records) == 5


def test_save_recommendation(db):
    """Test saving recommendations."""
    user = db.create_user("Test User")
    
    rec = db.save_recommendation(
        user_id=user.id,
        recommendation_date=date(2025, 10, 11),
        title="Test Recommendation",
        description="Test description",
        category="sleep",
        priority="high",
        confidence=0.9
    )
    
    assert rec.id is not None
    assert rec.title == "Test Recommendation"