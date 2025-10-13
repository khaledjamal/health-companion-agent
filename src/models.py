#models.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from datetime import date as DateType  # Rename to avoid conflict with field names
from enum import Enum


# ============================================================================
# Health Metrics Models
# ============================================================================

class HealthMetrics(BaseModel):
    """User health metrics for a single day."""
    
    date: DateType = Field(description="Date of the health metrics")
    user_id: int = Field(description="User identifier")
    
    # Sleep Metrics
    sleep_hours: float = Field(ge=0, le=24, description="Total sleep hours")
    sleep_quality: int = Field(ge=1, le=10, description="Subjective sleep quality (1-10)")
    
    # Heart Metrics
    hrv: int = Field(ge=0, le=300, description="Heart Rate Variability in ms")
    resting_heart_rate: int = Field(ge=30, le=200, description="Resting heart rate in bpm")
    
    # Activity Metrics
    steps: int = Field(ge=0, description="Daily step count")
    active_minutes: int = Field(ge=0, le=1440, description="Active minutes")
    
    # Wellness Metrics
    stress_score: int = Field(ge=1, le=10, description="Subjective stress level (1-10)")
    calories_consumed: Optional[int] = Field(None, ge=0, description="Calories consumed")
    water_intake_ml: Optional[int] = Field(None, ge=0, description="Water intake in ml")
    
    @validator('sleep_hours')
    def validate_sleep(cls, v):
        if v < 3 or v > 14:
            raise ValueError("Unrealistic sleep duration")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-10-11",
                "user_id": 1,
                "sleep_hours": 7.5,
                "sleep_quality": 8,
                "hrv": 65,
                "resting_heart_rate": 58,
                "steps": 10500,
                "active_minutes": 45,
                "stress_score": 4,
                "calories_consumed": 2200,
                "water_intake_ml": 2500
            }
        }


# ============================================================================
# Analysis Models
# ============================================================================

class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Anomaly(BaseModel):
    """Detected anomaly in health metrics."""
    metric: str = Field(description="Name of the metric with anomaly")
    current_value: float = Field(description="Current value of the metric")
    expected_range: str = Field(description="Expected range for this metric")
    severity: AnomalySeverity = Field(description="Severity of the anomaly")
    description: str = Field(description="Human-readable description of the anomaly")


class TrendDirection(str, Enum):
    """Direction of trends in metrics."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"


class Trend(BaseModel):
    """Trend analysis for a specific metric."""
    metric: str = Field(description="Name of the metric")
    direction: TrendDirection = Field(description="Direction of the trend")
    change_percentage: float = Field(description="Percentage change")
    description: str = Field(description="Human-readable description")


class HealthAnalysis(BaseModel):
    """Complete health data analysis result."""
    overall_score: int = Field(ge=0, le=100, description="Overall health score (0-100)")
    anomalies: List[Anomaly] = Field(default_factory=list, description="Detected anomalies")
    trends: List[Trend] = Field(default_factory=list, description="Detected trends")
    correlations: List[str] = Field(default_factory=list, description="Notable correlations")
    summary: str = Field(description="Summary of the analysis")


# ============================================================================
# Recommendation Models
# ============================================================================

class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class RecommendationCategory(str, Enum):
    """Categories of recommendations."""
    SLEEP = "sleep"
    RECOVERY = "recovery"
    ACTIVITY = "activity"
    NUTRITION = "nutrition"
    STRESS = "stress"
    GENERAL = "general"


class Recommendation(BaseModel):
    """Single actionable recommendation."""
    title: str = Field(description="Short title for the recommendation")
    description: str = Field(description="Detailed description of the recommendation")
    category: RecommendationCategory = Field(description="Category of recommendation")
    priority: RecommendationPriority = Field(description="Priority level")
    confidence: float = Field(ge=0, le=1, description="Confidence score (0-1)")
    rationale: str = Field(description="Why this recommendation is being made")
    expected_impact: str = Field(description="Expected impact if followed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Improve Sleep Consistency",
                "description": "Aim to sleep and wake at the same time daily, including weekends.",
                "category": "sleep",
                "priority": "high",
                "confidence": 0.9,
                "rationale": "Your HRV is below average, indicating poor recovery, which correlates with inconsistent sleep.",
                "expected_impact": "Improved HRV by 10-15% within 2 weeks"
            }
        }


class DailyReport(BaseModel):
    """Daily vitality report."""
    date: DateType = Field(description="Report date")
    user_id: int = Field(description="User identifier")
    analysis: HealthAnalysis = Field(description="Health analysis")
    recommendations: List[Recommendation] = Field(description="Personalized recommendations")
    key_insights: List[str] = Field(description="Key insights from the data")
    action_items: List[str] = Field(description="Top 3 action items for today")


# ============================================================================
# Knowledge Base Models
# ============================================================================

class KnowledgeDocument(BaseModel):
    """Document in the knowledge base."""
    content: str = Field(description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    topic: str = Field(description="Main topic of the document")
    relevance_score: Optional[float] = Field(None, description="Relevance score for retrieval")


class RetrievalQuery(BaseModel):
    """Query for knowledge base retrieval."""
    query: str = Field(description="Search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return")
    filter_topic: Optional[str] = Field(None, description="Filter by topic")


# ============================================================================
# Agent State Models
# ============================================================================

class AgentState(BaseModel):
    """State passed through the LangGraph agent."""
    
    # Input
    user_id: int
    date: DateType
    health_metrics: Optional[HealthMetrics] = None
    
    # Processing
    historical_data: List[HealthMetrics] = Field(default_factory=list)
    analysis: Optional[HealthAnalysis] = None
    retrieved_knowledge: List[KnowledgeDocument] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    
    # Output
    daily_report: Optional[DailyReport] = None
    
    # Metadata
    agent_steps: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Database Models (SQLModel)
# ============================================================================

from sqlmodel import SQLModel, Field as SQLField


class UserProfile(SQLModel, table=True):
    """User profile table."""
    __tablename__ = "user_profiles"
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    name: str
    age: Optional[int] = None
    goals: Optional[str] = None  # JSON string
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow)


class HealthRecord(SQLModel, table=True):
    """Health metrics record table."""
    __tablename__ = "health_records"
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="user_profiles.id")
    date: DateType
    
    # Metrics (stored as JSON for flexibility, but main fields here)
    sleep_hours: float
    sleep_quality: int
    hrv: int
    resting_heart_rate: int
    steps: int
    active_minutes: int
    stress_score: int
    calories_consumed: Optional[int] = None
    water_intake_ml: Optional[int] = None
    
    created_at: datetime = SQLField(default_factory=datetime.utcnow)


class RecommendationRecord(SQLModel, table=True):
    """Recommendation history table."""
    __tablename__ = "recommendations"
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="user_profiles.id")
    date: DateType
    
    title: str
    description: str
    category: str
    priority: str
    confidence: float
    
    # Feedback
    followed: Optional[bool] = None
    effectiveness: Optional[int] = None  # 1-5 rating
    
    created_at: datetime = SQLField(default_factory=datetime.utcnow)


class AgentExecution(SQLModel, table=True):
    """Agent execution log table."""
    __tablename__ = "agent_executions"
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="user_profiles.id")
    execution_date: datetime = SQLField(default_factory=datetime.utcnow)
    
    steps_taken: str  # JSON array of steps
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None