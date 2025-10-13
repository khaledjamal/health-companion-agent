#database.py

from sqlmodel import SQLModel, create_engine, Session, select
from typing import List, Optional
from datetime import date, datetime, timedelta
from loguru import logger

from ..models import (
    UserProfile,
    HealthRecord,
    RecommendationRecord,
    AgentExecution,
    HealthMetrics
)
from ..config import settings


class DatabaseManager:
    """Manages PostgreSQL database operations for user data and history."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL. Defaults to settings.
        """
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(
            self.database_url,
            echo=settings.environment == "development",
            pool_pre_ping=True  # Verify connections before using them
        )
        logger.info(f"Database initialized: {self.database_url}")
    
    def create_tables(self) -> None:
        """Create all database tables."""
        SQLModel.metadata.create_all(self.engine)
        logger.info("Database tables created successfully")
    
    def get_session(self) -> Session:
        """Get database session."""
        return Session(self.engine)
    
    # ========================================================================
    # User Profile Operations
    # ========================================================================
    
    def create_user(self, name: str, age: Optional[int] = None, goals: Optional[str] = None) -> UserProfile:
        """Create a new user profile.
        
        Args:
            name: User name
            age: User age
            goals: User health goals (JSON string)
            
        Returns:
            Created user profile
        """
        with self.get_session() as session:
            user = UserProfile(name=name, age=age, goals=goals)
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info(f"Created user profile: {user.id} - {user.name}")
            return user
    
    def get_user(self, user_id: int) -> Optional[UserProfile]:
        """Get user profile by ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile or None if not found
        """
        with self.get_session() as session:
            statement = select(UserProfile).where(UserProfile.id == user_id)
            user = session.exec(statement).first()
            return user
    
    def update_user(self, user_id: int, **kwargs) -> Optional[UserProfile]:
        """Update user profile.
        
        Args:
            user_id: User identifier
            **kwargs: Fields to update
            
        Returns:
            Updated user profile or None if not found
        """
        with self.get_session() as session:
            user = session.get(UserProfile, user_id)
            if user:
                for key, value in kwargs.items():
                    setattr(user, key, value)
                user.updated_at = datetime.utcnow()
                session.add(user)
                session.commit()
                session.refresh(user)
                logger.info(f"Updated user profile: {user_id}")
            return user
    
    # ========================================================================
    # Health Records Operations
    # ========================================================================
    
    def save_health_record(self, metrics: HealthMetrics) -> HealthRecord:
        """Save health metrics to database.
        
        Args:
            metrics: Health metrics to save
            
        Returns:
            Saved health record
        """
        with self.get_session() as session:
            # Check if record already exists for this date
            statement = select(HealthRecord).where(
                HealthRecord.user_id == metrics.user_id,
                HealthRecord.date == metrics.date
            )
            existing = session.exec(statement).first()
            
            if existing:
                # Update existing record
                for key, value in metrics.model_dump().items():
                    if key not in ['user_id', 'date']:
                        setattr(existing, key, value)
                session.add(existing)
                session.commit()
                session.refresh(existing)
                logger.info(f"Updated health record: user={metrics.user_id}, date={metrics.date}")
                return existing
            else:
                # Create new record
                record = HealthRecord(**metrics.model_dump())
                session.add(record)
                session.commit()
                session.refresh(record)
                logger.info(f"Created health record: user={metrics.user_id}, date={metrics.date}")
                return record
    
    def get_health_record(self, user_id: int, record_date: date) -> Optional[HealthRecord]:
        """Get health record for a specific date.
        
        Args:
            user_id: User identifier
            record_date: Date of the record
            
        Returns:
            Health record or None if not found
        """
        with self.get_session() as session:
            statement = select(HealthRecord).where(
                HealthRecord.user_id == user_id,
                HealthRecord.date == record_date
            )
            return session.exec(statement).first()
    
    def get_historical_records(
        self, 
        user_id: int, 
        days: int = 30,
        end_date: Optional[date] = None
    ) -> List[HealthRecord]:
        """Get historical health records.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            end_date: End date for the range (defaults to today)
            
        Returns:
            List of health records ordered by date descending
        """
        if end_date is None:
            end_date = date.today()
        
        start_date = end_date - timedelta(days=days)
        
        with self.get_session() as session:
            statement = select(HealthRecord).where(
                HealthRecord.user_id == user_id,
                HealthRecord.date >= start_date,
                HealthRecord.date <= end_date
            ).order_by(HealthRecord.date.desc())
            
            records = session.exec(statement).all()
            logger.info(f"Retrieved {len(records)} historical records for user {user_id}")
            return list(records)
    
    def get_metric_average(self, user_id: int, metric: str, days: int = 7) -> Optional[float]:
        """Calculate average for a specific metric over a period.
        
        Args:
            user_id: User identifier
            metric: Metric name (e.g., 'hrv', 'sleep_hours')
            days: Number of days to average over
            
        Returns:
            Average value or None if no data
        """
        records = self.get_historical_records(user_id, days=days)
        if not records:
            return None
        
        values = [getattr(record, metric) for record in records if hasattr(record, metric)]
        if not values:
            return None
        
        return sum(values) / len(values)
    
    # ========================================================================
    # Recommendation Operations
    # ========================================================================
    
    def save_recommendation(
        self,
        user_id: int,
        recommendation_date: date,
        title: str,
        description: str,
        category: str,
        priority: str,
        confidence: float
    ) -> RecommendationRecord:
        """Save a recommendation to the database.
        
        Args:
            user_id: User identifier
            recommendation_date: Date of the recommendation
            title: Recommendation title
            description: Recommendation description
            category: Category (sleep, recovery, etc.)
            priority: Priority level
            confidence: Confidence score
            
        Returns:
            Saved recommendation record
        """
        with self.get_session() as session:
            record = RecommendationRecord(
                user_id=user_id,
                date=recommendation_date,
                title=title,
                description=description,
                category=category,
                priority=priority,
                confidence=confidence
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info(f"Saved recommendation: {title} for user {user_id}")
            return record
    
    def get_recommendations(self, user_id: int, days: int = 7) -> List[RecommendationRecord]:
        """Get recent recommendations for a user.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            List of recommendations
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        with self.get_session() as session:
            statement = select(RecommendationRecord).where(
                RecommendationRecord.user_id == user_id,
                RecommendationRecord.date >= cutoff_date
            ).order_by(RecommendationRecord.date.desc())
            
            return list(session.exec(statement).all())
    
    def update_recommendation_feedback(
        self,
        recommendation_id: int,
        followed: bool,
        effectiveness: Optional[int] = None
    ) -> Optional[RecommendationRecord]:
        """Update feedback for a recommendation.
        
        Args:
            recommendation_id: Recommendation ID
            followed: Whether the recommendation was followed
            effectiveness: Effectiveness rating (1-5)
            
        Returns:
            Updated recommendation or None if not found
        """
        with self.get_session() as session:
            record = session.get(RecommendationRecord, recommendation_id)
            if record:
                record.followed = followed
                record.effectiveness = effectiveness
                session.add(record)
                session.commit()
                session.refresh(record)
                logger.info(f"Updated recommendation feedback: {recommendation_id}")
            return record
    
    # ========================================================================
    # Agent Execution Logging
    # ========================================================================
    
    def log_agent_execution(
        self,
        user_id: int,
        steps_taken: List[str],
        duration_seconds: float,
        success: bool,
        error_message: Optional[str] = None
    ) -> AgentExecution:
        """Log an agent execution.
        
        Args:
            user_id: User identifier
            steps_taken: List of steps the agent took
            duration_seconds: Execution duration
            success: Whether execution was successful
            error_message: Error message if failed
            
        Returns:
            Agent execution record
        """
        import json
        
        with self.get_session() as session:
            execution = AgentExecution(
                user_id=user_id,
                steps_taken=json.dumps(steps_taken),
                duration_seconds=duration_seconds,
                success=success,
                error_message=error_message
            )
            session.add(execution)
            session.commit()
            session.refresh(execution)
            logger.info(f"Logged agent execution for user {user_id}: success={success}")
            return execution
    
    def get_recent_executions(self, user_id: int, limit: int = 10) -> List[AgentExecution]:
        """Get recent agent executions.
        
        Args:
            user_id: User identifier
            limit: Maximum number of executions to return
            
        Returns:
            List of agent executions
        """
        with self.get_session() as session:
            statement = select(AgentExecution).where(
                AgentExecution.user_id == user_id
            ).order_by(AgentExecution.execution_date.desc()).limit(limit)
            
            return list(session.exec(statement).all())