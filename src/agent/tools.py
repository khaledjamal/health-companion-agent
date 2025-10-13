#tools.py

from typing import List, Optional
from datetime import date
from loguru import logger

from ..models import (
    HealthMetrics,
    HealthAnalysis,
    KnowledgeDocument,
    RetrievalQuery,
    Recommendation
)
from ..memory.database import DatabaseManager
from ..memory.vector_store import VectorStoreManager
from ..data.analyzer import HealthDataAnalyzer
from ..recommendations.generator import RecommendationGenerator
from ..llm.openai_client import OpenAIClient


class AgentTools:
    """Tools that the agent can use to perform tasks."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_store: VectorStoreManager,
        llm_client: OpenAIClient
    ):
        """Initialize agent tools.
        
        Args:
            db_manager: Database manager
            vector_store: Vector store manager
            llm_client: LLM client
        """
        self.db = db_manager
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.analyzer = HealthDataAnalyzer()
        self.recommender = RecommendationGenerator(llm_client)
        
        logger.info("Agent tools initialized")
    
    def load_health_data(self, user_id: int, target_date: date) -> Optional[HealthMetrics]:
        """Load health data for a specific date.
        
        Args:
            user_id: User identifier
            target_date: Date to load data for
            
        Returns:
            Health metrics or None if not found
        """
        logger.info(f"Loading health data for user {user_id} on {target_date}")
        
        record = self.db.get_health_record(user_id, target_date)
        if not record:
            logger.warning(f"No health data found for user {user_id} on {target_date}")
            return None
        
        # Convert database record to HealthMetrics
        metrics = HealthMetrics(
            date=record.date,
            user_id=record.user_id,
            sleep_hours=record.sleep_hours,
            sleep_quality=record.sleep_quality,
            hrv=record.hrv,
            resting_heart_rate=record.resting_heart_rate,
            steps=record.steps,
            active_minutes=record.active_minutes,
            stress_score=record.stress_score,
            calories_consumed=record.calories_consumed,
            water_intake_ml=record.water_intake_ml
        )
        
        logger.info(f"Loaded health data for {target_date}")
        return metrics
    
    def load_historical_data(
        self,
        user_id: int,
        days: int = 30,
        end_date: Optional[date] = None
    ) -> List[HealthMetrics]:
        """Load historical health data.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            end_date: End date for the range
            
        Returns:
            List of health metrics
        """
        logger.info(f"Loading {days} days of historical data for user {user_id}")
        
        records = self.db.get_historical_records(user_id, days, end_date)
        
        metrics_list = []
        for record in records:
            metrics = HealthMetrics(
                date=record.date,
                user_id=record.user_id,
                sleep_hours=record.sleep_hours,
                sleep_quality=record.sleep_quality,
                hrv=record.hrv,
                resting_heart_rate=record.resting_heart_rate,
                steps=record.steps,
                active_minutes=record.active_minutes,
                stress_score=record.stress_score,
                calories_consumed=record.calories_consumed,
                water_intake_ml=record.water_intake_ml
            )
            metrics_list.append(metrics)
        
        logger.info(f"Loaded {len(metrics_list)} historical records")
        return metrics_list
    
    def analyze_health_data(
        self,
        current_metrics: HealthMetrics,
        historical_metrics: List[HealthMetrics]
    ) -> HealthAnalysis:
        """Analyze health data for anomalies and trends.
        
        Args:
            current_metrics: Current health metrics
            historical_metrics: Historical metrics for comparison
            
        Returns:
            Health analysis
        """
        logger.info("Analyzing health data")
        
        analysis = self.analyzer.analyze(current_metrics, historical_metrics)
        
        logger.info(f"Analysis complete: score={analysis.overall_score}")
        return analysis
    
    def retrieve_knowledge(
        self,
        query: str,
        top_k: int = 3,
        filter_topic: Optional[str] = None
    ) -> List[KnowledgeDocument]:
        """Retrieve relevant knowledge from vector store.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_topic: Filter by topic
            
        Returns:
            List of relevant knowledge documents
        """
        logger.info(f"Retrieving knowledge for query: '{query[:50]}...'")
        
        documents = self.vector_store.retrieve(query, top_k, filter_topic)
        
        logger.info(f"Retrieved {len(documents)} knowledge documents")
        return documents
    
    def generate_recommendations(
        self,
        analysis: HealthAnalysis,
        knowledge_docs: List[KnowledgeDocument],
        user_context: str = ""
    ) -> List[Recommendation]:
        """Generate personalized recommendations.
        
        Args:
            analysis: Health analysis
            knowledge_docs: Retrieved knowledge
            user_context: Additional user context
            
        Returns:
            List of recommendations
        """
        logger.info("Generating recommendations")
        
        recommendations = self.recommender.generate(
            analysis,
            knowledge_docs,
            user_context
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def save_recommendations(
        self,
        user_id: int,
        report_date: date,
        recommendations: List[Recommendation]
    ) -> None:
        """Save recommendations to database.
        
        Args:
            user_id: User identifier
            report_date: Date of the report
            recommendations: List of recommendations to save
        """
        logger.info(f"Saving {len(recommendations)} recommendations for user {user_id}")
        
        for rec in recommendations:
            self.db.save_recommendation(
                user_id=user_id,
                recommendation_date=report_date,
                title=rec.title,
                description=rec.description,
                category=rec.category.value,
                priority=rec.priority.value,
                confidence=rec.confidence
            )
        
        logger.info("Recommendations saved to database")
    
    def get_user_context(self, user_id: int) -> str:
        """Get user context from profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User context string
        """
        user = self.db.get_user(user_id)
        
        if not user:
            return ""
        
        context_parts = [f"User: {user.name}"]
        
        if user.age:
            context_parts.append(f"Age: {user.age}")
        
        if user.goals:
            context_parts.append(f"Goals: {user.goals}")
        
        return ", ".join(context_parts)