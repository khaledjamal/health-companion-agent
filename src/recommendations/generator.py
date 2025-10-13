#generator.py

from typing import List
from loguru import logger
from pydantic import BaseModel

from ..models import (
    HealthAnalysis,
    Recommendation,
    RecommendationPriority,
    RecommendationCategory,
    Anomaly,
    AnomalySeverity,
    KnowledgeDocument
)
from ..llm.openai_client import OpenAIClient


class RecommendationResponse(BaseModel):
    """Structured response for recommendations."""
    recommendations: List[Recommendation]


class RecommendationGenerator:
    """Generates personalized health recommendations."""
    
    def __init__(self, llm_client: OpenAIClient):
        """Initialize recommendation generator.
        
        Args:
            llm_client: OpenAI client for generating recommendations
        """
        self.llm_client = llm_client
        logger.info("Recommendation generator initialized")
    
    def generate(
        self,
        analysis: HealthAnalysis,
        retrieved_knowledge: List[KnowledgeDocument],
        user_context: str = ""
    ) -> List[Recommendation]:
        """Generate personalized recommendations.
        
        Args:
            analysis: Health analysis results
            retrieved_knowledge: Retrieved knowledge from RAG
            user_context: Additional user context
            
        Returns:
            List of recommendations
        """
        logger.info("Generating recommendations")
        
        # Build context from analysis
        context = self._build_context(analysis, retrieved_knowledge, user_context)
        
        # Generate recommendations using LLM with structured output
        recommendations = self._generate_with_llm(context, analysis)
        
        # Post-process and validate
        recommendations = self._post_process(recommendations, analysis)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _build_context(
        self,
        analysis: HealthAnalysis,
        knowledge_docs: List[KnowledgeDocument],
        user_context: str
    ) -> str:
        """Build context for recommendation generation.
        
        Args:
            analysis: Health analysis
            knowledge_docs: Retrieved knowledge documents
            user_context: User context
            
        Returns:
            Context string
        """
        context_parts = [
            f"Overall Health Score: {analysis.overall_score}/100",
            f"\nAnalysis Summary: {analysis.summary}",
        ]
        
        # Add anomalies
        if analysis.anomalies:
            context_parts.append("\n\nDetected Issues:")
            for anomaly in analysis.anomalies:
                context_parts.append(f"- {anomaly.description} (Severity: {anomaly.severity.value})")
        
        # Add trends
        if analysis.trends:
            context_parts.append("\n\nTrends:")
            for trend in analysis.trends:
                context_parts.append(f"- {trend.description}")
        
        # Add correlations
        if analysis.correlations:
            context_parts.append("\n\nCorrelations:")
            for corr in analysis.correlations:
                context_parts.append(f"- {corr}")
        
        # Add retrieved knowledge
        if knowledge_docs:
            context_parts.append("\n\nRelevant Health Knowledge:")
            for i, doc in enumerate(knowledge_docs[:3], 1):  # Top 3 docs
                context_parts.append(f"\n{i}. {doc.topic.title()}:")
                context_parts.append(doc.content[:500] + "..." if len(doc.content) > 500 else doc.content)
        
        # Add user context
        if user_context:
            context_parts.append(f"\n\nUser Context: {user_context}")
        
        return "\n".join(context_parts)
    
    def _generate_with_llm(
        self,
        context: str,
        analysis: HealthAnalysis
    ) -> List[Recommendation]:
        """Generate recommendations using LLM with structured output.
        
        Args:
            context: Context for generation
            analysis: Health analysis
            
        Returns:
            List of recommendations
        """
        # Determine number of recommendations based on issues
        num_recommendations = min(5, max(3, len(analysis.anomalies) + 1))
        
        system_prompt = """You are an expert health and wellness coach. Generate personalized, actionable health recommendations based on the analysis.

Guidelines:
- Prioritize addressing high-severity anomalies first
- Make recommendations specific and actionable
- Include clear rationale based on the data
- Estimate expected impact
- Be encouraging but realistic
- Consider correlations when making recommendations
- Base recommendations on the provided health knowledge when relevant

Generate {num_recs} recommendations ordered by priority.""".format(num_recs=num_recommendations)
        
        user_prompt = f"""{context}

Generate {num_recommendations} personalized health recommendations addressing the most important issues."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_client.create_structured_completion(
                messages=messages,
                response_model=RecommendationResponse,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations with LLM: {e}")
            # Fallback to rule-based recommendations
            return self._generate_rule_based(analysis)
    
    def _generate_rule_based(self, analysis: HealthAnalysis) -> List[Recommendation]:
        """Generate rule-based recommendations as fallback.
        
        Args:
            analysis: Health analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Generate recommendations based on anomalies
        for anomaly in analysis.anomalies[:3]:  # Top 3 anomalies
            rec = self._anomaly_to_recommendation(anomaly)
            if rec:
                recommendations.append(rec)
        
        # Add a general recommendation if we have few
        if len(recommendations) < 2:
            recommendations.append(Recommendation(
                title="Maintain Consistency",
                description="Continue monitoring your health metrics daily to identify patterns and track improvements.",
                category=RecommendationCategory.GENERAL,
                priority=RecommendationPriority.MEDIUM,
                confidence=0.8,
                rationale="Consistent tracking helps identify trends and measure progress.",
                expected_impact="Better health awareness and ability to make informed decisions"
            ))
        
        return recommendations
    
    def _anomaly_to_recommendation(self, anomaly: Anomaly) -> Recommendation:
        """Convert an anomaly to a recommendation.
        
        Args:
            anomaly: Detected anomaly
            
        Returns:
            Recommendation
        """
        # Map anomalies to recommendations
        recommendations_map = {
            'sleep_hours': {
                'title': 'Improve Sleep Duration',
                'description': 'Aim for 7-9 hours of sleep per night. Establish a consistent bedtime routine.',
                'category': RecommendationCategory.SLEEP,
                'expected_impact': 'Improved recovery, better HRV, enhanced cognitive function'
            },
            'hrv': {
                'title': 'Enhance Recovery',
                'description': 'Focus on recovery: ensure adequate sleep, manage stress, and consider light activity or rest days.',
                'category': RecommendationCategory.RECOVERY,
                'expected_impact': 'Improved HRV by 10-20% within 1-2 weeks'
            },
            'stress_score': {
                'title': 'Reduce Stress',
                'description': 'Practice stress-reduction techniques: meditation, deep breathing, or yoga for 10-15 minutes daily.',
                'category': RecommendationCategory.STRESS,
                'expected_impact': 'Lower stress levels and improved overall well-being'
            },
            'resting_heart_rate': {
                'title': 'Monitor Heart Health',
                'description': 'Elevated resting heart rate may indicate stress or overtraining. Prioritize recovery and consider consulting a healthcare provider.',
                'category': RecommendationCategory.RECOVERY,
                'expected_impact': 'Normalized heart rate and better cardiovascular health'
            },
            'steps': {
                'title': 'Increase Daily Activity',
                'description': 'Aim for at least 8,000 steps daily. Take short walking breaks throughout the day.',
                'category': RecommendationCategory.ACTIVITY,
                'expected_impact': 'Improved cardiovascular health and energy levels'
            }
        }
        
        rec_data = recommendations_map.get(anomaly.metric)
        if not rec_data:
            return None
        
        # Map severity to priority
        priority_map = {
            AnomalySeverity.HIGH: RecommendationPriority.HIGH,
            AnomalySeverity.MEDIUM: RecommendationPriority.MEDIUM,
            AnomalySeverity.LOW: RecommendationPriority.LOW
        }
        
        return Recommendation(
            title=rec_data['title'],
            description=rec_data['description'],
            category=rec_data['category'],
            priority=priority_map[anomaly.severity],
            confidence=0.85,
            rationale=f"Your {anomaly.metric.replace('_', ' ')} is {anomaly.description.lower()}",
            expected_impact=rec_data['expected_impact']
        )
    
    def _post_process(
        self,
        recommendations: List[Recommendation],
        analysis: HealthAnalysis
    ) -> List[Recommendation]:
        """Post-process and validate recommendations.
        
        Args:
            recommendations: Generated recommendations
            analysis: Health analysis
            
        Returns:
            Validated recommendations
        """
        # Remove duplicates based on title
        seen_titles = set()
        unique_recs = []
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recs.append(rec)
        
        # Sort by priority
        priority_order = {
            RecommendationPriority.URGENT: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3
        }
        
        sorted_recs = sorted(
            unique_recs,
            key=lambda r: (priority_order[r.priority], -r.confidence)
        )
        
        # Limit to top 5
        return sorted_recs[:5]