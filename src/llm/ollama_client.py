#ollama_client.py

from langchain_ollama import ChatOllama
from typing import List, Optional
from loguru import logger

from ..config import settings


class OllamaClient:
    """Ollama client for local LLM inference."""
    
    def __init__(self, base_url: str = None, model: str = None):
        """Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        
        self.client = ChatOllama(
            base_url=self.base_url,
            model=self.model,
            temperature=0.3
        )
        
        logger.info(f"Ollama client initialized: {self.model} at {self.base_url}")
    
    def classify_anomaly_severity(
        self,
        metric: str,
        current_value: float,
        expected_range: str,
        deviation_percentage: float
    ) -> str:
        """Classify anomaly severity using local model.
        
        Args:
            metric: Metric name
            current_value: Current value
            expected_range: Expected range
            deviation_percentage: Percentage deviation
            
        Returns:
            Severity classification: "low", "medium", or "high"
        """
        prompt = f"""Classify the severity of this health metric anomaly:

Metric: {metric}
Current Value: {current_value}
Expected Range: {expected_range}
Deviation: {deviation_percentage:.1f}%

Classify as: low, medium, or high

Consider:
- Critical metrics (HRV, sleep, heart rate) deserve higher severity
- Larger deviations are more severe
- Return ONLY one word: low, medium, or high"""

        try:
            response = self.client.invoke(prompt)
            severity = response.content.strip().lower()
            
            # Validate response
            if severity not in ['low', 'medium', 'high']:
                logger.warning(f"Invalid severity from Ollama: {severity}, defaulting to medium")
                severity = 'medium'
            
            logger.debug(f"Classified {metric} anomaly as {severity}")
            return severity
            
        except Exception as e:
            logger.error(f"Error classifying anomaly with Ollama: {e}")
            return "medium"  # Default fallback
    
    def categorize_recommendation(
        self,
        recommendation_text: str
    ) -> str:
        """Categorize a recommendation using local model.
        
        Args:
            recommendation_text: Recommendation text
            
        Returns:
            Category: sleep, recovery, activity, nutrition, stress, or general
        """
        prompt = f"""Categorize this health recommendation:

"{recommendation_text}"

Categories:
- sleep: Sleep-related recommendations
- recovery: Recovery and rest
- activity: Exercise and physical activity
- nutrition: Diet and nutrition
- stress: Stress management
- general: General health advice

Return ONLY the category name (one word)."""

        try:
            response = self.client.invoke(prompt)
            category = response.content.strip().lower()
            
            valid_categories = ['sleep', 'recovery', 'activity', 'nutrition', 'stress', 'general']
            if category not in valid_categories:
                logger.warning(f"Invalid category from Ollama: {category}, defaulting to general")
                category = 'general'
            
            logger.debug(f"Categorized recommendation as {category}")
            return category
            
        except Exception as e:
            logger.error(f"Error categorizing recommendation with Ollama: {e}")
            return "general"  # Default fallback
    
    def summarize_trends(self, trends: List[str]) -> str:
        """Summarize health trends using local model.
        
        Args:
            trends: List of trend descriptions
            
        Returns:
            Summary text
        """
        if not trends:
            return "No significant trends detected."
        
        trends_text = "\n".join(f"- {t}" for t in trends)
        
        prompt = f"""Summarize these health trends in 1-2 sentences:

{trends_text}

Be concise and highlight the most important patterns."""

        try:
            response = self.client.invoke(prompt)
            summary = response.content.strip()
            logger.debug("Summarized trends with Ollama")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing trends with Ollama: {e}")
            return "Multiple health trends detected in your data."