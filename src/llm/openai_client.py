#openai_client.py

import instructor
from openai import OpenAI
from typing import List, Type, TypeVar
from pydantic import BaseModel
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings

T = TypeVar('T', bound=BaseModel)


class OpenAIClient:
    """OpenAI client with Instructor for structured outputs."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize OpenAI client with Instructor.
        
        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        
        # Initialize OpenAI client
        base_client = OpenAI(api_key=self.api_key)
        
        # Wrap with Instructor for structured outputs
        self.client = instructor.from_openai(base_client)
        
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def create_structured_completion(
        self,
        messages: List[dict],
        response_model: Type[T],
        temperature: float = None,
        max_tokens: int = 2000
    ) -> T:
        """Create a completion with structured output.
        
        Args:
            messages: List of message dictionaries
            response_model: Pydantic model for response structure
            temperature: Sampling temperature (defaults to settings)
            max_tokens: Maximum tokens in response
            
        Returns:
            Structured response matching response_model
        """
        if temperature is None:
            temperature = settings.openai_temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.debug(f"Structured completion created: {response_model.__name__}")
            return response
            
        except Exception as e:
            logger.error(f"Error creating structured completion: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def create_completion(
        self,
        messages: List[dict],
        temperature: float = None,
        max_tokens: int = 2000
    ) -> str:
        """Create a standard text completion.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Text response
        """
        if temperature is None:
            temperature = settings.openai_temperature
        
        try:
            # Use base OpenAI client for non-structured calls
            base_client = OpenAI(api_key=self.api_key)
            response = base_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Text completion created: {len(content)} chars")
            return content
            
        except Exception as e:
            logger.error(f"Error creating completion: {e}")
            raise
    
    def analyze_health_data(
        self,
        analysis_summary: str,
        anomalies: List[str],
        trends: List[str],
        correlations: List[str]
    ) -> str:
        """Analyze health data and provide insights.
        
        Args:
            analysis_summary: Summary of the analysis
            anomalies: List of anomaly descriptions
            trends: List of trend descriptions
            correlations: List of correlation insights
            
        Returns:
            AI-generated health insights
        """
        messages = [
            {
                "role": "system",
                "content": """You are an expert health and wellness coach. Analyze the provided health data and provide actionable insights.
Focus on:
- Interpreting the data in context
- Explaining what the patterns mean for overall health
- Identifying root causes of issues
- Being encouraging but honest

Keep your response concise (2-3 paragraphs)."""
            },
            {
                "role": "user",
                "content": f"""Analysis Summary: {analysis_summary}

Detected Anomalies:
{chr(10).join(f"- {a}" for a in anomalies) if anomalies else "None"}

Trends:
{chr(10).join(f"- {t}" for t in trends) if trends else "None"}

Correlations:
{chr(10).join(f"- {c}" for c in correlations) if correlations else "None"}

Provide your expert analysis and key insights."""
            }
        ]
        
        return self.create_completion(messages, temperature=0.7)
    
    def generate_reasoning(
        self,
        context: str,
        question: str
    ) -> str:
        """Generate reasoning about health data.
        
        Args:
            context: Context information
            question: Question to reason about
            
        Returns:
            Reasoning response
        """
        messages = [
            {
                "role": "system",
                "content": "You are a health data analyst. Provide clear, logical reasoning based on the given data."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}\n\nProvide your reasoning:"
            }
        ]
        
        return self.create_completion(messages, temperature=0.5)