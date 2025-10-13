#state.py

from typing import TypedDict, List, Optional, Annotated
from datetime import date
import operator

from ..models import (
    HealthMetrics,
    HealthAnalysis,
    Recommendation,
    KnowledgeDocument,
    DailyReport
)


class AgentState(TypedDict):
    """State passed through the LangGraph agent.
    
    This state is modified by each node in the graph.
    """
    # Input
    user_id: int
    date: date
    health_metrics: Optional[HealthMetrics]
    
    # Processing
    historical_data: List[HealthMetrics]
    analysis: Optional[HealthAnalysis]
    retrieved_knowledge: List[KnowledgeDocument]
    recommendations: List[Recommendation]
    
    # LLM reasoning
    reasoning_steps: Annotated[List[str], operator.add]  # Accumulate reasoning
    
    # Output
    daily_report: Optional[DailyReport]
    
    # Metadata
    agent_steps: Annotated[List[str], operator.add]  # Track execution steps
    errors: Annotated[List[str], operator.add]  # Track errors
    
    # Control flow
    should_continue: bool
    iteration_count: int