#nodes.py

from typing import Dict, Any
from loguru import logger

from .state import AgentState
from .tools import AgentTools
from ..models import DailyReport


class AgentNodes:
    """Node functions for the LangGraph agent."""
    
    def __init__(self, tools: AgentTools):
        """Initialize agent nodes.
        
        Args:
            tools: Agent tools
        """
        self.tools = tools
        logger.info("Agent nodes initialized")
    
    def load_data_node(self, state: AgentState) -> Dict[str, Any]:
        """Load health data for analysis.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Loading health data")
        
        try:
            # Load current day's data
            health_metrics = self.tools.load_health_data(
                state["user_id"],
                state["date"]
            )
            
            if not health_metrics:
                return {
                    "health_metrics": None,
                    "agent_steps": ["Failed to load health data - no data found"],
                    "errors": ["No health data found for the specified date"],
                    "should_continue": False
                }
            
            # Load historical data for context
            historical_data = self.tools.load_historical_data(
                state["user_id"],
                days=30,
                end_date=state["date"]
            )
            
            logger.info(f"Loaded current and {len(historical_data)} historical records")
            
            return {
                "health_metrics": health_metrics,
                "historical_data": historical_data,
                "agent_steps": [f"Loaded health data for {state['date']}"],
                "should_continue": True,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Error in load_data_node: {e}")
            return {
                "agent_steps": [f"Error loading data: {str(e)}"],
                "errors": [str(e)],
                "should_continue": False
            }
    
    def analyze_data_node(self, state: AgentState) -> Dict[str, Any]:
        """Analyze health data for patterns and anomalies.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Analyzing health data")
        
        try:
            if not state.get("health_metrics"):
                return {
                    "agent_steps": ["Skipped analysis - no health data"],
                    "errors": ["No health metrics to analyze"],
                    "should_continue": False
                }
            
            # Perform analysis
            analysis = self.tools.analyze_health_data(
                state["health_metrics"],
                state.get("historical_data", [])
            )
            
            logger.info(f"Analysis complete - Score: {analysis.overall_score}/100")
            
            # Build reasoning about the analysis
            reasoning = f"Health score is {analysis.overall_score}/100. "
            
            if analysis.anomalies:
                reasoning += f"Detected {len(analysis.anomalies)} anomalies. "
            
            if analysis.trends:
                reasoning += f"Identified {len(analysis.trends)} trends. "
            
            return {
                "analysis": analysis,
                "agent_steps": [f"Analyzed health data - Score: {analysis.overall_score}/100"],
                "reasoning_steps": [reasoning],
                "should_continue": True,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_data_node: {e}")
            return {
                "agent_steps": [f"Error analyzing data: {str(e)}"],
                "errors": [str(e)],
                "should_continue": False
            }
    
    def retrieve_knowledge_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant knowledge for recommendations.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Retrieving knowledge")
        
        try:
            analysis = state.get("analysis")
            if not analysis:
                return {
                    "agent_steps": ["Skipped knowledge retrieval - no analysis"],
                    "should_continue": True,
                    "iteration_count": state.get("iteration_count", 0) + 1
                }
            
            # Build queries based on anomalies
            queries = []
            
            # Query for each high/medium severity anomaly
            for anomaly in analysis.anomalies:
                if anomaly.severity.value in ["high", "medium"]:
                    queries.append(f"{anomaly.metric.replace('_', ' ')} optimization")
            
            # Add general health query if no specific issues
            if not queries:
                queries.append("general health optimization")
            
            # Retrieve knowledge for each query (limit to top 2 queries)
            all_documents = []
            for query in queries[:2]:
                docs = self.tools.retrieve_knowledge(query, top_k=2)
                all_documents.extend(docs)
            
            # Remove duplicates and limit
            unique_docs = []
            seen_content = set()
            for doc in all_documents:
                if doc.content not in seen_content:
                    seen_content.add(doc.content)
                    unique_docs.append(doc)
                    if len(unique_docs) >= 3:
                        break
            
            logger.info(f"Retrieved {len(unique_docs)} unique knowledge documents")
            
            reasoning = f"Retrieved {len(unique_docs)} relevant knowledge documents for context."
            
            return {
                "retrieved_knowledge": unique_docs,
                "agent_steps": [f"Retrieved {len(unique_docs)} knowledge documents"],
                "reasoning_steps": [reasoning],
                "should_continue": True,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve_knowledge_node: {e}")
            return {
                "agent_steps": [f"Error retrieving knowledge: {str(e)}"],
                "errors": [str(e)],
                "retrieved_knowledge": [],
                "should_continue": True,  # Continue anyway with empty knowledge
                "iteration_count": state.get("iteration_count", 0) + 1
            }
    
    def generate_recommendations_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate personalized recommendations.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Generating recommendations")
        
        try:
            analysis = state.get("analysis")
            if not analysis:
                return {
                    "agent_steps": ["Skipped recommendations - no analysis"],
                    "errors": ["No analysis available for recommendations"],
                    "should_continue": False
                }
            
            # Get user context
            user_context = self.tools.get_user_context(state["user_id"])
            
            # Generate recommendations
            recommendations = self.tools.generate_recommendations(
                analysis,
                state.get("retrieved_knowledge", []),
                user_context
            )
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            reasoning = f"Generated {len(recommendations)} personalized recommendations based on analysis and retrieved knowledge."
            
            return {
                "recommendations": recommendations,
                "agent_steps": [f"Generated {len(recommendations)} recommendations"],
                "reasoning_steps": [reasoning],
                "should_continue": True,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Error in generate_recommendations_node: {e}")
            return {
                "agent_steps": [f"Error generating recommendations: {str(e)}"],
                "errors": [str(e)],
                "recommendations": [],
                "should_continue": False
            }
    
    def create_report_node(self, state: AgentState) -> Dict[str, Any]:
        """Create final daily report.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Creating daily report")
        
        try:
            analysis = state.get("analysis")
            recommendations = state.get("recommendations", [])
            
            if not analysis:
                return {
                    "agent_steps": ["Cannot create report - no analysis"],
                    "errors": ["Missing analysis for report creation"],
                    "should_continue": False
                }
            
            # Extract key insights
            key_insights = []
            
            # Add score insight
            if analysis.overall_score >= 80:
                key_insights.append(f"Excellent health score of {analysis.overall_score}/100")
            elif analysis.overall_score >= 60:
                key_insights.append(f"Good health score of {analysis.overall_score}/100 with improvement opportunities")
            else:
                key_insights.append(f"Health score of {analysis.overall_score}/100 - focus areas identified")
            
            # Add anomaly insights
            high_priority_anomalies = [a for a in analysis.anomalies if a.severity.value == "high"]
            if high_priority_anomalies:
                key_insights.append(f"{len(high_priority_anomalies)} critical areas need immediate attention")
            
            # Add trend insights
            improving_trends = [t for t in analysis.trends if t.direction.value == "improving"]
            if improving_trends:
                key_insights.append(f"{len(improving_trends)} metrics showing positive trends")
            
            # Extract top 3 action items from recommendations
            action_items = []
            for rec in recommendations[:3]:
                action_items.append(f"{rec.title}: {rec.description}")
            
            # Create report
            report = DailyReport(
                date=state["date"],
                user_id=state["user_id"],
                analysis=analysis,
                recommendations=recommendations,
                key_insights=key_insights,
                action_items=action_items
            )
            
            # Save recommendations to database
            self.tools.save_recommendations(
                state["user_id"],
                state["date"],
                recommendations
            )
            
            logger.info("Daily report created successfully")
            
            return {
                "daily_report": report,
                "agent_steps": ["Created daily health report"],
                "should_continue": False,  # End execution
                "iteration_count": state.get("iteration_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Error in create_report_node: {e}")
            return {
                "agent_steps": [f"Error creating report: {str(e)}"],
                "errors": [str(e)],
                "should_continue": False
            }