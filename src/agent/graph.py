#graph.py

from langgraph.graph import StateGraph, END
from typing import Literal
from loguru import logger
from datetime import datetime

from .state import AgentState
from .nodes import AgentNodes
from .tools import AgentTools
from ..memory.database import DatabaseManager
from ..memory.vector_store import VectorStoreManager
from ..llm.openai_client import OpenAIClient


class HealthAgentGraph:
    """LangGraph-based health agent."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_store: VectorStoreManager,
        llm_client: OpenAIClient
    ):
        """Initialize the health agent graph.
        
        Args:
            db_manager: Database manager
            vector_store: Vector store manager
            llm_client: LLM client
        """
        self.db = db_manager
        self.vector_store = vector_store
        self.llm_client = llm_client
        
        # Initialize tools and nodes
        self.tools = AgentTools(db_manager, vector_store, llm_client)
        self.nodes = AgentNodes(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("Health agent graph initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Compiled state graph
        """
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("load_data", self.nodes.load_data_node)
        workflow.add_node("analyze_data", self.nodes.analyze_data_node)
        workflow.add_node("retrieve_knowledge", self.nodes.retrieve_knowledge_node)
        workflow.add_node("generate_recommendations", self.nodes.generate_recommendations_node)
        workflow.add_node("create_report", self.nodes.create_report_node)
        
        # Define edges
        workflow.set_entry_point("load_data")
        
        # Conditional edge after load_data
        workflow.add_conditional_edges(
            "load_data",
            self._should_continue_after_load,
            {
                "continue": "analyze_data",
                "end": END
            }
        )
        
        # Linear flow after analysis
        workflow.add_edge("analyze_data", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "create_report")
        workflow.add_edge("create_report", END)
        
        # Compile graph
        app = workflow.compile()
        
        logger.info("Agent graph compiled successfully")
        return app
    
    def _should_continue_after_load(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if agent should continue after loading data.
        
        Args:
            state: Current state
            
        Returns:
            Next step: "continue" or "end"
        """
        if not state.get("should_continue", True):
            logger.warning("Stopping execution - data load failed")
            return "end"
        
        if not state.get("health_metrics"):
            logger.warning("Stopping execution - no health metrics")
            return "end"
        
        return "continue"
    
    def run(self, user_id: int, target_date) -> AgentState:
        """Run the agent for a specific user and date.
        
        Args:
            user_id: User identifier
            target_date: Date to analyze
            
        Returns:
            Final agent state with results
        """
        from datetime import date
        
        # Convert date if needed
        if isinstance(target_date, str):
            target_date = datetime.fromisoformat(target_date).date()
        
        logger.info(f"Running health agent for user {user_id} on {target_date}")
        
        # Initialize state
        initial_state = AgentState(
            user_id=user_id,
            date=target_date,
            health_metrics=None,
            historical_data=[],
            analysis=None,
            retrieved_knowledge=[],
            recommendations=[],
            reasoning_steps=[],
            daily_report=None,
            agent_steps=[],
            errors=[],
            should_continue=True,
            iteration_count=0
        )
        
        # Execute graph
        start_time = datetime.utcnow()
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Log execution
            success = final_state.get("daily_report") is not None
            self.db.log_agent_execution(
                user_id=user_id,
                steps_taken=final_state.get("agent_steps", []),
                duration_seconds=duration,
                success=success,
                error_message="; ".join(final_state.get("errors", [])) if final_state.get("errors") else None
            )
            
            logger.info(f"Agent execution completed in {duration:.2f}s - Success: {success}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            self.db.log_agent_execution(
                user_id=user_id,
                steps_taken=initial_state.get("agent_steps", []),
                duration_seconds=duration,
                success=False,
                error_message=str(e)
            )
            
            raise
    
    def visualize(self, output_path: str = "docs/agent_graph.png") -> None:
        """Visualize the agent graph (requires graphviz).
        
        Args:
            output_path: Path to save visualization
        """
        try:
            from IPython.display import Image, display
            
            # Get graph visualization
            graph_image = self.graph.get_graph().draw_mermaid_png()
            
            # Save to file
            with open(output_path, "wb") as f:
                f.write(graph_image)
            
            logger.info(f"Graph visualization saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not visualize graph: {e}")