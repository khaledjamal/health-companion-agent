#streamlit_app.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import pandas as pd
from pathlib import Path

from config import settings
from memory.database import DatabaseManager
from memory.vector_store import VectorStoreManager
from llm.openai_client import OpenAIClient
from agent.graph import HealthAgentGraph
from data.loader import DataLoader

# Page config
st.set_page_config(
    page_title="ARC Health Companion",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager()
    st.session_state.vector_store = VectorStoreManager()
    st.session_state.llm_client = OpenAIClient()
    st.session_state.agent = HealthAgentGraph(
        st.session_state.db,
        st.session_state.vector_store,
        st.session_state.llm_client
    )


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">🏥 ARC Health Companion</p>', unsafe_allow_html=True)
    st.markdown("*Your AI-powered health optimization assistant*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        user_id = st.number_input("User ID", min_value=1, value=1)
        
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔍 Analyze", "📁 Data Management", "📚 Knowledge Base"]
        )
        
        st.divider()
        
        st.caption("ARC Health Companion v1.0")
    
    # Route to selected page
    if page == "📊 Dashboard":
        show_dashboard(user_id)
    elif page == "🔍 Analyze":
        show_analyze(user_id)
    elif page == "📁 Data Management":
        show_data_management(user_id)
    elif page == "📚 Knowledge Base":
        show_knowledge_base()


def show_dashboard(user_id: int):
    """Display the main dashboard.
    
    Args:
        user_id: User identifier
    """
    st.header("📊 Health Dashboard")
    
    # Check if user has data
    records = st.session_state.db.get_historical_records(user_id, days=30)
    
    if not records:
        st.warning("No health data found. Please upload data in the Data Management section.")
        return
    
    # Metrics overview
    latest_record = records[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Sleep Hours",
            f"{latest_record.sleep_hours:.1f}h",
            delta=_calculate_delta(records, 'sleep_hours')
        )
    
    with col2:
        st.metric(
            "HRV",
            f"{latest_record.hrv} ms",
            delta=_calculate_delta(records, 'hrv')
        )
    
    with col3:
        st.metric(
            "Steps",
            f"{latest_record.steps:,}",
            delta=_calculate_delta(records, 'steps')
        )
    
    with col4:
        st.metric(
            "Stress Level",
            f"{latest_record.stress_score}/10",
            delta=_calculate_delta(records, 'stress_score', inverse=True)
        )
    
    st.divider()
    
    # Trend Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sleep & HRV Trends")
        _plot_dual_metric(records, 'sleep_hours', 'hrv', 'Sleep (hrs)', 'HRV (ms)')
    
    with col2:
        st.subheader("Activity Trends")
        _plot_single_metric(records, 'steps', 'Steps')
    
    st.divider()
    
    # Recent Recommendations
    st.subheader("📋 Recent Recommendations")
    recommendations = st.session_state.db.get_recommendations(user_id, days=7)
    
    if recommendations:
        for rec in recommendations[:5]:
            with st.container():
                priority_emoji = {
                    "urgent": "🔴",
                    "high": "🟡",
                    "medium": "🔵",
                    "low": "⚪"
                }.get(rec.priority, "⚪")
                
                st.markdown(f"**{priority_emoji} {rec.title}** ({rec.date})")
                st.caption(rec.description)
                st.divider()
    else:
        st.info("No recommendations yet. Run an analysis to get personalized insights.")


def show_analyze(user_id: int):
    """Show analysis page.
    
    Args:
        user_id: User identifier
    """
    st.header("🔍 Health Analysis")
    
    # Date selector
    analysis_date = st.date_input(
        "Select date to analyze",
        value=date.today(),
        max_value=date.today()
    )
    
    # Run analysis button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        
        with st.spinner("Running agent analysis..."):
            try:
                # Run agent
                final_state = st.session_state.agent.run(user_id, analysis_date)
                
                # Display results
                report = final_state.get("daily_report")
                
                if report:
                    st.success("Analysis complete!")
                    
                    # Overall Score
                    st.subheader("Overall Health Score")
                    score_color = "green" if report.analysis.overall_score >= 80 else "orange" if report.analysis.overall_score >= 60 else "red"
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=report.analysis.overall_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Health Score"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': score_color},
                            'steps': [
                                {'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 80], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary
                    st.subheader("📝 Summary")
                    st.info(report.analysis.summary)
                    
                    # Key Insights
                    if report.key_insights:
                        st.subheader("💡 Key Insights")
                        for insight in report.key_insights:
                            st.markdown(f"- {insight}")
                    
                    # Anomalies
                    if report.analysis.anomalies:
                        st.subheader("⚠️ Detected Issues")
                        for anomaly in report.analysis.anomalies:
                            severity_color = {
                                "high": "error",
                                "medium": "warning",
                                "low": "info"
                            }.get(anomaly.severity.value, "info")
                            
                            if severity_color == "error":
                                st.error(anomaly.description)
                            elif severity_color == "warning":
                                st.warning(anomaly.description)
                            else:
                                st.info(anomaly.description)
                    
                    # Recommendations
                    if report.recommendations:
                        st.subheader("🎯 Personalized Recommendations")
                        
                        for i, rec in enumerate(report.recommendations, 1):
                            with st.expander(f"{i}. {rec.title} ({rec.priority.value} priority)"):
                                st.markdown(f"**Description:** {rec.description}")
                                st.markdown(f"**Category:** {rec.category.value}")
                                st.markdown(f"**Rationale:** {rec.rationale}")
                                st.markdown(f"**Expected Impact:** {rec.expected_impact}")
                                st.markdown(f"**Confidence:** {rec.confidence:.0%}")
                    
                    # Agent Steps (expandable)
                    with st.expander("🤖 Agent Execution Steps"):
                        for step in final_state.get("agent_steps", []):
                            st.text(f"• {step}")
                    
                else:
                    st.error("Analysis failed. No report generated.")
                    
                    if final_state.get("errors"):
                        st.error("Errors:")
                        for error in final_state["errors"]:
                            st.text(f"- {error}")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")


def show_data_management(user_id: int):
    """Show data management page.
    
    Args:
        user_id: User identifier
    """
    st.header("📁 Data Management")
    
    tab1, tab2 = st.tabs(["📤 Upload Data", "📊 View Data"])
    
    with tab1:
        st.subheader("Upload Health Data")
        
        uploaded_file = st.file_uploader(
            "Choose a JSON or CSV file",
            type=['json', 'csv'],
            help="Upload health metrics data"
        )
        
        if uploaded_file:
            try:
                # Save uploaded file temporarily
                temp_path = Path(f"temp_{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load data
                if uploaded_file.name.endswith('.json'):
                    metrics = DataLoader.load_from_json(temp_path)
                    metrics.user_id = user_id
                    st.session_state.db.save_health_record(metrics)
                    st.success(f"✅ Loaded 1 health record")
                    
                elif uploaded_file.name.endswith('.csv'):
                    metrics_list = DataLoader.load_from_csv(temp_path)
                    for metrics in metrics_list:
                        metrics.user_id = user_id
                        st.session_state.db.save_health_record(metrics)
                    st.success(f"✅ Loaded {len(metrics_list)} health records")
                
                # Clean up
                temp_path.unlink()
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    with tab2:
        st.subheader("Health Records")
        
        days = st.slider("Days to display", 7, 90, 30)
        records = st.session_state.db.get_historical_records(user_id, days=days)
        
        if records:
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    'Date': record.date,
                    'Sleep (hrs)': record.sleep_hours,
                    'Sleep Quality': record.sleep_quality,
                    'HRV': record.hrv,
                    'RHR': record.resting_heart_rate,
                    'Steps': record.steps,
                    'Active (min)': record.active_minutes,
                    'Stress': record.stress_score
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "health_data.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.info("No health records found.")


def show_knowledge_base():
    """Show knowledge base management page."""
    st.header("📚 Knowledge Base")
    
    # Get collection info
    info = st.session_state.vector_store.get_collection_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Documents", info['total_documents'])
    
    with col2:
        st.metric("Unique Topics", info['unique_topics'])
    
    st.divider()
    
    # Display topics
    if info['topics']:
        st.subheader("Available Topics")
        for topic in info['topics']:
            st.markdown(f"- {topic}")
    
    st.divider()
    
    # Test retrieval
    st.subheader("🔍 Test Knowledge Retrieval")
    
    query = st.text_input("Enter a health-related query")
    
    if query:
        docs = st.session_state.vector_store.retrieve(query, top_k=3)
        
        if docs:
            st.success(f"Found {len(docs)} relevant documents")
            
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Document {i}: {doc.topic} (Relevance: {doc.relevance_score:.2f})"):
                    st.text(doc.content)
        else:
            st.warning("No relevant documents found")


def _calculate_delta(records, metric: str, inverse: bool = False):
    """Calculate metric change from previous record.
    
    Args:
        records: List of health records
        metric: Metric name
        inverse: If True, negative change is good (e.g., stress)
        
    Returns:
        Delta string for st.metric
    """
    if len(records) < 2:
        return None
    
    current = getattr(records[0], metric)
    previous = getattr(records[1], metric)
    
    delta = current - previous
    
    if inverse:
        delta = -delta
    
    return f"{delta:+.1f}" if isinstance(delta, float) else f"{delta:+d}"


def _plot_dual_metric(records, metric1: str, metric2: str, label1: str, label2: str):
    """Plot two metrics on dual y-axes.
    
    Args:
        records: List of health records
        metric1: First metric name
        metric2: Second metric name
        label1: First metric label
        label2: Second metric label
    """
    dates = [r.date for r in reversed(records)]
    values1 = [getattr(r, metric1) for r in reversed(records)]
    values2 = [getattr(r, metric2) for r in reversed(records)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=values1,
        name=label1,
        mode='lines+markers',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=values2,
        name=label2,
        mode='lines+markers',
        line=dict(color='#764ba2', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        yaxis=dict(title=label1),
        yaxis2=dict(title=label2, overlaying='y', side='right'),
        hovermode='x unified',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _plot_single_metric(records, metric: str, label: str):
    """Plot a single metric over time.
    
    Args:
        records: List of health records
        metric: Metric name
        label: Metric label
    """
    dates = [r.date for r in reversed(records)]
    values = [getattr(r, metric) for r in reversed(records)]
    
    fig = px.line(
        x=dates, y=values,
        labels={'x': 'Date', 'y': label},
        markers=True
    )
    
    fig.update_traces(line_color='#667eea', line_width=2)
    fig.update_layout(hovermode='x unified', height=300)
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()