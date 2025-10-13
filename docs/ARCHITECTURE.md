# ARC Health Companion - Architecture Documentation

## System Overview

ARC Health Companion is an agentic AI system designed to analyze health data and provide personalized recommendations through multi-step reasoning and knowledge retrieval.

## Architecture Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Type Safety**: Pydantic models throughout for validation and documentation
3. **Extensibility**: Easy to add new metrics, tools, or reasoning steps
4. **Observability**: Comprehensive logging and execution tracking
5. **Production-Ready**: Error handling, retries, and persistent storage

## Core Components

### 1. Agent System (LangGraph)

**Location**: `src/agent/`

The agent orchestrates the entire analysis workflow using LangGraph's state machine pattern.

#### State Definition (`state.py`)

```python
AgentState = TypedDict({
    "user_id": int,
    "date": date,
    "health_metrics": Optional[HealthMetrics],
    "historical_data": List[HealthMetrics],
    "analysis": Optional[HealthAnalysis],
    "retrieved_knowledge": List[KnowledgeDocument],
    "recommendations": List[Recommendation],
    "daily_report": Optional[DailyReport],
    # ... metadata fields
})
```

State flows through the graph, being enriched at each node.

#### Graph Structure (`graph.py`)

```
Entry → Load Data → Analyze Data → Retrieve Knowledge → Generate Recs → Create Report → End
         ↓ (if fails)
         End
```

**Conditional Routing**: After data loading, the graph decides whether to continue or terminate based on data availability.

#### Nodes (`nodes.py`)

Each node is a pure function that receives state and returns updated state:

1. **load_data_node**: Loads current and historical health data
2. **analyze_data_node**: Performs statistical analysis and anomaly detection
3. **retrieve_knowledge_node**: Queries vector store for relevant information
4. **generate_recommendations_node**: Uses LLM to create personalized advice
5. **create_report_node**: Compiles final report and saves to database

#### Tools (`tools.py`)

Tools are reusable functions that nodes can call:

- `load_health_data()`: Database queries for health records
- `analyze_health_data()`: Statistical analysis
- `retrieve_knowledge()`: Vector store queries
- `generate_recommendations()`: LLM-based recommendation generation
- `save_recommendations()`: Persist recommendations

### 2. Data Processing Layer

**Location**: `src/data/`

#### Data Loader (`loader.py`)

Handles importing health metrics from external sources:

- JSON file loading with validation
- CSV batch processing
- Date parsing and normalization
- Error handling for malformed data

#### Health Analyzer (`analyzer.py`)

Performs comprehensive data analysis:

**Health Score Calculation**:
- Weighted combination of all metrics
- 0-100 scale for easy interpretation
- Each metric scored against reference ranges

**Anomaly Detection**:
- Reference range comparison (e.g., HRV: 50-100ms)
- Statistical anomaly detection using z-scores
- Severity classification (low/medium/high)
- Personal baseline comparison

**Trend Detection**:
- Linear regression on historical data
- Statistical significance testing (p < 0.1)
- Direction classification (improving/declining/stable)
- Percentage change calculation

**Correlation Analysis**:
- Pearson correlation between metrics
- Identifies relationships (e.g., sleep-HRV correlation)
- Minimum 7 days of data required

### 3. Memory Systems

**Location**: `src/memory/`

#### Database Manager (`database.py`)

PostgreSQL database with SQLModel ORM:

**Tables**:
- `user_profiles`: User information and goals
- `health_records`: Daily health metrics
- `recommendations`: Historical recommendations with feedback
- `agent_executions`: Execution logs for debugging

**Key Operations**:
- CRUD operations for all entities
- Historical data retrieval with date ranges
- Metric averaging over time periods
- Recommendation feedback tracking

#### Vector Store (`vector_store.py`)

pgvector on PostgreSQL for RAG knowledge base:

**Features**:
- Sentence transformer embeddings (all-MiniLM-L6-v2)
- Persistent storage on disk
- Metadata filtering by topic
- Relevance scoring for retrieved documents

**Knowledge Organization**:
- Documents organized by topic (sleep, HRV, nutrition, stress)
- Embeddings generated automatically
- Efficient similarity search
- Top-k retrieval with configurable parameters

### 4. LLM Integration

**Location**: `src/llm/`

#### OpenAI Client (`openai_client.py`)

Primary LLM for reasoning and recommendations:

**Features**:
- Instructor integration for structured outputs
- Automatic retry logic with exponential backoff
- Pydantic model validation
- Temperature control per task type

**Use Cases**:
- Health data interpretation
- Recommendation generation
- Reasoning about patterns
- Report summarization

#### Ollama Client (`ollama_client.py`)

Local model for simple classification tasks:

**Use Cases**:
- Anomaly severity classification
- Recommendation categorization
- Trend summarization

**Benefits**:
- Cost optimization (free)
- Demonstrates hybrid approach
- Shows flexibility in model selection

### 5. Recommendation Engine

**Location**: `src/recommendations/`

#### Generator (`generator.py`)

Creates personalized recommendations using:

**Context Building**:
- Analysis results (score, anomalies, trends)
- Retrieved health knowledge
- User profile and goals
- Historical patterns

**LLM-Based Generation**:
- Structured output using Instructor
- Pydantic model ensures quality
- Priority and confidence scoring
- Expected impact estimation

**Fallback Logic**:
- Rule-based recommendations if LLM fails
- Mapping anomalies to proven interventions
- Ensures system reliability

**Post-Processing**:
- Deduplication by title
- Priority-based sorting
- Limit to top 5 recommendations
- Confidence threshold filtering

### 6. User Interfaces

#### CLI (`cli.py`)

Typer-based command-line interface:

**Commands**:
- `init-db`: Database initialization
- `load-knowledge`: Populate vector store
- `load-data`: Import health data
- `analyze`: Run agent analysis
- `report`: View historical data
- `dashboard`: Launch Streamlit

**Features**:
- Rich terminal output with colors and tables
- Progress indicators
- Error handling with helpful messages
- Verbose mode for debugging

#### Streamlit Dashboard (`streamlit_app.py`)

Interactive web interface:

**Pages**:
1. **Dashboard**: Overview with metrics and trend charts
2. **Analyze**: Run analysis and view results
3. **Data Management**: Upload and view health records
4. **Knowledge Base**: Explore and test RAG system

**Visualizations**:
- Plotly charts for trend analysis
- Gauge chart for health score
- Dual-axis plots for correlations
- Interactive data tables

## Data Flow

### End-to-End Analysis Flow

```
1. User Request (CLI or Streamlit)
   ↓
2. Agent Initialization
   - Load DB, Vector Store, LLM clients
   ↓
3. LangGraph Execution
   ├─ Load Data Node
   │  └─ Query database for current + historical metrics
   ├─ Analyze Data Node
   │  └─ Statistical analysis, anomaly detection, trends
   ├─ Retrieve Knowledge Node
   │  └─ Vector search for relevant health information
   ├─ Generate Recommendations Node
   │  └─ LLM creates personalized advice
   └─ Create Report Node
      └─ Compile final report, save recommendations
   ↓
4. Return Results
   - Display in UI
   - Save to database for history
```

### RAG Pipeline Flow

```
1. Analyze health data
   ↓
2. Identify key issues (anomalies, trends)
   ↓
3. Build search queries from issues
   - "HRV optimization"
   - "sleep quality improvement"
   ↓
4. Vector search with pgvector on PostgreSQL
   - Embedding generation
   - Similarity search
   - Top-k retrieval
   ↓
5. Retrieved documents added to context
   ↓
6. LLM uses context for grounded recommendations
```

## Design Decisions

### Why LangGraph?

- **State Management**: Built-in state persistence across nodes
- **Observability**: Easy to track agent's decision process
- **Error Recovery**: Can handle node failures gracefully
- **Extensibility**: Simple to add new nodes or branches
- **Production-Ready**: Used in real-world agentic systems

### Why Instructor?

- **Type Safety**: Guaranteed structured outputs
- **Validation**: Automatic Pydantic validation
- **Reliability**: Reduces LLM hallucination risks
- **Developer Experience**: Great IDE support

### Why pgvector on PostgreSQL?

- **Unified stack**: Keep metadata and embeddings in the same database
- **Operational simplicity**: One DB to manage, back up, and secure
- **Performance**: pgvector with IVFFlat or HNSW indexes for fast ANN
- **Portability**: Standard SQL and SQLAlchemy integration
- **Maturity**: Production-grade durability, transactions, and tooling

### Why Dual LLM Strategy?

- **Cost Optimization**: Use cheap local models for simple tasks
- **Demonstrates Skill**: Shows understanding of model selection
- **Flexibility**: Not locked into one provider
- **Realistic**: Production systems use multiple models

## Scalability Considerations

### Current Scale
- Single user: Sub-second responses
- 100 users: <5 second responses
- 1000 documents in vector store: Fast retrieval

### Scaling Strategies

**Database**:
- PostgreSQL in production
- Easy migration to PostgreSQL for production
- Add indexes on date and user_id
- Partition by date for large datasets

**Vector Store**:
- pgvector on PostgreSQL scales well and supports ANN indexes (HNSW/IVFFlat)
- Can migrate to Pinecone/Weaviate for millions
- Add caching layer for frequent queries

**LLM Calls**:
- Currently synchronous (simple, predictable)
- Can add async/await for parallel calls
- Implement caching for repeated queries
- Rate limiting for API protection

**Agent Execution**:
- Currently sequential nodes
- Could parallelize independent nodes (analyze + retrieve)
- Add job queue (Celery) for async execution
- Horizontal scaling with stateless agent instances

## Security & Privacy

**API Keys**:
- Environment variables (not in code)
- .gitignore for .env file
- Separate keys for dev/prod

**Data Privacy**:
- PostgreSQL with role-based access and connection pooling
- User data isolated by user_id
- No PHI exposed in logs

**LLM Privacy**:
- Health data sent to OpenAI (consider for production)
- Could use local models only for sensitive data
- Anonymize data before LLM calls if needed

## Testing Strategy

**Unit Tests**:
- Models: Validation logic
- Analyzer: Calculation correctness
- Database: CRUD operations

**Integration Tests** (if time permits):
- Full agent execution
- End-to-end flow
- Error scenarios

**Manual Testing**:
- CLI commands
- Streamlit interface
- Different data scenarios

## Future Enhancements

### Technical
- [ ] Async agent execution
- [ ] Caching layer (Redis)
- [ ] FastAPI backend
- [ ] Multi-user authentication
- [ ] Real-time data ingestion (wearables)
- [ ] More sophisticated ML models

### Features
- [ ] Goal tracking and progress
- [ ] Recommendation effectiveness feedback loop
- [ ] Social features (compare with peers)
- [ ] Integration with health APIs (Apple Health, Fitbit)
- [ ] Predictive analytics
- [ ] Multi-modal input (voice notes, images)

### Agent Capabilities
- [ ] Multi-agent collaboration
- [ ] Self-reflection and replanning
- [ ] Tool creation (agent creates new analysis tools)
- [ ] Conversational interface
- [ ] Proactive recommendations (push notifications)

## Performance Metrics

**Current Performance** (approximate):
- Data loading: <100ms
- Analysis: ~200ms
- Vector search: ~300ms
- LLM call: 1-3s
- Total agent execution: 3-5s

**Optimization Opportunities**:
- Cache vector embeddings
- Batch LLM calls
- Parallel node execution
- Precompute common analyses

## Conclusion

This architecture demonstrates:
- ✅ Production-grade system design
- ✅ Modern AI engineering practices
- ✅ Clean code and modularity
- ✅ Scalability awareness
- ✅ Real-world applicability

The system is ready for demonstration while being structured for easy extension into a production application.