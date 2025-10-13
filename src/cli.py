#cli.py

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import json

from .config import settings
from .memory.database import DatabaseManager
from .memory.vector_store import VectorStoreManager
from .llm.openai_client import OpenAIClient
from .agent.graph import HealthAgentGraph
from .data.loader import DataLoader
from .models import HealthMetrics, KnowledgeDocument

app = typer.Typer(help="ARC Health Companion CLI")
console = Console()


@app.command()
def init_db():
    """Initialize the database and create tables."""
    console.print("[bold blue]Initializing database...[/bold blue]")
    
    db = DatabaseManager()
    db.create_tables()
    
    console.print("[bold green]✓ Database initialized successfully[/bold green]")
    
    # Create default user if none exists
    user = db.get_user(1)
    if not user:
        user = db.create_user(
            name="Demo User",
            age=30,
            goals="Improve sleep quality and increase HRV"
        )
        console.print(f"[bold green]✓ Created default user: {user.name}[/bold green]")


@app.command()
def load_knowledge(directory: str = typer.Option("data/knowledge_base", help="Directory containing knowledge documents")):
    """Load knowledge base documents into vector store."""
    console.print(f"[bold blue]Loading knowledge base from {directory}...[/bold blue]")
    
    vector_store = VectorStoreManager()
    
    # Clear existing data
    vector_store.clear_collection()
    console.print("[yellow]Cleared existing knowledge base[/yellow]")
    
    # Load documents
    with Progress() as progress:
        task = progress.add_task("[cyan]Loading documents...", total=100)
        
        count = vector_store.load_from_directory(directory)
        
        progress.update(task, completed=100)
    
    console.print(f"[bold green]✓ Loaded {count} documents into knowledge base[/bold green]")
    
    # Show collection info
    info = vector_store.get_collection_info()
    console.print(f"[dim]Topics: {', '.join(info['topics'])}[/dim]")


@app.command()
def load_data(
    filepath: str = typer.Argument(..., help="Path to data file"),
    user_id: int = typer.Option(1, "--user-id", help="User identifier")
):
    """Load health data from JSON or CSV file."""
    console.print(f"[bold blue]Loading health data from {filepath}...[/bold blue]")
    
    db = DatabaseManager()
    path = Path(filepath)
    
    try:
        if path.suffix == '.json':
            metrics = DataLoader.load_from_json(filepath)
            metrics.user_id = user_id
            db.save_health_record(metrics)
            console.print(f"[bold green]✓ Loaded 1 health record[/bold green]")
            
        elif path.suffix == '.csv':
            metrics_list = DataLoader.load_from_csv(filepath)
            for metrics in metrics_list:
                metrics.user_id = user_id
                db.save_health_record(metrics)
            console.print(f"[bold green]✓ Loaded {len(metrics_list)} health records[/bold green]")
        else:
            console.print("[bold red]Error: Unsupported file format. Use .json or .csv[/bold red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[bold red]Error loading data: {e}[/bold red]")
        raise typer.Exit(1)



@app.command()
def analyze(
    target_date: str = typer.Argument("today", help="Date (YYYY-MM-DD) or 'today'"),
    user_id: int = typer.Option(1, "--user-id", help="User identifier"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution steps")
):
    """Run health analysis for a specific date."""
    # Parse date
    if target_date.lower() == "today":
        analysis_date = date.today()
    else:
        try:
            analysis_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            console.print("[bold red]Error: Invalid date format. Use YYYY-MM-DD or 'today'[/bold red]")
            raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Running Health Analysis[/bold]\nUser ID: {user_id}\nDate: {analysis_date}",
        border_style="blue"
    ))
    
    # Initialize components
    with console.status("[bold cyan]Initializing agent..."):
        db = DatabaseManager()
        vector_store = VectorStoreManager()
        llm_client = OpenAIClient()
        agent = HealthAgentGraph(db, vector_store, llm_client)
    
    # Run agent
    console.print("\n[bold cyan]Agent Execution:[/bold cyan]")
    
    try:
        final_state = agent.run(user_id, analysis_date)
        
        # Display results
        if verbose and final_state.get("agent_steps"):
            console.print("\n[bold]Execution Steps:[/bold]")
            for step in final_state["agent_steps"]:
                console.print(f"  • {step}")
        
        # Display errors if any
        if final_state.get("errors"):
            console.print("\n[bold red]Errors:[/bold red]")
            for error in final_state["errors"]:
                console.print(f"  ✗ {error}", style="red")
        
        # Display report
        report = final_state.get("daily_report")
        if report:
            _display_report(report)
        else:
            console.print("[bold red]No report generated[/bold red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[bold red]Agent execution failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def report(
    user_id: int = typer.Option(1, "--user-id", help="User identifier"),
    days: int = typer.Option(7, "--days", help="Number of days to look back")
):
    """View recent health records and recommendations."""
    db = DatabaseManager()
    
    console.print(Panel(
        f"[bold]Health Report[/bold]\nUser ID: {user_id}\nLast {days} days",
        border_style="blue"
    ))
    
    # Get recent records
    records = db.get_historical_records(user_id, days=days)
    
    if not records:
        console.print("[yellow]No health records found[/yellow]")
        return
    
    # Display records table
    table = Table(title="Health Metrics", show_header=True)
    table.add_column("Date", style="cyan")
    table.add_column("Sleep (hrs)", justify="right")
    table.add_column("HRV", justify="right")
    table.add_column("RHR", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Stress", justify="right")
    
    for record in records[:10]:  # Show last 10
        table.add_row(
            str(record.date),
            f"{record.sleep_hours:.1f}",
            str(record.hrv),
            str(record.resting_heart_rate),
            f"{record.steps:,}",
            str(record.stress_score)
        )
    
    console.print(table)
    
    # Get recent recommendations
    recommendations = db.get_recommendations(user_id, days=days)
    
    if recommendations:
        console.print(f"\n[bold]Recent Recommendations ({len(recommendations)}):[/bold]")
        for rec in recommendations[:5]:
            priority_color = {
                "urgent": "red",
                "high": "yellow",
                "medium": "blue",
                "low": "dim"
            }.get(rec.priority, "white")
            
            console.print(f"\n[{priority_color}]● {rec.title}[/{priority_color}] ({rec.date})")
            console.print(f"  {rec.description}")


@app.command()
def dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    
    console.print("[bold blue]Launching Streamlit dashboard...[/bold blue]")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py"])
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error launching dashboard: {e}[/bold red]")


def _display_report(report):
    """Display a daily report in the console."""
    console.print("\n" + "="*70)
    console.print(f"[bold cyan]Daily Health Report - {report.date}[/bold cyan]")
    console.print("="*70)
    
    # Overall Score
    score_color = "green" if report.analysis.overall_score >= 80 else "yellow" if report.analysis.overall_score >= 60 else "red"
    console.print(f"\n[bold]Overall Health Score:[/bold] [{score_color}]{report.analysis.overall_score}/100[/{score_color}]")
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]\n{report.analysis.summary}")
    
    # Key Insights
    if report.key_insights:
        console.print("\n[bold]Key Insights:[/bold]")
        for insight in report.key_insights:
            console.print(f"  • {insight}")
    
    # Anomalies
    if report.analysis.anomalies:
        console.print("\n[bold]Detected Issues:[/bold]")
        for anomaly in report.analysis.anomalies:
            severity_color = {
                "high": "red",
                "medium": "yellow",
                "low": "dim"
            }.get(anomaly.severity.value, "white")
            
            console.print(f"  [{severity_color}]⚠ {anomaly.description}[/{severity_color}]")
    
    # Trends
    if report.analysis.trends:
        console.print("\n[bold]Trends:[/bold]")
        for trend in report.analysis.trends:
            direction_symbol = "↑" if trend.direction.value == "improving" else "↓" if trend.direction.value == "declining" else "→"
            console.print(f"  {direction_symbol} {trend.description}")
    
    # Recommendations
    if report.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(report.recommendations, 1):
            priority_color = {
                "urgent": "red",
                "high": "yellow",
                "medium": "blue",
                "low": "dim"
            }.get(rec.priority.value, "white")
            
            console.print(f"\n[bold]{i}. {rec.title}[/bold] ([{priority_color}]{rec.priority.value}[/{priority_color}])")
            console.print(f"   {rec.description}")
            console.print(f"   [dim]Expected Impact: {rec.expected_impact}[/dim]")
    
    # Action Items
    if report.action_items:
        console.print("\n[bold green]Top Action Items:[/bold green]")
        for i, action in enumerate(report.action_items[:3], 1):
            console.print(f"  {i}. {action}")
    
    console.print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    app()