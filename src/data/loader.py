#loader.py

import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from datetime import date, datetime
from loguru import logger

from ..models import HealthMetrics


class DataLoader:
    """Loads health data from JSON and CSV files."""
    
    @staticmethod
    def load_from_json(filepath: Union[str, Path]) -> HealthMetrics:
        """Load health metrics from a JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            HealthMetrics object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or missing required fields
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert date string to date object if needed
            if isinstance(data.get('date'), str):
                data['date'] = datetime.fromisoformat(data['date']).date()
            
            metrics = HealthMetrics(**data)
            logger.info(f"Loaded health metrics from {filepath}")
            return metrics
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.error(f"Error loading health metrics: {e}")
            raise
    
    @staticmethod
    def load_from_csv(filepath: Union[str, Path]) -> List[HealthMetrics]:
        """Load multiple health metrics from a CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of HealthMetrics objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            metrics_list = []
            for _, row in df.iterrows():
                try:
                    metrics = HealthMetrics(**row.to_dict())
                    metrics_list.append(metrics)
                except Exception as e:
                    logger.warning(f"Skipping invalid row: {e}")
                    continue
            
            logger.info(f"Loaded {len(metrics_list)} health metrics from {filepath}")
            return metrics_list
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise ValueError(f"Invalid CSV format: {e}")
    
    @staticmethod
    def load_multiple_json(directory: Union[str, Path]) -> List[HealthMetrics]:
        """Load all JSON files from a directory.
        
        Args:
            directory: Path to directory containing JSON files
            
        Returns:
            List of HealthMetrics objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        metrics_list = []
        for json_file in directory.glob("*.json"):
            try:
                metrics = DataLoader.load_from_json(json_file)
                metrics_list.append(metrics)
            except Exception as e:
                logger.warning(f"Skipping {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(metrics_list)} health metrics from {directory}")
        return metrics_list
    
    @staticmethod
    def save_to_json(metrics: HealthMetrics, filepath: Union[str, Path]) -> None:
        """Save health metrics to a JSON file.
        
        Args:
            metrics: HealthMetrics object
            filepath: Path to save JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = metrics.model_dump()
        # Convert date to string for JSON serialization
        data['date'] = data['date'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved health metrics to {filepath}")
    
    @staticmethod
    def save_to_csv(metrics_list: List[HealthMetrics], filepath: Union[str, Path]) -> None:
        """Save multiple health metrics to a CSV file.
        
        Args:
            metrics_list: List of HealthMetrics objects
            filepath: Path to save CSV file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = [m.model_dump() for m in metrics_list]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved {len(metrics_list)} health metrics to {filepath}")