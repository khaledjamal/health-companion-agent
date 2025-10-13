#analyzer.py

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from loguru import logger

from ..models import (
    HealthMetrics,
    HealthAnalysis,
    Anomaly,
    AnomalySeverity,
    Trend,
    TrendDirection
)


class HealthDataAnalyzer:
    """Analyzes health data for anomalies, trends, and correlations."""
    
    # Reference ranges for metrics (based on general health guidelines)
    REFERENCE_RANGES = {
        'sleep_hours': (7.0, 9.0),
        'sleep_quality': (7, 10),
        'hrv': (50, 100),
        'resting_heart_rate': (50, 70),
        'steps': (8000, 12000),
        'active_minutes': (30, 60),
        'stress_score': (1, 4),
    }
    
    # Weights for overall health score
    METRIC_WEIGHTS = {
        'sleep_hours': 0.20,
        'sleep_quality': 0.15,
        'hrv': 0.25,
        'resting_heart_rate': 0.15,
        'steps': 0.10,
        'active_minutes': 0.10,
        'stress_score': 0.05,
    }
    
    def __init__(self):
        """Initialize the analyzer."""
        logger.info("Health data analyzer initialized")
    
    def analyze(
        self,
        current_metrics: HealthMetrics,
        historical_metrics: Optional[List[HealthMetrics]] = None
    ) -> HealthAnalysis:
        """Perform complete health analysis.
        
        Args:
            current_metrics: Current day's health metrics
            historical_metrics: Historical metrics for comparison
            
        Returns:
            Complete health analysis
        """
        logger.info(f"Analyzing health data for date: {current_metrics.date}")
        
        # Calculate overall health score
        overall_score = self._calculate_health_score(current_metrics)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(current_metrics, historical_metrics)
        
        # Detect trends (if historical data available)
        trends = []
        if historical_metrics and len(historical_metrics) >= 3:
            trends = self._detect_trends(current_metrics, historical_metrics)
        
        # Find correlations
        correlations = []
        if historical_metrics and len(historical_metrics) >= 7:
            correlations = self._find_correlations(historical_metrics)
        
        # Generate summary
        summary = self._generate_summary(overall_score, anomalies, trends)
        
        analysis = HealthAnalysis(
            overall_score=overall_score,
            anomalies=anomalies,
            trends=trends,
            correlations=correlations,
            summary=summary
        )
        
        logger.info(f"Analysis complete: score={overall_score}, anomalies={len(anomalies)}, trends={len(trends)}")
        return analysis
    
    def _calculate_health_score(self, metrics: HealthMetrics) -> int:
        """Calculate overall health score (0-100).
        
        Args:
            metrics: Health metrics
            
        Returns:
            Health score from 0-100
        """
        score = 0.0
        
        for metric_name, weight in self.METRIC_WEIGHTS.items():
            if not hasattr(metrics, metric_name):
                continue
            
            value = getattr(metrics, metric_name)
            metric_score = self._score_metric(metric_name, value)
            score += metric_score * weight
        
        # Convert to 0-100 scale
        final_score = int(score * 100)
        return max(0, min(100, final_score))
    
    def _score_metric(self, metric_name: str, value: float) -> float:
        """Score an individual metric (0.0-1.0).
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            Score from 0.0 to 1.0
        """
        if metric_name not in self.REFERENCE_RANGES:
            return 0.5  # Neutral score if no reference
        
        min_val, max_val = self.REFERENCE_RANGES[metric_name]
        
        # Special handling for stress (lower is better)
        if metric_name == 'stress_score':
            if value <= min_val:
                return 1.0
            elif value >= max_val * 2:
                return 0.0
            else:
                return 1.0 - (value - min_val) / (max_val * 2 - min_val)
        
        # For other metrics, within range is best
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            # Below range
            return max(0.0, value / min_val)
        else:
            # Above range (not always bad, but cap the benefit)
            excess = (value - max_val) / max_val
            return max(0.0, 1.0 - excess)
    
    def _detect_anomalies(
        self,
        current_metrics: HealthMetrics,
        historical_metrics: Optional[List[HealthMetrics]]
    ) -> List[Anomaly]:
        """Detect anomalies in current metrics.
        
        Args:
            current_metrics: Current day's metrics
            historical_metrics: Historical metrics for context
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Check against reference ranges
        for metric_name, (min_val, max_val) in self.REFERENCE_RANGES.items():
            if not hasattr(current_metrics, metric_name):
                continue
            
            value = getattr(current_metrics, metric_name)
            
            # Determine severity and create anomaly if out of range
            if value < min_val:
                severity = self._calculate_severity(metric_name, value, min_val, max_val, below=True)
                anomalies.append(Anomaly(
                    metric=metric_name,
                    current_value=value,
                    expected_range=f"{min_val}-{max_val}",
                    severity=severity,
                    description=f"{metric_name.replace('_', ' ').title()} is below optimal range ({value} < {min_val})"
                ))
            elif value > max_val:
                # Some metrics going above range isn't necessarily bad
                if metric_name not in ['steps', 'active_minutes', 'sleep_quality']:
                    severity = self._calculate_severity(metric_name, value, min_val, max_val, below=False)
                    anomalies.append(Anomaly(
                        metric=metric_name,
                        current_value=value,
                        expected_range=f"{min_val}-{max_val}",
                        severity=severity,
                        description=f"{metric_name.replace('_', ' ').title()} is above optimal range ({value} > {max_val})"
                    ))
        
        # Statistical anomaly detection if historical data available
        if historical_metrics and len(historical_metrics) >= 7:
            statistical_anomalies = self._detect_statistical_anomalies(current_metrics, historical_metrics)
            anomalies.extend(statistical_anomalies)
        
        return anomalies
    
    def _calculate_severity(
        self,
        metric_name: str,
        value: float,
        min_val: float,
        max_val: float,
        below: bool
    ) -> AnomalySeverity:
        """Calculate anomaly severity.
        
        Args:
            metric_name: Metric name
            value: Current value
            min_val: Minimum reference value
            max_val: Maximum reference value
            below: True if value is below range
            
        Returns:
            Anomaly severity
        """
        if below:
            deviation_pct = abs((min_val - value) / min_val)
        else:
            deviation_pct = abs((value - max_val) / max_val)
        
        # Critical metrics (HRV, sleep) get higher severity
        critical_metrics = ['hrv', 'sleep_hours', 'resting_heart_rate']
        
        if deviation_pct > 0.3 or (metric_name in critical_metrics and deviation_pct > 0.2):
            return AnomalySeverity.HIGH
        elif deviation_pct > 0.15:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _detect_statistical_anomalies(
        self,
        current_metrics: HealthMetrics,
        historical_metrics: List[HealthMetrics]
    ) -> List[Anomaly]:
        """Detect anomalies using statistical methods (z-score).
        
        Args:
            current_metrics: Current metrics
            historical_metrics: Historical metrics
            
        Returns:
            List of statistical anomalies
        """
        anomalies = []
        
        metrics_to_check = ['hrv', 'sleep_hours', 'resting_heart_rate', 'stress_score']
        
        for metric_name in metrics_to_check:
            historical_values = [getattr(m, metric_name) for m in historical_metrics if hasattr(m, metric_name)]
            
            if len(historical_values) < 3:
                continue
            
            mean = np.mean(historical_values)
            std = np.std(historical_values)
            
            if std == 0:
                continue
            
            current_value = getattr(current_metrics, metric_name)
            z_score = abs((current_value - mean) / std)
            
            # Z-score > 2 is statistically significant
            if z_score > 2:
                severity = AnomalySeverity.HIGH if z_score > 3 else AnomalySeverity.MEDIUM
                
                anomalies.append(Anomaly(
                    metric=metric_name,
                    current_value=current_value,
                    expected_range=f"{mean-std:.1f}-{mean+std:.1f}",
                    severity=severity,
                    description=f"{metric_name.replace('_', ' ').title()} deviates significantly from your personal average (z-score: {z_score:.1f})"
                ))
        
        return anomalies
    
    def _detect_trends(
        self,
        current_metrics: HealthMetrics,
        historical_metrics: List[HealthMetrics]
    ) -> List[Trend]:
        """Detect trends in metrics over time.
        
        Args:
            current_metrics: Current metrics
            historical_metrics: Historical metrics (should be sorted by date)
            
        Returns:
            List of detected trends
        """
        trends = []
        
        # Metrics to analyze for trends
        metrics_to_check = ['hrv', 'sleep_hours', 'resting_heart_rate', 'stress_score', 'steps']
        
        for metric_name in metrics_to_check:
            historical_values = [getattr(m, metric_name) for m in historical_metrics if hasattr(m, metric_name)]
            
            if len(historical_values) < 3:
                continue
            
            # Calculate trend using linear regression
            x = np.arange(len(historical_values))
            y = np.array(historical_values)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Only report significant trends (p < 0.1)
            if p_value < 0.1:
                # Calculate percentage change
                mean_value = np.mean(historical_values)
                change_pct = (slope * len(historical_values) / mean_value) * 100
                
                # Determine direction
                if abs(change_pct) < 5:
                    direction = TrendDirection.STABLE
                else:
                    # For stress and resting_heart_rate, lower is better
                    if metric_name in ['stress_score', 'resting_heart_rate']:
                        direction = TrendDirection.IMPROVING if slope < 0 else TrendDirection.DECLINING
                    else:
                        direction = TrendDirection.IMPROVING if slope > 0 else TrendDirection.DECLINING
                
                trends.append(Trend(
                    metric=metric_name,
                    direction=direction,
                    change_percentage=abs(change_pct),
                    description=f"{metric_name.replace('_', ' ').title()} is {direction.value} ({abs(change_pct):.1f}% change over {len(historical_values)} days)"
                ))
        
        return trends
    
    def _find_correlations(self, historical_metrics: List[HealthMetrics]) -> List[str]:
        """Find correlations between metrics.
        
        Args:
            historical_metrics: Historical metrics
            
        Returns:
            List of correlation insights
        """
        correlations = []
        
        if len(historical_metrics) < 7:
            return correlations
        
        # Extract metric arrays
        sleep_hours = np.array([m.sleep_hours for m in historical_metrics])
        hrv = np.array([m.hrv for m in historical_metrics])
        stress = np.array([m.stress_score for m in historical_metrics])
        rhr = np.array([m.resting_heart_rate for m in historical_metrics])
        
        # Check sleep-HRV correlation
        corr_sleep_hrv = np.corrcoef(sleep_hours, hrv)[0, 1]
        if abs(corr_sleep_hrv) > 0.5:
            direction = "positively" if corr_sleep_hrv > 0 else "negatively"
            correlations.append(f"Sleep duration is {direction} correlated with HRV (r={corr_sleep_hrv:.2f})")
        
        # Check stress-HRV correlation
        corr_stress_hrv = np.corrcoef(stress, hrv)[0, 1]
        if abs(corr_stress_hrv) > 0.5:
            direction = "positively" if corr_stress_hrv > 0 else "negatively"
            correlations.append(f"Stress is {direction} correlated with HRV (r={corr_stress_hrv:.2f})")
        
        # Check sleep-RHR correlation
        corr_sleep_rhr = np.corrcoef(sleep_hours, rhr)[0, 1]
        if abs(corr_sleep_rhr) > 0.5:
            direction = "positively" if corr_sleep_rhr > 0 else "negatively"
            correlations.append(f"Sleep is {direction} correlated with resting heart rate (r={corr_sleep_rhr:.2f})")
        
        return correlations
    
    def _generate_summary(
        self,
        overall_score: int,
        anomalies: List[Anomaly],
        trends: List[Trend]
    ) -> str:
        """Generate a text summary of the analysis.
        
        Args:
            overall_score: Overall health score
            anomalies: Detected anomalies
            trends: Detected trends
            
        Returns:
            Summary text
        """
        summary_parts = []
        
        # Score interpretation
        if overall_score >= 80:
            summary_parts.append(f"Your overall health score is excellent at {overall_score}/100.")
        elif overall_score >= 60:
            summary_parts.append(f"Your overall health score is good at {overall_score}/100, with room for improvement.")
        else:
            summary_parts.append(f"Your overall health score is {overall_score}/100, indicating areas that need attention.")
        
        # Anomaly summary
        high_severity_anomalies = [a for a in anomalies if a.severity == AnomalySeverity.HIGH]
        if high_severity_anomalies:
            metrics = ", ".join([a.metric.replace('_', ' ') for a in high_severity_anomalies])
            summary_parts.append(f"Critical attention needed for: {metrics}.")
        elif anomalies:
            summary_parts.append(f"{len(anomalies)} metrics are outside optimal ranges.")
        else:
            summary_parts.append("All metrics are within healthy ranges.")
        
        # Trend summary
        improving_trends = [t for t in trends if t.direction == TrendDirection.IMPROVING]
        declining_trends = [t for t in trends if t.direction == TrendDirection.DECLINING]
        
        if improving_trends:
            summary_parts.append(f"{len(improving_trends)} metrics showing positive trends.")
        if declining_trends:
            summary_parts.append(f"{len(declining_trends)} metrics showing declining trends that need attention.")
        
        return " ".join(summary_parts)