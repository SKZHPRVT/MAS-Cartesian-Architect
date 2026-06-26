"""
Декартов фильтр — проецирует группы на квадрат Декарта 
и принимает бинарное решение: ординар или режект.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from ..core.metrics import GroupMetrics, FilterThresholds, FilterDecision


class CartesianFilter:
    """
    Фильтр, работающий в логике квадрата Декарта.
    Проецирует группу на 2 оси и принимает бинарное решение.
    """
    
    def __init__(self,
                 x_axis: str = "solution_quality",
                 y_axis: str = "task_completion_rate",
                 x_threshold: float = 0.0,
                 y_threshold: float = 0.0,
                 thresholds: Optional[FilterThresholds] = None):
        
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.thresholds = thresholds or FilterThresholds()
        
        # Статистика
        self.total_evaluated = 0
        self.total_kept = 0
        self.total_rejected = 0
    
    def project(self, metrics: GroupMetrics) -> Tuple[float, float]:
        """Проецирует группу на квадрат Декарта."""
        x_val = getattr(metrics, self.x_axis)
        y_val = getattr(metrics, self.y_axis)
        return x_val, y_val
    
    def get_quadrant(self, x: float, y: float) -> str:
        """Определяет квадрант Декарта."""
        if x >= self.x_threshold and y >= self.y_threshold:
            return "++"
        elif x >= self.x_threshold and y < self.y_threshold:
            return "+-"
        elif x < self.x_threshold and y >= self.y_threshold:
            return "-+"
        else:
            return "--"
    
    def evaluate(self, metrics: GroupMetrics) -> FilterDecision:
        """
        Бинарная оценка группы.
        Группа проходит, если удовлетворяет ВСЕМ порогам.
        """
        self.total_evaluated += 1
        
        checks = {
            "task_completion": metrics.task_completion_rate >= self.thresholds.min_task_completion,
            "resource_efficiency": metrics.resource_efficiency >= self.thresholds.min_resource_efficiency,
            "time_efficiency": metrics.time_efficiency >= self.thresholds.min_time_efficiency,
            "solution_quality": metrics.solution_quality >= self.thresholds.min_solution_quality,
            "error_rate": metrics.error_rate <= self.thresholds.max_error_rate,
            "consistency": metrics.consistency >= self.thresholds.min_consistency,
            "coordination_overhead": metrics.coordination_overhead <= self.thresholds.max_coordination_overhead,
            "conflict_rate": metrics.conflict_rate <= self.thresholds.max_conflict_rate,
            "communication_load": metrics.communication_load <= self.thresholds.max_communication_load,
            "learning_rate": metrics.learning_rate >= self.thresholds.min_learning_rate,
            "robustness": metrics.robustness >= self.thresholds.min_robustness,
            "scalability": metrics.scalability >= self.thresholds.min_scalability,
        }
        
        all_passed = all(checks.values())
        
        if all_passed:
            self.total_kept += 1
            return FilterDecision.KEEP
        else:
            self.total_rejected += 1
            return FilterDecision.REJECT
    
    def evaluate_with_details(self, metrics: GroupMetrics) -> Dict[str, Any]:
        """Оценивает группу и возвращает детальную информацию."""
        x, y = self.project(metrics)
        quadrant = self.get_quadrant(x, y)
        decision = self.evaluate(metrics)
        
        # Собираем причины отказа
        failed_checks = []
        
        if metrics.task_completion_rate < self.thresholds.min_task_completion:
            failed_checks.append(f"task_completion: {metrics.task_completion_rate:.2f} < {self.thresholds.min_task_completion}")
        if metrics.solution_quality < self.thresholds.min_solution_quality:
            failed_checks.append(f"solution_quality: {metrics.solution_quality:.2f} < {self.thresholds.min_solution_quality}")
        if metrics.error_rate > self.thresholds.max_error_rate:
            failed_checks.append(f"error_rate: {metrics.error_rate:.2f} > {self.thresholds.max_error_rate}")
        if metrics.conflict_rate > self.thresholds.max_conflict_rate:
            failed_checks.append(f"conflict_rate: {metrics.conflict_rate:.2f} > {self.thresholds.max_conflict_rate}")
        if metrics.coordination_overhead > self.thresholds.max_coordination_overhead:
            failed_checks.append(f"coordination_overhead: {metrics.coordination_overhead:.2f} > {self.thresholds.max_coordination_overhead}")
        
        # Расстояние до идеала (1, 1)
        distance_to_ideal = np.sqrt((1 - x)**2 + (1 - y)**2)
        
        return {
            "decision": decision,
            "quadrant": quadrant,
            "x_value": x,
            "y_value": y,
            "distance_to_ideal": distance_to_ideal,
            "failed_checks": failed_checks,
            "all_metrics": metrics.to_dict()
        }
    
    def filter_groups(self, 
                      groups_metrics: Dict[str, GroupMetrics]
                      ) -> Dict[str, Any]:
        """
        Фильтрует все группы, возвращает выживших и отсеянных.
        """
        survivors = []
        rejected = []
        quadrant_dist = {"++": [], "+-": [], "-+": [], "--": []}
        
        for group_id, metrics in groups_metrics.items():
            details = self.evaluate_with_details(metrics)
            
            quadrant_dist[details["quadrant"]].append(group_id)
            
            if details["decision"] == FilterDecision.KEEP:
                survivors.append({
                    "id": group_id,
                    "quadrant": details["quadrant"],
                    "x": details["x_value"],
                    "y": details["y_value"],
                    "distance": details["distance_to_ideal"]
                })
            else:
                rejected.append({
                    "id": group_id,
                    "quadrant": details["quadrant"],
                    "reasons": details["failed_checks"]
                })
        
        # Сортируем выживших по близости к идеалу
        survivors.sort(key=lambda s: s["distance"])
        
        return {
            "total": len(groups_metrics),
            "survivors": survivors,
            "rejected": rejected,
            "quadrant_distribution": {
                q: len(groups) for q, groups in quadrant_dist.items()
            },
            "survival_rate": len(survivors) / max(1, len(groups_metrics))
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику фильтра."""
        return {
            "total_evaluated": self.total_evaluated,
            "total_kept": self.total_kept,
            "total_rejected": self.total_rejected,
            "keep_rate": self.total_kept / max(1, self.total_evaluated)
        }