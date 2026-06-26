"""
Адаптивный фильтр — автоматически настраивает пороги 
на основе распределения групп и обратной связи.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from ..core.metrics import GroupMetrics, FilterThresholds


class ThresholdOptimizer:
    """
    Фильтр, который сам калибрует пороги для бинарной фильтрации.
    """
    
    def __init__(self, target_survival_rate: float = 0.3):
        self.target_survival_rate = target_survival_rate
        self.thresholds = FilterThresholds()
        self.history: List[Dict[str, Any]] = []
        self.survivors: List[str] = []
    
    def calibrate(self, all_metrics: List[GroupMetrics]) -> FilterThresholds:
        """
        Калибрует пороги так, чтобы выжило примерно target_survival_rate групп.
        """
        n = len(all_metrics)
        if n < 2:
            return self.thresholds
        
        percentile = 100 * (1 - self.target_survival_rate)
        
        # Позитивные параметры (больше = лучше)
        positive_params = {
            "min_task_completion": [m.task_completion_rate for m in all_metrics],
            "min_resource_efficiency": [m.resource_efficiency for m in all_metrics],
            "min_time_efficiency": [m.time_efficiency for m in all_metrics],
            "min_solution_quality": [m.solution_quality for m in all_metrics],
            "min_consistency": [m.consistency for m in all_metrics],
            "min_robustness": [m.robustness for m in all_metrics],
            "min_scalability": [m.scalability_score for m in all_metrics],
            "min_learning_rate": [m.learning_rate for m in all_metrics],
        }
        
        # Негативные параметры (меньше = лучше)
        negative_params = {
            "max_error_rate": [m.error_rate for m in all_metrics],
            "max_coordination_overhead": [m.coordination_overhead for m in all_metrics],
            "max_conflict_rate": [m.conflict_rate for m in all_metrics],
            "max_communication_load": [m.communication_load for m in all_metrics],
        }
        
        for param_name, values in positive_params.items():
            threshold = np.percentile(values, percentile)
            setattr(self.thresholds, param_name, float(threshold))
        
        for param_name, values in negative_params.items():
            threshold = np.percentile(values, 100 - percentile)
            setattr(self.thresholds, param_name, float(threshold))
        
        return self.thresholds
    
    def update_from_feedback(self, group_id: str, was_successful: bool, was_kept: bool):
        """
        Обучается на основе обратной связи.
        """
        self.history.append({
            "group_id": group_id,
            "successful": was_successful,
            "kept": was_kept
        })
        
        # Анализируем последние 20 решений
        recent = self.history[-20:]
        
        # Ложные отсеивания: группа была успешной, но её отсеяли
        false_rejections = sum(
            1 for h in recent 
            if h["successful"] and not h["kept"]
        )
        
        # Ложные принятия: группа была неуспешной, но её пропустили
        false_acceptances = sum(
            1 for h in recent 
            if not h["successful"] and h["kept"]
        )
        
        adjustment = 0.02
        
        if false_rejections > 2:
            # Слишком строгий — ослабляем
            self.thresholds.adjust(-adjustment)
        elif false_acceptances > 2:
            # Слишком мягкий — ужесточаем
            self.thresholds.adjust(adjustment)
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Возвращает текущие пороги."""
        return self.thresholds.to_dict()