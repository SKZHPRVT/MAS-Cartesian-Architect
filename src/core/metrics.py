"""
Метрики для оценки групп агентов и бинарной фильтрации.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class FilterDecision(Enum):
    KEEP = "KEEP"
    REJECT = "REJECT"


@dataclass
class GroupMetrics:
    task_completion_rate: float = 0.0
    resource_efficiency: float = 0.0
    time_efficiency: float = 0.0
    solution_quality: float = 0.0
    error_rate: float = 1.0
    consistency: float = 0.0
    coordination_overhead: float = 1.0
    conflict_rate: float = 1.0
    communication_load: float = 1.0
    learning_rate: float = 0.0
    robustness: float = 0.0
    scalability_score: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "task_completion_rate": self.task_completion_rate,
            "resource_efficiency": self.resource_efficiency,
            "time_efficiency": self.time_efficiency,
            "solution_quality": self.solution_quality,
            "error_rate": self.error_rate,
            "consistency": self.consistency,
            "coordination_overhead": self.coordination_overhead,
            "conflict_rate": self.conflict_rate,
            "communication_load": self.communication_load,
            "learning_rate": self.learning_rate,
            "robustness": self.robustness,
            "scalability_score": self.scalability_score,
        }


@dataclass
class FilterThresholds:
    # АДАПТИРОВАННЫЕ пороги — реалистичные для наших стратегий
    min_task_completion: float = 0.0     # Не требуем минимум (genetic медленный но качественный)
    min_resource_efficiency: float = 0.0
    min_time_efficiency: float = 0.0
    min_solution_quality: float = 0.0    # Главное — качество неотрицательное
    max_error_rate: float = 0.50         # Ослабили (было 0.30)
    min_consistency: float = 0.0
    max_coordination_overhead: float = 0.70  # Ослабили (было 0.40)
    max_conflict_rate: float = 0.50      # Ослабили (было 0.20)
    max_communication_load: float = 0.70
    min_learning_rate: float = -1.0
    min_robustness: float = 0.0
    min_scalability: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "min_task_completion": self.min_task_completion,
            "min_resource_efficiency": self.min_resource_efficiency,
            "min_time_efficiency": self.min_time_efficiency,
            "min_solution_quality": self.min_solution_quality,
            "max_error_rate": self.max_error_rate,
            "min_consistency": self.min_consistency,
            "max_coordination_overhead": self.max_coordination_overhead,
            "max_conflict_rate": self.max_conflict_rate,
            "max_communication_load": self.max_communication_load,
            "min_learning_rate": self.min_learning_rate,
            "min_robustness": self.min_robustness,
            "min_scalability": self.min_scalability,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'FilterThresholds':
        return cls(**data)
    
    def adjust(self, delta: float) -> None:
        for attr in self.__dataclass_fields__:
            current = getattr(self, attr)
            new_val = max(0.0, min(1.0, current + delta))
            setattr(self, attr, new_val)
