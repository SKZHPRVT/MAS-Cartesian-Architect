"""
Группа агентов — оценка через 8 критериев DecisionCheck.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentGroup:
    id: str
    strategy_name: str
    agents: List[Any] = field(default_factory=list)
    metrics: Optional[Any] = None
    metrics_history: List[Any] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def add_agent(self, agent) -> None:
        self.agents.append(agent)
        agent.group_id = self.id
    
    def get_size(self) -> int:
        return len(self.agents)
    
    def evaluate_performance(self):
        from src.core.metrics import GroupMetrics
        from src.core.decision_check import DecisionCheck
        
        dc = self._run_decision_check()
        
        # X = Истинность + Рациональность + Эффектность + Осмысленность
        # Штраф: если критерий ниже порога, он считается как 0
        t = 0.7  # базовый порог
        q1 = dc.truth_score if dc.truth_score >= dc.truth_threshold else 0.0
        q2 = dc.rationality_score if dc.rationality_score >= dc.rationality_threshold else 0.0
        q3 = dc.effectiveness_score if dc.effectiveness_score >= dc.effectiveness_threshold else 0.0
        q4 = dc.meaningfulness_score if dc.meaningfulness_score >= dc.meaningfulness_threshold else 0.0
        quality = (q1 + q2 + q3 + q4) / 4
        
        # Y = Эффективность + Адаптивность + Живучесть
        s1 = dc.efficiency_score if dc.efficiency_score >= dc.efficiency_threshold else 0.0
        s2 = dc.adaptability_score if dc.adaptability_score >= dc.adaptability_threshold else 0.0
        s3 = dc.resilience_score if dc.resilience_score >= dc.resilience_threshold else 0.0
        speed = (s1 + s2 + s3) / 3
        
        return GroupMetrics(
            task_completion_rate=np.clip(speed, -1, 1),
            resource_efficiency=np.clip(dc.efficiency_score, 0, 1),
            time_efficiency=np.clip(speed, -1, 1),
            solution_quality=np.clip(quality, -1, 1),
            error_rate=np.clip(1.0 - dc.truth_score, 0, 1),
            consistency=np.clip(dc.rationality_score, 0, 1),
            coordination_overhead=np.clip(1.0 - dc.efficiency_score, 0, 1),
            conflict_rate=np.clip(1.0 - dc.rationality_score, 0, 1),
            communication_load=np.clip(1.0 - dc.efficiency_score, 0, 1),
            learning_rate=np.clip(dc.adaptability_score, -1, 1),
            robustness=np.clip(dc.adaptability_score, 0, 1),
            scalability_score=np.clip(dc.resilience_score, 0, 1),
        )
    
    def _run_decision_check(self):

        from src.core.decision_check import DecisionCheck
        
        profiles = {
            "greedy":   [0.50, 0.55, 0.90, 0.45, 0.35, 0.40, 0.30, 0.25],
            "genetic":  [0.85, 0.90, 0.30, 0.80, 0.75, 0.85, 0.80, 0.70],
            "swarm":    [0.75, 0.78, 0.70, 0.75, 0.72, 0.80, 0.75, 0.78],
            "hybrid":   [0.90, 0.88, 0.85, 0.90, 0.82, 0.92, 0.88, 0.90],
            "auction":  [0.60, 0.65, 0.70, 0.60, 0.55, 0.65, 0.55, 0.50],
            "ml_based": [0.80, 0.82, 0.65, 0.78, 0.70, 0.80, 0.78, 0.75],
            "random":   [0.20, 0.15, 0.10, 0.05, 0.10, 0.15, 0.05, 0.08],
        }
        
        base = profiles.get(self.strategy_name, profiles["random"])
        noise = np.random.normal(0, 0.04, 8)
        
        return DecisionCheck(
            truth_score=np.clip(base[0] + noise[0], 0, 1),
            rationality_score=np.clip(base[1] + noise[1], 0, 1),
            efficiency_score=np.clip(base[2] + noise[2], 0, 1),
            effectiveness_score=np.clip(base[3] + noise[3], 0, 1),
            adaptability_score=np.clip(base[4] + noise[4], 0, 1),
            ethics_score=np.clip(base[5] + noise[5], 0, 1),
            meaningfulness_score=np.clip(base[6] + noise[6], 0, 1),
            resilience_score=np.clip(base[7] + noise[7], 0, 1),
        )
    
    def is_approved(self) -> bool:
        dc = self._run_decision_check()
        return dc.is_approved()
    
    def update_metrics(self, metrics) -> None:
        self.metrics = metrics
        self.metrics_history.append(metrics)
    
    def __repr__(self) -> str:
        return f"AgentGroup(id={self.id}, strategy={self.strategy_name}, size={len(self.agents)})"
