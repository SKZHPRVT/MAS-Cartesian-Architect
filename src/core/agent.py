"""
Базовый агент MAS системы.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class AgentState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    COORDINATING = "coordinating"
    ERROR = "error"


@dataclass
class Agent:
    """
    Базовый агент мультиагентной системы.
    """
    id: int
    position: np.ndarray
    radius_of_influence: float = 1.0
    
    # Состояние агента
    state: AgentState = AgentState.IDLE
    
    # Локальный контекст — агенты, которые реально влияют на этого
    local_context: Set[int] = field(default_factory=set)
    
    # Принадлежность к группе
    group_id: Optional[str] = None
    
    # Метрики производительности
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    
    # Параметры стратегии
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    def update_position(self, new_position: np.ndarray) -> None:
        """Обновляет позицию агента."""
        self.position = np.array(new_position)
    
    def add_to_context(self, agent_id: int) -> None:
        """Добавляет агента в локальный контекст."""
        self.local_context.add(agent_id)
    
    def remove_from_context(self, agent_id: int) -> None:
        """Удаляет агента из локального контекста."""
        self.local_context.discard(agent_id)
    
    def is_influenced_by(self, other: 'Agent') -> bool:
        """Проверяет, влияет ли другой агент на этого."""
        distance = np.linalg.norm(self.position - other.position)
        return distance < (self.radius_of_influence + other.radius_of_influence)
    
    def record_performance(self, metrics: Dict[str, float]) -> None:
        """Записывает метрики производительности."""
        self.performance_history.append(metrics)
    
    def get_average_performance(self, window: int = 10) -> Dict[str, float]:
        """Возвращает среднюю производительность за последние window шагов."""
        if not self.performance_history:
            return {}
        
        recent = self.performance_history[-window:]
        avg = {}
        for key in recent[0].keys():
            avg[key] = np.mean([m[key] for m in recent])
        return avg
    
    def __repr__(self) -> str:
        return f"Agent(id={self.id}, state={self.state.value}, group={self.group_id})"