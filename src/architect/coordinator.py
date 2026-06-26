"""
Главный Архитектор системы — оркестрирует всё:
кластеризацию, фильтрацию, выбор стратегий.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

from ..core.agent import Agent
from ..core.group import AgentGroup
from ..core.metrics import GroupMetrics, FilterThresholds, FilterDecision
from ..coordinator.hclc import HCLCCoordinator
from ..coordinator.cartesian_filter import CartesianFilter
from .threshold_optimizer import ThresholdOptimizer


class Coordinator:
    """
    Главный Архитектор MAS-системы.
    
    Задачи:
    1. Координирует агентов через HCLC (избегаем комбинаторного взрыва)
    2. Оценивает группы в квадрате Декарта
    3. Принимает бинарные решения: ординар (оставить) / режект (отсеять)
    4. Адаптивно настраивает пороги фильтрации
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Инициализируем компоненты
        hclc_config = self.config.get("hclc", {})
        self.coordinator = HCLCCoordinator(
            max_cluster_size=hclc_config.get("max_cluster_size", 5),
            influence_overlap_threshold=hclc_config.get("influence_overlap_threshold", 0.3)
        )
        
        filter_config = self.config.get("cartesian_filter", {})
        thresholds_config = filter_config.get("thresholds", {})
        self.filter = CartesianFilter(
            x_axis=filter_config.get("x_axis", "solution_quality"),
            y_axis=filter_config.get("y_axis", "task_completion_rate"),
            x_threshold=filter_config.get("x_threshold", 0.0),
            y_threshold=filter_config.get("y_threshold", 0.0),
            thresholds=FilterThresholds(**thresholds_config) if thresholds_config else None
        )
        
        architect_config = self.config.get("architect", {})
        self.adaptive_filter = ThresholdOptimizer(
            target_survival_rate=architect_config.get("target_survival_rate", 0.3)
        )
        
        # Реестр групп
        self.groups: Dict[str, AgentGroup] = {}
        self.agents: List[Agent] = []
        
        # История решений
        self.decisions_history: List[Dict[str, Any]] = []
        
        # Статистика раундов
        self.round_stats: List[Dict[str, Any]] = []
    
    def register_agents(self, agents: List[Agent]) -> None:
        """Регистрирует агентов в системе."""
        self.agents = agents
        self.coordinator.build_local_contexts(agents)
        self.coordinator.cluster_agents(agents)
        self.coordinator.compute_cluster_interactions(agents)
    
    def register_group(self, group: AgentGroup) -> None:
        """Регистрирует группу агентов."""
        self.groups[group.id] = group
    
    def create_groups_from_strategies(self, 
                                      strategy_names: List[str],
                                      agents_per_group: int = 5) -> List[AgentGroup]:
        """
        Создаёт группы агентов на основе списка стратегий.
        """
        groups = []
        
        for i, strategy_name in enumerate(strategy_names):
            group = AgentGroup(
                id=f"group_{i}_{strategy_name}",
                strategy_name=strategy_name
            )
            
            # Добавляем агентов в группу
            start_idx = i * agents_per_group
            end_idx = min(start_idx + agents_per_group, len(self.agents))
            
            for j in range(start_idx, end_idx):
                group.add_agent(self.agents[j])
            
            groups.append(group)
            self.register_group(group)
        
        return groups
    
    def evaluate_all_groups(self) -> Dict[str, GroupMetrics]:
        """Оценивает все группы и возвращает их метрики."""
        metrics = {}
        
        for group_id, group in self.groups.items():
            group_metrics = group.evaluate_performance()
            group.update_metrics(group_metrics)
            metrics[group_id] = group_metrics
        
        return metrics
    
    def run_round(self) -> Dict[str, Any]:
        """
        Запускает один раунд координации и оценки.
        
        1. Координирует агентов через HCLC
        2. Оценивает все группы
        3. Фильтрует через декартов фильтр
        4. Принимает бинарные решения
        """
        # Шаг 1: Координация (обновляем контексты)
        self.coordinator.build_local_contexts(self.agents)
        complexity_stats = self.coordinator.get_complexity_reduction(len(self.agents))
        
        # Шаг 2: Оценка групп
        groups_metrics = self.evaluate_all_groups()
        
        # Шаг 3: Фильтрация
        filter_results = self.filter.filter_groups(groups_metrics)
        
        # Шаг 4: Калибровка адаптивного фильтра
        all_metrics = list(groups_metrics.values())
        self.adaptive_filter.calibrate(all_metrics)
        
        # Собираем результаты раунда
        round_result = {
            "complexity_stats": complexity_stats,
            "filter_results": filter_results,
            "survivors": [s["id"] for s in filter_results["survivors"]],
            "n_survivors": len(filter_results["survivors"]),
            "n_rejected": len(filter_results["rejected"]),
            "survival_rate": filter_results["survival_rate"],
            "quadrant_distribution": filter_results["quadrant_distribution"]
        }
        
        self.round_stats.append(round_result)
        
        return round_result
    
    def get_best_groups(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Возвращает N лучших групп на основе последнего раунда.
        """
        if not self.round_stats:
            return []
        
        last_round = self.round_stats[-1]
        filter_results = last_round["filter_results"]
        
        return filter_results["survivors"][:n]
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по системе."""
        return {
            "total_agents": len(self.agents),
            "total_groups": len(self.groups),
            "strategies": list(set(g.strategy_name for g in self.groups.values())),
            "rounds_completed": len(self.round_stats),
            "complexity_reduction": self.coordinator.get_complexity_reduction(len(self.agents)),
            "filter_stats": self.filter.get_statistics(),
            "current_thresholds": self.adaptive_filter.get_current_thresholds(),
            "last_round": self.round_stats[-1] if self.round_stats else None
        }
    
    def get_quadrant_map(self) -> Dict[str, List[str]]:
        """
        Возвращает карту квадрантов: какие группы в каком квадранте.
        """
        quadrant_map = {"++": [], "+-": [], "-+": [], "--": []}
        
        for group_id, group in self.groups.items():
            if group.metrics:
                x, y = self.filter.project(group.metrics)
                quadrant = self.filter.get_quadrant(x, y)
                quadrant_map[quadrant].append(group_id)
        
        return quadrant_map