"""
Иерархическая кластеризация с локальными контекстами (HCLC).
Решает проблему комбинаторного взрыва через ограничение области взаимодействия.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
import heapq

from ..core.agent import Agent


class HCLCCoordinator:
    """
    Координатор с иерархической кластеризацией.
    Ограничивает взаимодействия агентов, группируя их в кластеры.
    """
    
    def __init__(self, 
                 max_cluster_size: int = 5,
                 influence_overlap_threshold: float = 0.3):
        self.max_cluster_size = max_cluster_size
        self.overlap_threshold = influence_overlap_threshold
        self.clusters: List[Set[int]] = []
        self.cluster_interactions: Dict[Tuple[int, int], Set[int]] = {}
    
    def build_local_contexts(self, agents: List[Agent]) -> None:
        """
        Строит локальные контексты: для каждого агента определяет,
        какие другие агенты действительно влияют на него.
        """
        for i, agent_i in enumerate(agents):
            agent_i.local_context = set()
            for j, agent_j in enumerate(agents):
                if i != j:
                    if agent_i.is_influenced_by(agent_j):
                        agent_i.add_to_context(j)
    
    def cluster_agents(self, agents: List[Agent]) -> List[Set[int]]:
        """
        Кластеризует агентов на основе пересечения локальных контекстов.
        """
        visited = set()
        clusters = []
        
        def dfs(agent_id: int, current_cluster: Set[int]):
            visited.add(agent_id)
            current_cluster.add(agent_id)
            
            for neighbor_id in agents[agent_id].local_context:
                if neighbor_id not in visited:
                    if agent_id in agents[neighbor_id].local_context:
                        if len(current_cluster) < self.max_cluster_size:
                            dfs(neighbor_id, current_cluster)
        
        for agent_id in range(len(agents)):
            if agent_id not in visited:
                new_cluster = set()
                dfs(agent_id, new_cluster)
                clusters.append(new_cluster)
        
        self.clusters = clusters
        return clusters
    
    def compute_cluster_interactions(self, agents: List[Agent]) -> Dict[Tuple[int, int], Set[int]]:
        """
        Определяет взаимодействия между кластерами.
        Находит только релевантные межкластерные связи.
        """
        interactions = {}
        
        for i, cluster_i in enumerate(self.clusters):
            for j, cluster_j in enumerate(self.clusters):
                if i < j:
                    border_agents = set()
                    for agent_i in cluster_i:
                        for agent_j in cluster_j:
                            if (agent_j in agents[agent_i].local_context and 
                                agent_i in agents[agent_j].local_context):
                                border_agents.add(agent_i)
                                border_agents.add(agent_j)
                    
                    if border_agents:
                        interactions[(i, j)] = border_agents
        
        self.cluster_interactions = interactions
        return interactions
    
    def get_complexity_reduction(self, n_agents: int) -> Dict[str, Any]:
        """
        Вычисляет, насколько снизилась сложность по сравнению с полным перебором.
        """
        full_cartesian = n_agents ** 2
        
        if not self.clusters:
            return {
                "full_cartesian": full_cartesian,
                "hierarchical_complexity": full_cartesian,
                "reduction_factor": 1.0,
                "n_clusters": 0,
                "avg_cluster_size": 0
            }
        
        avg_cluster_size = np.mean([len(c) for c in self.clusters])
        n_clusters = len(self.clusters)
        hierarchical_complexity = (
            avg_cluster_size ** 2 * n_clusters + 
            len(self.cluster_interactions) * avg_cluster_size
        )
        
        return {
            "full_cartesian": full_cartesian,
            "hierarchical_complexity": int(hierarchical_complexity),
            "reduction_factor": full_cartesian / max(1, hierarchical_complexity),
            "n_clusters": n_clusters,
            "avg_cluster_size": avg_cluster_size
        }