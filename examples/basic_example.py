"""
Базовый пример использования MAS Cartesian Architect.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.core.agent import Agent
from src.core.group import AgentGroup
from src.architect.meta_architect import MetaArchitect
from src.visualization.plotter import CartesianVisualizer


def run_basic_example():
    """
    Базовый пример: создаём агентов, группы с разными стратегиями,
    запускаем Архитектора, смотрим результаты.
    """
    print("=" * 60)
    print("MAS Cartesian Architect - Базовый пример")
    print("=" * 60)
    
    # 1. Создаём агентов
    print("\n1. Создание агентов...")
    np.random.seed(42)
    n_agents = 50
    
    agents = []
    for i in range(n_agents):
        pos = np.random.rand(2) * 10
        radius = np.random.uniform(0.5, 2.0)
        agents.append(Agent(id=i, position=pos, radius_of_influence=radius))
    
    print(f"   Создано {len(agents)} агентов")
    
    # 2. Конфигурация Архитектора
    config = {
        "hclc": {
            "max_cluster_size": 5,
            "influence_overlap_threshold": 0.3
        },
        "cartesian_filter": {
            "x_axis": "solution_quality",
            "y_axis": "task_completion_rate",
            "x_threshold": 0.0,
            "y_threshold": 0.0
        },
        "architect": {
            "target_survival_rate": 0.3,
            "exploration_rate": 0.1
        }
    }
    
    # 3. Создаём Архитектора
    print("\n2. Инициализация Архитектора...")
    architect = MetaArchitect(config)
    architect.register_agents(agents)
    
    # 4. Создаём группы с разными стратегиями
    strategies = [
        "greedy", "greedy",
        "genetic", "genetic",
        "swarm", "swarm",
        "hybrid", "hybrid",
        "auction", "auction",
        "ml_based",
        "random"  # Эта должна отсеяться
    ]
    
    print("\n3. Создание групп со стратегиями...")
    groups = architect.create_groups_from_strategies(strategies, agents_per_group=4)
    
    for group in groups:
        print(f"   {group}")
    
    # 5. Запускаем несколько раундов
    print("\n4. Запуск раундов координации...")
    n_rounds = 5
    
    for round_num in range(n_rounds):
        print(f"\n--- Раунд {round_num + 1} ---")
        result = architect.run_round()
        
        print(f"   Выжило групп: {result['n_survivors']}")
        print(f"   Отсеяно: {result['n_rejected']}")
        print(f"   Выживаемость: {result['survival_rate']:.1%}")
        print(f"   Распределение по квадрантам: {result['quadrant_distribution']}")
        print(f"   Снижение сложности: {result['complexity_stats']['reduction_factor']:.1f}x")
    
    # 6. Показываем лучшие группы
    print("\n5. Топ-3 лучших групп:")
    best_groups = architect.get_best_groups(3)
    for i, group_info in enumerate(best_groups):
        print(f"   {i+1}. {group_info['id']} — {group_info['quadrant']} "
              f"(distance to ideal: {group_info['distance']:.3f})")
    
    # 7. Визуализация
    print("\n6. Визуализация результатов...")
    
    # Собираем метрики последнего раунда
    groups_metrics = {
        group_id: group.metrics 
        for group_id, group in architect.groups.items() 
        if group.metrics is not None
    }
    
    survivors = result.get("survivors", [])
    
    visualizer = CartesianVisualizer()
    visualizer.plot_quadrant_map(
        groups_metrics=groups_metrics,
        survivors=survivors,
        title="Результаты фильтрации групп в квадрате Декарта",
        x_axis=config["cartesian_filter"]["x_axis"],
        y_axis=config["cartesian_filter"]["y_axis"]
    )
    
    # График эволюции
    visualizer.plot_evolution(architect.round_stats)
    
    # 8. Сводка
    print("\n7. Сводка по системе:")
    summary = architect.get_system_summary()
    print(f"   Всего агентов: {summary['total_agents']}")
    print(f"   Всего групп: {summary['total_groups']}")
    print(f"   Стратегии: {summary['strategies']}")
    print(f"   Раундов: {summary['rounds_completed']}")
    print(f"   Снижение сложности: {summary['complexity_reduction']['reduction_factor']:.1f}x")
    print(f"   Всего оценено: {summary['filter_stats']['total_evaluated']}")
    print(f"   Прошло фильтр: {summary['filter_stats']['total_kept']}")
    
    print("\n" + "=" * 60)
    print("Пример завершён!")
    print("=" * 60)
    
    return architect, visualizer


if __name__ == "__main__":
    architect, visualizer = run_basic_example()