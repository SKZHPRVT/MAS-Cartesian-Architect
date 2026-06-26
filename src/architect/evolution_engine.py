"""
Популяционный Архитектор — поиск ОРДИНАРА (единственной лучшей группы).
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class EvolutionConfig:
    initial_population_size: int = 50
    min_survivors: int = 5
    max_survivors: int = 15
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_enabled: bool = True


class EvolutionEngine:
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.generations_history: List[Dict[str, Any]] = []
    
    def generate_initial_population(self, base_strategies: List[str], agents_pool: List[Any]) -> List[Any]:
        from src.core.group import AgentGroup
        
        population = []
        n_base = len(base_strategies)
        
        for i in range(self.config.initial_population_size):
            base_strategy = base_strategies[i % n_base]
            variation = self._create_strategy_variation(base_strategy, i, self.config.initial_population_size)
            group_agents = self._select_agents_for_group(agents_pool, i, variation)
            
            group = AgentGroup(id=f"pop_{i}_{variation['name']}", strategy_name=variation["base_strategy"])
            group.params = variation
            for agent in group_agents:
                group.add_agent(agent)
            
            population.append(group)
        
        print(f"Создана начальная популяция: {len(population)} групп")
        return population
    
    def _create_strategy_variation(self, base_strategy: str, variation_id: int, total_variations: int) -> Dict[str, Any]:
        strategy_profiles = {
            "greedy": {"exploration_bias": 0.2, "coordination_style": "centralized", "risk_tolerance": 0.3, "typical_quadrant": "+-"},
            "genetic": {"exploration_bias": 0.8, "coordination_style": "decentralized", "risk_tolerance": 0.7, "typical_quadrant": "-+"},
            "swarm": {"exploration_bias": 0.6, "coordination_style": "emergent", "risk_tolerance": 0.5, "typical_quadrant": "++"},
            "hybrid": {"exploration_bias": 0.5, "coordination_style": "adaptive", "risk_tolerance": 0.6, "typical_quadrant": "++"},
            "auction": {"exploration_bias": 0.4, "coordination_style": "market", "risk_tolerance": 0.4, "typical_quadrant": "+-"},
            "ml_based": {"exploration_bias": 0.7, "coordination_style": "learned", "risk_tolerance": 0.5, "typical_quadrant": "++"},
            "random": {"exploration_bias": 1.0, "coordination_style": "random", "risk_tolerance": 1.0, "typical_quadrant": "--"}
        }
        
        base_profile = strategy_profiles.get(base_strategy, strategy_profiles["random"])
        variation_seed = variation_id / total_variations
        
        exploration = np.clip(base_profile["exploration_bias"] + np.random.normal(0, 0.15) + variation_seed * 0.2, 0.05, 0.95)
        risk = np.clip(base_profile["risk_tolerance"] + np.random.normal(0, 0.15) + variation_seed * 0.2, 0.05, 0.95)
        agents_count = int(np.clip(5 + np.random.normal(0, 2) + variation_seed * 10, 3, 15))
        
        return {
            "name": f"{base_strategy}_v{variation_id}",
            "base_strategy": base_strategy,
            "exploration_bias": exploration,
            "risk_tolerance": risk,
            "coordination_style": base_profile["coordination_style"],
            "typical_quadrant": base_profile["typical_quadrant"],
            "agents_count": agents_count,
            "variation_seed": variation_seed,
        }
    
    def _select_agents_for_group(self, agents_pool, group_id, strategy):
        n_needed = strategy["agents_count"]
        n_available = len(agents_pool)
        step = max(1, n_available // n_needed)
        indices = list(range(0, n_available, step))[:n_needed]
        return [agents_pool[i] for i in indices]
    
    def evaluate_population(self, population, architect) -> Dict[str, Any]:
        results = {"++": [], "+-": [], "-+": [], "--": [], "all_metrics": {}}
        
        for group in population:
            metrics = group.evaluate_performance()
            group.update_metrics(metrics)
            
            x = getattr(metrics, architect.filter.x_axis)
            y = getattr(metrics, architect.filter.y_axis)
            quadrant = architect.filter.get_quadrant(x, y)
            
            results[quadrant].append({
                "group_id": group.id, "strategy": group.strategy_name,
                "x": x, "y": y, "metrics": metrics, "params": group.params
            })
            results["all_metrics"][group.id] = metrics
        
        coverage = self._calculate_coverage(results)
        
        # Ранжируем ВСЕ группы — ОРДИНАР = самая близкая к (1,1)
        all_ranked = []
        for quadrant in ["++", "+-", "-+", "--"]:
            for info in results[quadrant]:
                distance = np.sqrt((1 - info["x"])**2 + (1 - info["y"])**2)
                all_ranked.append({**info, "distance_to_ideal": distance})
        all_ranked.sort(key=lambda x: x["distance_to_ideal"])
        
        ordinal = all_ranked[0] if all_ranked else None
        
        return {
            "population_size": len(population),
            "quadrant_distribution": {q: len(groups) for q, groups in results.items() if q != "all_metrics"},
            "coverage_percent": coverage["percent"],
            "survivors": results["++"],
            "best_performers": all_ranked[:5],
            "ordinal": ordinal,
            "detailed_results": results
        }
    
    def _calculate_coverage(self, results: Dict) -> Dict[str, Any]:
        grid_size = 10
        coverage_grid = np.zeros((grid_size, grid_size))
        
        for quadrant_groups in ["++", "+-", "-+", "--"]:
            for group_info in results[quadrant_groups]:
                x = group_info["x"]
                y = group_info["y"]
                cell_x = np.clip(int((x + 1.0) / 2.0 * grid_size), 0, grid_size - 1)
                cell_y = np.clip(int((y + 1.0) / 2.0 * grid_size), 0, grid_size - 1)
                coverage_grid[cell_y, cell_x] = 1
        
        covered_cells = np.sum(coverage_grid)
        total_cells = grid_size * grid_size
        coverage_percent = (covered_cells / total_cells) * 100
        
        return {"percent": coverage_percent, "covered_cells": int(covered_cells), "total_cells": total_cells}
    
    def _rank_by_distance(self, survivors: List[Dict]) -> List[Dict]:
        ranked = []
        for s in survivors:
            distance = np.sqrt((1 - s["x"])**2 + (1 - s["y"])**2)
            ranked.append({**s, "distance_to_ideal": distance})
        ranked.sort(key=lambda x: x["distance_to_ideal"])
        return ranked
    
    def evolve_population(self, population, evaluation_results, agents_pool):
        ordinal = evaluation_results.get("ordinal")
        new_population = []
        
        # 1. ЭЛИТА (20%) — клоны ординара и топ-5
        elite_count = max(1, int(self.config.initial_population_size * 0.2))
        if ordinal:
            original = self._find_group_by_id(population, ordinal["group_id"])
            if original:
                new_population.append(self._clone_group(original, f"elite_{len(self.generations_history)}_ordinal"))
        
        best = evaluation_results.get("best_performers", [])
        for i in range(min(elite_count - 1, len(best))):
            original = self._find_group_by_id(population, best[i]["group_id"])
            if original and original.id not in [g.id for g in new_population]:
                new_population.append(self._clone_group(original, f"elite_{len(self.generations_history)}_{i}"))
        
        # 2. МУТАЦИИ (30%) — вариации лучших
        mutation_count = int(self.config.initial_population_size * 0.3)
        for i in range(mutation_count):
            if best:
                parent_info = best[np.random.randint(0, len(best))]
                parent = self._find_group_by_id(population, parent_info["group_id"])
                if parent:
                    mutated = self._mutate_strategy(parent.params)
                    from src.core.group import AgentGroup
                    g = AgentGroup(id=f"mut_{len(self.generations_history)}_{i}", strategy_name=mutated["base_strategy"])
                    g.params = mutated
                    for a in self._select_agents_for_group(agents_pool, i, mutated):
                        g.add_agent(a)
                    new_population.append(g)
        
        # 3. РАЗНООБРАЗИЕ (20%) — сохраняем не-выживших для покрытия
        all_ids = [g.id for g in population]
        best_ids = [b["group_id"] for b in best[:5]]
        non_best_ids = [gid for gid in all_ids if gid not in best_ids]
        diversity_count = int(self.config.initial_population_size * 0.2)
        for i in range(min(diversity_count, len(non_best_ids))):
            g = self._find_group_by_id(population, non_best_ids[i])
            if g:
                new_population.append(self._clone_group(g, f"div_{len(self.generations_history)}_{i}"))
        
        # 4. НОВИЧКИ (оставшиеся)
        while len(new_population) < self.config.initial_population_size:
            from src.core.group import AgentGroup
            strategies = list(set(g.strategy_name for g in population))
            s = np.random.choice(strategies)
            v = self._create_strategy_variation(s, 9999 + len(new_population), 10000)
            g = AgentGroup(id=f"new_{len(new_population)}", strategy_name=s)
            g.params = v
            for a in self._select_agents_for_group(agents_pool, len(new_population), v):
                g.add_agent(a)
            new_population.append(g)
        
        return new_population
    
    def _mutate_strategy(self, params: Dict) -> Dict:
        mutated = params.copy()
        mutated["exploration_bias"] = np.clip(params["exploration_bias"] + np.random.normal(0, 0.1), 0.05, 0.95)
        mutated["risk_tolerance"] = np.clip(params["risk_tolerance"] + np.random.normal(0, 0.1), 0.05, 0.95)
        mutated["name"] = f"{params['base_strategy']}_mut"
        return mutated
    
    def _clone_group(self, original, new_id: str):
        from src.core.group import AgentGroup
        new_group = AgentGroup(id=new_id, strategy_name=original.strategy_name)
        new_group.params = original.params.copy()
        for agent in original.agents:
            new_group.add_agent(agent)
        return new_group
    
    def _find_group_by_id(self, population, group_id: str):
        for group in population:
            if group.id == group_id:
                return group
        return None
    
    def run_evolution(self, base_strategies: List[str], agents_pool: List[Any], architect) -> Dict[str, Any]:
        print("=" * 60)
        print("ЗАПУСК ПОПУЛЯЦИОННОЙ ЭВОЛЮЦИИ — ПОИСК ОРДИНАРА")
        print(f"Популяция: {self.config.initial_population_size} групп, поколений: {self.config.generations}")
        print("=" * 60)
        
        population = self.generate_initial_population(base_strategies, agents_pool)
        best_ordinal = None
        best_distance = float('inf')
        
        for generation in range(self.config.generations):
            print(f"\n--- Поколение {generation + 1}/{self.config.generations} ---")
            
            eval_results = self.evaluate_population(population, architect)
            dist = eval_results["quadrant_distribution"]
            coverage = eval_results["coverage_percent"]
            ordinal = eval_results.get("ordinal")
            
            print(f"Квадранты: ++={dist['++']}, +-={dist['+-']}, -+={dist['-+']}, --={dist['--']}")
            print(f"Покрытие: {coverage:.1f}%")
            
            if ordinal:
                print(f"ОРДИНАР: {ordinal['group_id']} ({ordinal['strategy']}) | X={ordinal['x']:+.3f} Y={ordinal['y']:+.3f} | d={ordinal['distance_to_ideal']:.4f}")
                if ordinal["distance_to_ideal"] < best_distance:
                    best_distance = ordinal["distance_to_ideal"]
                    best_ordinal = ordinal
                    print(f"🏆 НОВЫЙ ОРДИНАР! {ordinal['group_id']} — расстояние {best_distance:.4f}")
            
            self.generations_history.append({
                "generation": generation,
                "population_size": len(population),
                "coverage": coverage,
                "best_distance": best_distance,
                "quadrant_distribution": dist,
                "ordinal_id": ordinal["group_id"] if ordinal else "нет"
            })
            
            if generation < self.config.generations - 1:
                population = self.evolve_population(population, eval_results, agents_pool)
        
        print("\n" + "=" * 60)
        print("ЭВОЛЮЦИЯ ЗАВЕРШЕНА — ОРДИНАР НАЙДЕН")
        print("=" * 60)
        if best_ordinal:
            print(f"ОРДИНАР: {best_ordinal['group_id']}")
            print(f"Стратегия: {best_ordinal['strategy']}")
            print(f"Качество: {best_ordinal['x']:+.3f}")
            print(f"Выполнение: {best_ordinal['y']:+.3f}")
            print(f"Расстояние до идеала: {best_distance:.4f}")
        
        return {
            "ordinal": best_ordinal,
            "best_distance": best_distance,
            "generations_history": self.generations_history,
            "final_population_size": len(population)
        }
