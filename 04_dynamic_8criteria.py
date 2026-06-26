"""
04 — ДИНАМИКА: 8 КРИТЕРИЕВ. Те же данные что в 03.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from src.core.agent import Agent
from src.core.group import AgentGroup

np.random.seed(42)
agents = [Agent(id=i, position=np.random.rand(2)*10, radius_of_influence=1.0) for i in range(100)]
strategies = ["greedy", "genetic", "swarm", "hybrid", "auction", "ml_based", "random"]
SC = {'greedy':'#e74c3c','genetic':'#3498db','swarm':'#2ecc71','hybrid':'#9b59b6','auction':'#f39c12','ml_based':'#1abc9c','random':'#95a5a6'}

print("=" * 60)
print("04 — ДИНАМИКА: 8 КРИТЕРИЕВ")
print("=" * 60)

population = []
for i in range(50):
    s = strategies[i % 7]
    g = AgentGroup(id=f"pop_{i}_{s}", strategy_name=s)
    for j in range(2): g.add_agent(agents[(i*2+j) % 100])
    population.append(g)

history = []
snapshots = {}
best_ordinal = None
best_score = 0

for gen in range(10):
    gen_data = []
    for g in population:
        m = g.evaluate_performance()
        dc = g._run_decision_check()
        x, y = m.solution_quality, m.task_completion_rate
        q = "++" if x>=0.5 and y>=0.5 else "+-" if x>=0.5 else "-+" if y>=0.5 else "--"
        score = dc.overall_score()
        passed = dc.is_approved()
        gen_data.append({"x": x, "y": y, "q": q, "strategy": g.strategy_name, "id": g.id,
                         "score": score, "passed": passed, "dc": dc})
    
    snapshots[gen] = gen_data
    n_pp = sum(1 for r in gen_data if r['q']=="++")
    n_pm = sum(1 for r in gen_data if r['q']=="+-")
    n_mp = sum(1 for r in gen_data if r['q']=="-+")
    n_mm = sum(1 for r in gen_data if r['q']=="--")
    n_ord = sum(1 for r in gen_data if r['passed'])
    best = max(gen_data, key=lambda r: r['score'])
    
    print(f"Поколение {gen+1:2}: ++={n_pp:2} +-={n_pm:2} -+={n_mp:2} --={n_mm:2} | Ординаров: {n_ord:2} | лучший: {best['strategy']} score={best['score']:.2f}")
    if best['score'] > best_score:
        best_score = best['score']; best_ordinal = best
        print(f"  🏆 ОРДИНАР: {best['id']} score={best_score:.2f}")
    
    history.append({"gen": gen+1, "n_ord": n_ord, "best_score": best['score'], "n_pp": n_pp, "n_mm": n_mm})
    
    # Та же эволюция что в 03
    gen_data.sort(key=lambda r: r['score'], reverse=True)
    surv = [r for r in gen_data[:10]]
    rest = [r for r in gen_data[10:] if r['q']!='++']; np.random.shuffle(rest)
    surv.extend(rest[:15])
    ids = set(r['id'] for r in surv)
    population = [g for g in population if g.id in ids]
    while len(population) < 50:
        s = strategies[len(population)%7]
        g = AgentGroup(id=f"new_{gen}_{len(population)}_{s}", strategy_name=s)
        for j in range(2): g.add_agent(agents[np.random.randint(0,100)])
        population.append(g)

print(f"\nОРДИНАР: {best_ordinal['strategy']} score={best_score:.2f}")

# Графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
gens = range(1, 11)

ax1 = axes[0,0]
ax1.bar(gens, [h['n_ord'] for h in history], color='#9b59b6', alpha=0.7)
ax1.set_title('КОЛИЧЕСТВО ОРДИНАРОВ'); ax1.grid(True, alpha=0.3)

ax2 = axes[0,1]
ax2.plot(gens, [h['best_score'] for h in history], 'o-', color='purple', linewidth=2, markersize=8)
ax2.set_title('ЛУЧШИЙ SCORE'); ax2.grid(True, alpha=0.3)

# Радар поколения 1
ax3 = plt.subplot(2, 2, 3, projection='polar')
labels = ['Истина','Рацио','Эффект-ть','Эффект-сть','Адапт','Этика','Осмысл','Живучесть']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])
for r in snapshots[0]:
    dc = r['dc']
    vals = [dc.truth_score, dc.rationality_score, dc.efficiency_score, dc.effectiveness_score,
            dc.adaptability_score, dc.ethics_score, dc.meaningfulness_score, dc.resilience_score]
    vals = np.concatenate([vals, [vals[0]]])
    ax3.plot(angles, vals, '-', color=SC[r['strategy']], alpha=0.3 if not r['passed'] else 0.8, linewidth=1)
ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(labels, fontsize=6)
ax3.set_title(f'ПОКОЛЕНИЕ 1 (ординаров: {history[0]["n_ord"]})')

# Радар ординара
ax4 = plt.subplot(2, 2, 4, projection='polar')
dc = best_ordinal['dc']
vals = [dc.truth_score, dc.rationality_score, dc.efficiency_score, dc.effectiveness_score,
        dc.adaptability_score, dc.ethics_score, dc.meaningfulness_score, dc.resilience_score]
vals = np.concatenate([vals, [vals[0]]])
ax4.fill(angles, vals, alpha=0.25, color='#9b59b6')
ax4.plot(angles, vals, 'o-', color='#9b59b6', linewidth=3, markersize=10)
ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(labels, fontsize=6)
ax4.set_title(f'ОРДИНАР: {best_ordinal["strategy"]} ({best_score:.2f})')

plt.suptitle(f'ДИНАМИКА: 8 КРИТЕРИЕВ | ОРДИНАР: {best_ordinal["strategy"]} score={best_score:.2f}', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
print("\nГотово!")
