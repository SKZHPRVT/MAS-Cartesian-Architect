"""
03 — ДИНАМИКА: КВАДРАТ ДЕКАРТА. 50 групп × 10 поколений.
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
print("03 — ДИНАМИКА: КВАДРАТ ДЕКАРТА")
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
best_d = float('inf')

for gen in range(10):
    gen_data = []
    for g in population:
        m = g.evaluate_performance()
        x, y = m.solution_quality, m.task_completion_rate
        q = "++" if x>=0.5 and y>=0.5 else "+-" if x>=0.5 else "-+" if y>=0.5 else "--"
        d = np.sqrt((1-x)**2 + (1-y)**2)
        gen_data.append({"x": x, "y": y, "q": q, "strategy": g.strategy_name, "id": g.id, "d": d})
    
    snapshots[gen] = gen_data
    n_pp = sum(1 for r in gen_data if r['q']=="++")
    n_pm = sum(1 for r in gen_data if r['q']=="+-")
    n_mp = sum(1 for r in gen_data if r['q']=="-+")
    n_mm = sum(1 for r in gen_data if r['q']=="--")
    best = min(gen_data, key=lambda r: r['d'])
    
    print(f"Поколение {gen+1:2}: ++={n_pp:2} +-={n_pm:2} -+={n_mp:2} --={n_mm:2} | лучший: {best['strategy']} d={best['d']:.3f}")
    if best['d'] < best_d:
        best_d = best['d']; best_ordinal = best
        print(f"  🏆 ОРДИНАР: {best['id']} d={best_d:.4f}")
    
    history.append({"gen": gen+1, "n_pp": n_pp, "n_pm": n_pm, "n_mp": n_mp, "n_mm": n_mm, "best_d": best_d})
    
    # Эволюция
    gen_data.sort(key=lambda r: r['d'])
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

print(f"\nОРДИНАР: {best_ordinal['strategy']} X={best_ordinal['x']:.3f} Y={best_ordinal['y']:.3f} d={best_d:.4f}")

# Графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
gens = range(1, 11)

ax1 = axes[0,0]
for q,c,k in [("++","green","n_pp"),("+-","orange","n_pm"),("-+","blue","n_mp"),("--","red","n_mm")]:
    ax1.plot(gens, [h[k] for h in history], 'o-', color=c, label=q, linewidth=2, markersize=6)
ax1.set_title('РАСПРЕДЕЛЕНИЕ ПО КВАДРАНТАМ'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2 = axes[0,1]
ax2.plot(gens, [h['best_d'] for h in history], 'o-', color='purple', linewidth=2, markersize=8)
ax2.set_title('СХОДИМОСТЬ ОРДИНАРА (d → 0)'); ax2.grid(True, alpha=0.3)

for ax, gen, title in [(axes[1,0], 0, f'ПОКОЛЕНИЕ 1 (++={history[0]["n_pp"]}, --={history[0]["n_mm"]})'),
                        (axes[1,1], 9, f'ПОКОЛЕНИЕ 10 (++={history[9]["n_pp"]}, --={history[9]["n_mm"]})')]:
    ax.axhline(y=0.5,color='black',lw=1); ax.axvline(x=0.5,color='black',lw=1)
    for q,c,x1,x2,y1,y2 in [("++",'green',0.5,1,0.5,1),("+-",'orange',0.5,1,0,0.5),("-+",'blue',0,0.5,0.5,1),("--",'red',0,0.5,0,0.5)]:
        ax.fill_between([x1,x2], y1,y2, alpha=0.06, color=c)
    for r in snapshots[gen]:
        ax.scatter(r['x'], r['y'], c=SC[r['strategy']], s=25, alpha=0.6, edgecolors='white', linewidth=0.2)
    if gen == 9:
        ax.scatter([best_ordinal['x']], [best_ordinal['y']], s=400, c='gold', marker='*', edgecolors='darkred', linewidth=2.5, zorder=10)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title(title); ax.grid(True, alpha=0.3)

plt.suptitle(f'ДИНАМИКА: КВАДРАТ ДЕКАРТА | ОРДИНАР: {best_ordinal["strategy"]} d={best_d:.3f}', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
print("\nГотово!")
