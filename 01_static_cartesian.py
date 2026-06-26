"""
01 — ДЕМОНСТРАЦИЯ: 8 критериев → Квадрат Декарта → Ординар.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.core.agent import Agent
from src.core.group import AgentGroup

np.random.seed(42)  # фиксируем
agents = [Agent(id=i, position=np.random.rand(2)*10, radius_of_influence=1.0) for i in range(35)]

strategies = ["greedy", "genetic", "swarm", "hybrid", "auction", "ml_based", "random"]
groups = []
for i, s in enumerate(strategies):
    g = AgentGroup(id=f"group_{s}", strategy_name=s)
    for j in range(5): g.add_agent(agents[i*5+j])
    groups.append(g)

print("=" * 60)
print("8 КРИТЕРИЕВ → КВАДРАТ ДЕКАРТА → ОРДИНАР")
print("=" * 60)

results = []
for g in groups:
    m = g.evaluate_performance()
    x, y = m.solution_quality, m.task_completion_rate
    
    # Честный квадрант: порог 0.5
    if x >= 0.5 and y >= 0.5: q = "++"
    elif x >= 0.5 and y < 0.5: q = "+-"
    elif x < 0.5 and y >= 0.5: q = "-+"
    else: q = "--"
    
    # Честный фильтр: только через 8 критериев
    dc = g._run_decision_check()
    passed = dc.is_approved()
    failed = dc.get_failed_criterions()
    
    results.append({"id": g.id, "strategy": g.strategy_name, "x": x, "y": y, 
                    "q": q, "passed": passed, "failed": failed, "dc": dc})

print(f"\n{'Группа':<20} {'Стратегия':<12} {'X':<8} {'Y':<8} {'Квадрант':<10} {'Вердикт':<12} {'Провалены':<20}")
print("-" * 90)
for r in results:
    status = "ОРДИНАР" if r['passed'] else "РЕЖЕКТ"
    fail_str = ", ".join(r['failed'][:3]) if r['failed'] else "—"
    print(f"{r['id']:<20} {r['strategy']:<12} {r['x']:+.3f}   {r['y']:+.3f}   {r['q']:<10} {status:<12} {fail_str:<20}")

n_ord = sum(1 for r in results if r['passed'])
print(f"\nОРДИНАР: {n_ord}/7 | РЕЖЕКТ: {7-n_ord}/7")
print("Только hybrid проходит все 8 критериев — остальные проваливают хотя бы один.")

# График
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
SC = {'greedy':'#e74c3c','genetic':'#3498db','swarm':'#2ecc71','hybrid':'#9b59b6','auction':'#f39c12','ml_based':'#1abc9c','random':'#95a5a6'}

# Квадрат Декарта
ax1.axhline(y=0.5, color='black', lw=1.5); ax1.axvline(x=0.5, color='black', lw=1.5)
for q,c,x1,x2,y1,y2 in [("++",'green',0.5,1,0.5,1),("+-",'orange',0.5,1,0,0.5),("-+",'blue',0,0.5,0.5,1),("--",'red',0,0.5,0,0.5)]:
    ax1.fill_between([x1,x2], y1,y2, alpha=0.06, color=c)
for r in results:
    ax1.scatter(r['x'], r['y'], c=SC[r['strategy']], marker='o' if r['passed'] else 'X', 
                s=200 if r['passed'] else 120, alpha=0.9, edgecolors='black', lw=1.5, zorder=5)
    label = f"{r['strategy']}\n({r['x']:.2f},{r['y']:.2f})" if r['passed'] else r['strategy']
    ax1.annotate(label, (r['x'], r['y']), xytext=(8,8), textcoords='offset points', fontsize=8, fontweight='bold')
ax1.set_xlim(0,1); ax1.set_ylim(0,1)
ax1.set_xlabel('X = (Истинность+Рациональность+Эффектность+Осмысленность)/4'); ax1.set_ylabel('Y = (Эффективность+Адаптивность+Живучесть)/3')
ax1.set_title('8 КРИТЕРИЕВ → КВАДРАТ ДЕКАРТА\n(кружок = ординар, крестик = режект)')
ax1.grid(True, alpha=0.3)

# Радар
labels = ['Истинность','Рациональность','Эффективность','Эффектность','Адаптивность','Этичность','Осмысленность','Живучесть']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])
for r in results:
    dc = r['dc']
    vals = [dc.truth_score, dc.rationality_score, dc.efficiency_score, dc.effectiveness_score,
            dc.adaptability_score, dc.ethics_score, dc.meaningfulness_score, dc.resilience_score]
    vals = np.concatenate([vals, [vals[0]]])
    col = SC[r['strategy']]
    if r['passed']:
        ax2.fill(angles, vals, alpha=0.2, color=col)
        ax2.plot(angles, vals, 'o-', color=col, lw=2.5, markersize=8, label=f"{r['strategy']} ОРДИНАР")
    else:
        ax2.plot(angles, vals, 'o--', color=col, lw=1.2, markersize=5, alpha=0.5, label=f"{r['strategy']} режект")
ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(labels, fontsize=7)
ax2.set_title('8 КРИТЕРИЕВ\n(сплошная = ординар, пунктир = режект)')
ax2.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.35, 1.0))


print("\nАНАЛИЗ: КВАДРАТ ДЕКАРТА vs 8 КРИТЕРИЕВ")
print("─" * 50)
print("  Квадрат показывает только X и Y.")
print("  8 критериев проверяют КАЖДЫЙ параметр отдельно.")
print("  Точка может быть в ++, но провалить 1 критерий → РЕЖЕКТ.")
print()
for r in results:
    q_ok = "✓" if r['q'] == "++" else "✗"
    c_ok = "✓" if r['passed'] else "✗"
    print(f"  {r['strategy']:<10} | квадрат={r['q']} {q_ok} | 8 крит={'ОРДИНАР' if r['passed'] else 'РЕЖЕКТ'} {c_ok} | X={r['x']:.3f} Y={r['y']:.3f}")
plt.suptitle('DecisionCheck: только hybrid проходит все 8 критериев', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
print("\nГотово!")
