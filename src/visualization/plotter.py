"""
Визуализация квадрата Декарта и результатов фильтрации.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Optional

from ..core.metrics import GroupMetrics, FilterDecision


class CartesianVisualizer:
    """
    Визуализатор для отображения групп в квадрате Декарта.
    """
    
    def __init__(self, 
                 x_label: str = "Качество решения",
                 y_label: str = "Выполнение задач",
                 figsize: tuple = (12, 10)):
        self.x_label = x_label
        self.y_label = y_label
        self.figsize = figsize
    
    def plot_quadrant_map(self,
                          groups_metrics: Dict[str, GroupMetrics],
                          survivors: List[str] = None,
                          title: str = "Квадрат Декарта: Группы агентов",
                          x_axis: str = "solution_quality",
                          y_axis: str = "task_completion_rate",
                          x_threshold: float = 0.0,
                          y_threshold: float = 0.0,
                          show_annotations: bool = True,
                          save_path: Optional[str] = None):
        """
        Строит карту групп в квадрате Декарта.
        """
        survivors = survivors or []
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Рисуем квадранты
        self._draw_quadrants(ax, x_threshold, y_threshold)
        
        # Наносим группы
        for group_id, metrics in groups_metrics.items():
            x = getattr(metrics, x_axis)
            y = getattr(metrics, y_axis)
            
            is_survivor = group_id in survivors
            
            if is_survivor:
                color = '#2ecc71'  # Зелёный
                marker = 'o'
                size = 200
                edge_color = '#27ae60'
                alpha = 0.9
            else:
                color = '#e74c3c'  # Красный
                marker = 'x'
                size = 120
                edge_color = '#c0392b'
                alpha = 0.6
            
            ax.scatter(x, y, c=color, marker=marker, s=size,
                      edgecolors=edge_color, linewidth=1.5, alpha=alpha,
                      zorder=5)
            
            if show_annotations:
                # Краткое имя для аннотации
                short_name = group_id.replace("group_", "G")
                ax.annotate(short_name, (x, y), 
                          xytext=(7, 7), textcoords='offset points',
                          fontsize=8, alpha=0.8,
                          bbox=dict(boxstyle='round,pad=0.3', 
                                  facecolor='white', alpha=0.7))
        
        # Настройка осей
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel(self.x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(self.y_label, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        # Легенда
        legend_elements = [
            mpatches.Patch(facecolor='#2ecc71', alpha=0.3, 
                          label=f'++ Идеально (выжившие: {len(survivors)})'),
            mpatches.Patch(facecolor='#f39c12', alpha=0.15, label='+- Хорошо по X'),
            mpatches.Patch(facecolor='#3498db', alpha=0.15, label='-+ Хорошо по Y'),
            mpatches.Patch(facecolor='#e74c3c', alpha=0.15, label='-- Требует улучшения'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        # Статистика
        stats_text = self._generate_stats_text(groups_metrics, survivors, x_axis, y_axis)
        ax.text(1.2, 1.15, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_evolution(self,
                       history: List[Dict[str, Any]],
                       save_path: Optional[str] = None):
        """
        Строит график эволюции системы по раундам.
        """
        if not history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        rounds = range(1, len(history) + 1)
        
        # График 1: Количество выживших групп
        ax1 = axes[0, 0]
        survivors = [h.get("n_survivors", 0) for h in history]
        rejected = [h.get("n_rejected", 0) for h in history]
        
        ax1.bar(rounds, survivors, label='Ординар (выжившие)', color='#2ecc71', alpha=0.8)
        ax1.bar(rounds, rejected, bottom=survivors, label='Отсеяны', color='#e74c3c', alpha=0.8)
        ax1.set_xlabel('Раунд')
        ax1.set_ylabel('Количество групп')
        ax1.set_title('Результаты фильтрации')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Survival rate
        ax2 = axes[0, 1]
        survival_rates = [h.get("survival_rate", 0) * 100 for h in history]
        ax2.plot(rounds, survival_rates, 'o-', color='#2ecc71', linewidth=2, markersize=8)
        ax2.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='Цель: 30%')
        ax2.set_xlabel('Раунд')
        ax2.set_ylabel('Процент выживших (%)')
        ax2.set_title('Динамика выживаемости групп')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График 3: Распределение по квадрантам
        ax3 = axes[1, 0]
        quadrants = ["++", "+-", "-+", "--"]
        colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']
        
        for q, c in zip(quadrants, colors):
            counts = [h.get("quadrant_distribution", {}).get(q, 0) for h in history]
            ax3.plot(rounds, counts, 'o-', color=c, label=q, linewidth=2, markersize=6)
        
        ax3.set_xlabel('Раунд')
        ax3.set_ylabel('Количество групп')
        ax3.set_title('Распределение по квадрантам')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # График 4: Снижение сложности
        ax4 = axes[1, 1]
        if history and "complexity_stats" in history[0]:
            reductions = [h["complexity_stats"].get("reduction_factor", 1) for h in history]
            ax4.plot(rounds, reductions, 'o-', color='#9b59b6', linewidth=2, markersize=8)
            ax4.set_xlabel('Раунд')
            ax4.set_ylabel('Коэффициент снижения')
            ax4.set_title('Снижение комбинаторной сложности')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Эволюция MAS-системы с Архитектором', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def _draw_quadrants(self, ax, x_threshold: float, y_threshold: float):
        """Рисует квадранты декартова квадрата."""
        # Оси
        ax.axhline(y=y_threshold, color='black', linewidth=1.5, zorder=1)
        ax.axvline(x=x_threshold, color='black', linewidth=1.5, zorder=1)
        
        # Закрашиваем квадранты
        ax.fill_between([x_threshold, 1.15], y_threshold, 1.15, 
                       alpha=0.1, color='#2ecc71', zorder=0)
        ax.fill_between([x_threshold, 1.15], -1.15, y_threshold, 
                       alpha=0.07, color='#f39c12', zorder=0)
        ax.fill_between([-1.15, x_threshold], y_threshold, 1.15, 
                       alpha=0.07, color='#3498db', zorder=0)
        ax.fill_between([-1.15, x_threshold], -1.15, y_threshold, 
                       alpha=0.07, color='#e74c3c', zorder=0)
        
        # Подписи квадрантов
        ax.text(0.7, 0.7, '++', fontsize=20, fontweight='bold', 
               color='#27ae60', alpha=0.3, ha='center', va='center',
               transform=ax.transAxes)
        ax.text(0.7, 0.3, '+-', fontsize=20, fontweight='bold', 
               color='#f39c12', alpha=0.3, ha='center', va='center',
               transform=ax.transAxes)
        ax.text(0.3, 0.7, '-+', fontsize=20, fontweight='bold', 
               color='#3498db', alpha=0.3, ha='center', va='center',
               transform=ax.transAxes)
        ax.text(0.3, 0.3, '--', fontsize=20, fontweight='bold', 
               color='#e74c3c', alpha=0.3, ha='center', va='center',
               transform=ax.transAxes)
    
    def _generate_stats_text(self, 
                             metrics: Dict[str, GroupMetrics],
                             survivors: List[str],
                             x_axis: str,
                             y_axis: str) -> str:
        """Генерирует текст со статистикой."""
        total = len(metrics)
        kept = len(survivors)
        rejected = total - kept
        
        texts = [
            f"Всего групп: {total}",
            f"Выжило: {kept}",
            f"Отсеяно: {rejected}",
            f"Выживаемость: {kept/max(1,total)*100:.1f}%",
            "",
            f"Ось X: {x_axis}",
            f"Ось Y: {y_axis}",
        ]
        
        return "\n".join(texts)