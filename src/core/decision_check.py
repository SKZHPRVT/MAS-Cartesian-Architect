"""
8 критериев аттестации решения агента перед исполнением.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from enum import Enum


class CriterionStatus(Enum):
    PASS = "✓"
    FAIL = "✗"
    WARN = "⚠"


@dataclass
class DecisionCheck:
    # 1. ИСТИННОСТЬ — данные чистые?
    truth_score: float = 0.0
    truth_threshold: float = 0.85
    
    # 2. РАЦИОНАЛЬНОСТЬ — логика целая?
    rationality_score: float = 0.0
    rationality_threshold: float = 0.85
    
    # 3. ЭФФЕКТИВНОСТЬ — какой ценой?
    efficiency_score: float = 0.0
    efficiency_threshold: float = 0.6
    
    # 4. ЭФФЕКТНОСТЬ — это вообще нужно?
    effectiveness_score: float = 0.0
    effectiveness_threshold: float = 0.85
    
    # 5. АДАПТИВНОСТЬ — не сломается завтра?
    adaptability_score: float = 0.0
    adaptability_threshold: float = 0.6
    
    # 6. ЭТИЧНОСТЬ — не навредит?
    ethics_score: float = 0.0
    ethics_threshold: float = 0.90
    
    # 7. ОСМЫСЛЕННОСТЬ — есть ли смысл вообще?
    meaningfulness_score: float = 0.0
    meaningfulness_threshold: float = 0.7
    
    # 8. ЖИВУЧЕСТЬ — выдержит ли удар?
    resilience_score: float = 0.0
    resilience_threshold: float = 0.6
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _eval(self, score: float, threshold: float) -> CriterionStatus:
        if score >= threshold: return CriterionStatus.PASS
        elif score >= threshold - 0.15: return CriterionStatus.WARN
        else: return CriterionStatus.FAIL
    
    def get_all_statuses(self) -> Dict[str, CriterionStatus]:
        return {
            "Истинность":     self._eval(self.truth_score, self.truth_threshold),
            "Рациональность": self._eval(self.rationality_score, self.rationality_threshold),
            "Эффективность":  self._eval(self.efficiency_score, self.efficiency_threshold),
            "Эффектность":    self._eval(self.effectiveness_score, self.effectiveness_threshold),
            "Адаптивность":   self._eval(self.adaptability_score, self.adaptability_threshold),
            "Этичность":      self._eval(self.ethics_score, self.ethics_threshold),
            "Осмысленность":  self._eval(self.meaningfulness_score, self.meaningfulness_threshold),
            "Живучесть":      self._eval(self.resilience_score, self.resilience_threshold),
        }
    
    def is_approved(self) -> bool:
        return all(s == CriterionStatus.PASS for s in self.get_all_statuses().values())
    
    def get_failed_criterions(self) -> list:
        return [n for n, s in self.get_all_statuses().items() if s != CriterionStatus.PASS]
    
    def overall_score(self) -> float:
        scores = [self.truth_score, self.rationality_score, self.efficiency_score,
                  self.effectiveness_score, self.adaptability_score, self.ethics_score,
                  self.meaningfulness_score, self.resilience_score]
        return sum(scores) / len(scores)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "truth": self.truth_score, "rationality": self.rationality_score,
            "efficiency": self.efficiency_score, "effectiveness": self.effectiveness_score,
            "adaptability": self.adaptability_score, "ethics": self.ethics_score,
            "meaningfulness": self.meaningfulness_score, "resilience": self.resilience_score,
        }
