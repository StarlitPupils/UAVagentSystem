import json
import numpy as np
from typing import Dict, List, Any

class CollaborationAnalyzer:
    def __init__(self): self.tasks = []
    def analyze_task(self, task_log: Dict) -> Dict:
        events = task_log.get('events', [])
        agent_messages = [e for e in events if e.get('type') == 'agent_message']
        rounds = len(agent_messages)
        idle_ratios = {}
        return {'task_id': task_log.get('task_id'), 'collaboration_rounds': rounds, 'idle_ratios': idle_ratios}
    def compute_decision_consistency(self, plans: List[Dict]) -> float:
        if len(plans) < 2: return 1.0
        similarities = []
        for i in range(len(plans)):
            for j in range(i+1, len(plans)):
                set_i, set_j = set(plans[i].keys()), set(plans[j].keys())
                common = 0
                for k in set_i & set_j:
                    if plans[i].get(k) == plans[j].get(k): common += 1
                union = len(set_i | set_j)
                sim = common / union if union > 0 else 0
                similarities.append(sim)
        return np.mean(similarities) if similarities else 0.0
    def compute_fallback_quality(self, llm_plans: List[Dict], fallback_plans: List[Dict]) -> float:
        if not llm_plans or not fallback_plans: return 0.0
        pairs = min(len(llm_plans), len(fallback_plans))
        sims = [self._plan_similarity(llm_plans[i], fallback_plans[i]) for i in range(pairs)]
        return np.mean(sims) if sims else 0.0
    def _plan_similarity(self, plan1: Dict, plan2: Dict) -> float:
        set1, set2 = set(plan1.keys()), set(plan2.keys())
        common = sum(1 for k in set1 & set2 if plan1.get(k) == plan2.get(k))
        union = len(set1 | set2)
        return common / union if union > 0 else 0.0