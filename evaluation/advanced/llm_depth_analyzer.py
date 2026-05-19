import numpy as np
from typing import Dict, List
from collections import Counter

class LLMDepthAnalyzer:
    def __init__(self): self.calls = []
    def record_call(self, call_data: Dict): self.calls.append(call_data)
    def compute_statistics(self) -> Dict:
        if not self.calls: return {}
        total_tokens = sum(c.get('prompt_tokens', 0) + c.get('completion_tokens', 0) for c in self.calls)
        avg_prompt_tokens = np.mean([c.get('prompt_tokens', 0) for c in self.calls])
        avg_completion_tokens = np.mean([c.get('completion_tokens', 0) for c in self.calls])
        avg_reasoning_length = np.mean([len(c.get('reasoning_content', '')) for c in self.calls if c.get('reasoning_content')])
        ttft_values = [c.get('ttft', 0) for c in self.calls if c.get('ttft')]
        avg_ttft = np.mean(ttft_values) if ttft_values else 0
        total_gen_time = sum(c.get('generation_time', 0) for c in self.calls)
        error_types = Counter(c.get('error_type', 'success') for c in self.calls if not c.get('success', True))
        return {'total_calls': len(self.calls), 'total_tokens': total_tokens, 'avg_prompt_tokens': avg_prompt_tokens,
                'avg_completion_tokens': avg_completion_tokens, 'avg_reasoning_length': avg_reasoning_length,
                'avg_ttft': avg_ttft, 'total_generation_time': total_gen_time, 'error_distribution': dict(error_types)}