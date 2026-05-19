import json
import os
from typing import Any, Dict
def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
def save_json(filepath: str, data: Dict[str, Any]):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)