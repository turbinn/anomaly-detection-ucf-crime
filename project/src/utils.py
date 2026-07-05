# src/utils.py
import os
import json
import random
import numpy as np
import torch
from datetime import datetime

def set_seed(seed=42):
    """Фиксация seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_params(model):
    """Подсчёт параметров модели"""
    return sum(p.numel() for p in model.parameters()) / 1e6  # в миллионах

def save_results_to_json(results, model_name, save_path):
    """Сохранение результатов в JSON"""
    os.makedirs(save_path, exist_ok=True)
    
    # Преобразуем numpy массивы в списки
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, np.float32) or isinstance(value, np.float64):
            results_serializable[key] = float(value)
        else:
            results_serializable[key] = value
    
    results_serializable['model_name'] = model_name
    results_serializable['timestamp'] = datetime.now().isoformat()
    
    with open(os.path.join(save_path, f'{model_name}_results.json'), 'w') as f:
        json.dump(results_serializable, f, indent=2)

def load_results_from_json(model_name, results_path):
    """Загрузка результатов из JSON"""
    with open(os.path.join(results_path, f'{model_name}_results.json'), 'r') as f:
        return json.load(f)

def format_time(seconds):
    """Форматирование времени"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"