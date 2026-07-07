# scripts/train_all_models.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.train import train_model
from src.utils import compute_metrics, save_metrics
from src.models import get_model_params


def train_all_models(data_root, save_dir='checkpoints', results_dir='results'):
    
    # модели для сравнения
    models_to_train = [
        "r3d_18",
        "mc3_18", 
        "r2plus1d_18",
        "mvit",
        "cnn_lstm"
    ]
    
    results = {}
    
    for model_name in models_to_train:
        print("\n" + "="*50)
        print(f"Training {model_name}")
        print("="*50)
        
        try:
            # обучаем модель
            model, history, best_acc = train_model(
                model_name=model_name,
                data_root=data_root,
                epochs=3,
                lr=0.0001,
                frames_per_video=16,
                save_dir=save_dir
            )
            
            results[model_name] = {
                'best_accuracy': best_acc,
                'history': history,
                'params': get_model_params(model_name)
            }
            
            print(f"{model_name} completed. Best accuracy: {best_acc:.2f}%")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # сохраняем общие результаты
    import json
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # замени на путь к твоим данным
    DATA_ROOT = "T:/recognize_anomaly/data/ucf_crime_small"
    
    results = train_all_models(DATA_ROOT)
    
    print("\n" + "="*50)
    print("Final Results:")
    print("="*50)
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name}: {result['best_accuracy']:.2f}% ({result['params']:,} params)")
        else:
            print(f"{model_name}: ERROR - {result['error']}")