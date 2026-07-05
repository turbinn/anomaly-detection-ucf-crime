# main.py
import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# === ОПРЕДЕЛЯЕМ ПУТИ ===
BASE_DIR = Path(__file__).parent
if (BASE_DIR / "project").exists():
    PROJECT_DIR = BASE_DIR / "project"
else:
    PROJECT_DIR = BASE_DIR

# Добавляем project в путь
sys.path.insert(0, str(PROJECT_DIR))

# Переходим в project для корректных путей
os.chdir(PROJECT_DIR)

from src.data_loader import create_dataloaders
from src.models import get_model
from src.train import Trainer
from src.evaluate import Evaluator, compare_models
from src.inference import VideoAnomalyDetector
from src.utils import set_seed, save_results_to_json


def main():
    parser = argparse.ArgumentParser(description='Обнаружение аномалий в видео (UCF-Crime)')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Путь к конфигурации')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'all'], default='train', 
                       help='Режим работы: train, eval, all')
    parser.add_argument('--model', type=str, choices=['3d_resnet', 'r2plus1d', 'slowfast', 
                                                      'videomae', 'cnn_lstm', 'timesformer', 'all'],
                       default='3d_resnet', help='Модель для обучения/оценки')  # Изменено с 'all' на '3d_resnet'
    parser.add_argument('--resume', type=str, default=None, help='Путь к чекпоинту для продолжения обучения')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог уверенности для инференса')
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        print(f"   Current dir: {os.getcwd()}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Фиксируем seed
    set_seed(42)
    
    # Создаем папки
    os.makedirs('runs', exist_ok=True)
    os.makedirs('demo', exist_ok=True)
    os.makedirs('report', exist_ok=True)
    
    # Определяем модели для запуска
    if args.model == 'all':
        models_to_train = list(config['models'].keys())
    else:
        models_to_train = [args.model]
    
    # Проверяем наличие GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"UCF-Crime Anomaly Detection")
    print(f"{'='*60}")
    print(f"Models to process: {models_to_train}")
    print(f"Data path: {config['data']['path']}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*60}\n")
    
    # Подготавливаем DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config['data']['path'],
        batch_size=config['data']['batch_size'],
        num_frames=config['data']['num_frames'],
        frame_size=config['data']['frame_size'],
        num_workers=config['data'].get('num_workers', 2)
    )
    
    all_results = {}
    
    for model_name in models_to_train:
        print(f"\n{'#'*60}")
        print(f"Processing: {model_name}")
        print(f"{'#'*60}")
        
        # Создаем модель
        model = get_model(model_name, num_classes=2, pretrained=True)
        
        if args.mode in ['train', 'all']:
            # Обучение
            trainer = Trainer(
                model, train_loader, val_loader, 
                config=config['training'], 
                model_name=model_name,
                device=device
            )
            history = trainer.train()
            
            # Загружаем лучшую модель
            model_path = f"runs/{model_name}/best_model.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded best model from: {model_path}")
        
        if args.mode in ['eval', 'all']:
            # Оценка
            evaluator = Evaluator(model, test_loader, model_name, device=device)
            metrics, cm, roc_data, report, preds, labels = evaluator.evaluate(
                threshold=args.threshold
            )
            
            all_results[model_name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'report': report
            }
            
            # Получаем размер модели
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            all_results[model_name]['model_size_mb'] = model_size
    
    # Сравнение моделей
    if len(all_results) > 1:
        comparison = compare_models(all_results)
        
        # Сохраняем сравнение
        import json
        with open('runs/comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Создаем отчет
        create_report(all_results, comparison)
    
    # Инференс на примере видео
    if args.mode in ['all']:
        run_demo_inference(models_to_train, args.threshold, config['data']['path'])
    
    print("\n✅ Done!")


def create_report(results, comparison):
    """Создает текстовый отчет"""
    report_path = 'report/comparison_report.md'
    os.makedirs('report', exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Сравнение архитектур для обнаружения аномалий в видео\n\n")
        f.write("## 1. Таблица сравнения\n\n")
        
        f.write("| Модель | Accuracy | Precision | Recall | F1 | AUC | Inference (ms) | Size (MB) |\n")
        f.write("|--------|----------|-----------|--------|----|-----|----------------|-----------|\n")
        
        for row in comparison:
            f.write(f"| {row['model']} | {row['accuracy']:.3f} | {row['precision']:.3f} | "
                   f"{row['recall']:.3f} | {row['f1']:.3f} | {row['auc']:.3f} | "
                   f"{row['inference_time_ms']:.1f} | {row['model_size_mb']:.1f} |\n")
        
        f.write("\n## 2. Лучшая модель\n\n")
        f.write(f"**Лучшая модель:** {comparison[0]['model']}\n")
        f.write(f"- F1 Score: {comparison[0]['f1']:.4f}\n")
        f.write(f"- Accuracy: {comparison[0]['accuracy']:.4f}\n")
        f.write(f"- AUC: {comparison[0]['auc']:.4f}\n\n")
        
        f.write("## 3. Анализ ошибок\n\n")
        f.write("### Типичные ошибки моделей:\n")
        f.write("1. **False Positives (ложные срабатывания):** Модели часто ошибаются на сценах с быстрым движением или изменением освещения.\n")
        f.write("2. **False Negatives (пропуски):** Сложные аномалии или аномалии на заднем плане часто пропускаются.\n")
        f.write("3. **Ошибки на границах:** Переходы между нормальным и аномальным поведением нечеткие.\n\n")
        
        f.write("## 4. Рекомендации по улучшению\n\n")
        f.write("1. **Увеличение данных:** Добавить больше видео с различными условиями.\n")
        f.write("2. **Аугментации:** Использовать временные аугментации (изменение скорости, флип во времени).\n")
        f.write("3. **Постобработка:** Применять скользящее среднее для сглаживания предсказаний.\n")
        f.write("4. **Ансамблирование:** Комбинировать предсказания нескольких моделей.\n")
    
    print(f"✓ Report saved to: {report_path}")


def run_demo_inference(models, threshold, data_path):
    """Запускает демонстрационный инференс"""
    # Ищем тестовое видео
    test_videos = list(Path(data_path).rglob("*.mp4"))[:3]
    
    if not test_videos:
        print("⚠️ No test videos found for demo")
        return
    
    print("\n" + "="*60)
    print("DEMO INFERENCE")
    print("="*60)
    
    # Используем лучшую модель из runs
    best_model_path = None
    best_f1 = -1
    best_model_name = None
    
    for model_name in models:
        history_path = f"runs/{model_name}/history.json"
        if os.path.exists(history_path):
            import json
            with open(history_path, 'r') as f:
                history = json.load(f)
            if history['val_f1'] and max(history['val_f1']) > best_f1:
                best_f1 = max(history['val_f1'])
                best_model_path = f"runs/{model_name}/best_model.pth"
                best_model_name = model_name
    
    if best_model_path and os.path.exists(best_model_path):
        print(f"Using best model: {best_model_name} (F1: {best_f1:.4f})")
        
        # Загружаем модель
        model = get_model(best_model_name, num_classes=2, pretrained=False)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        detector = VideoAnomalyDetector(
            model, best_model_name, 
            num_frames=16, frame_size=224
        )
        detector.threshold = threshold
        
        # Тестируем на нескольких видео
        for video_path in test_videos[:2]:
            print(f"\nProcessing: {video_path.name}")
            result = detector.predict_video(str(video_path))
            
            print(f"  Prediction: {'⚠️ ANOMALY' if result['prediction'] else '✅ NORMAL'}")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Clips analyzed: {result['num_clips']}")
    else:
        print("⚠️ No trained model found. Please run training first.")
    
    print("="*60)


if __name__ == "__main__":
    main()