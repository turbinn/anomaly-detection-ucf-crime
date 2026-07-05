import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import time
import os
from src.utils import get_model_params

class Evaluator:
    def __init__(self, model, test_loader, model_name, device=None):
        self.model = model
        self.test_loader = test_loader
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        self.run_dir = f"runs/{model_name}"
        os.makedirs(self.run_dir, exist_ok=True)
        
    def evaluate(self, threshold=0.5):
        """Полная оценка модели"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for data, labels in tqdm(self.test_loader, desc=f"Testing {self.model_name}"):
                data = data.to(self.device)
                
                start_time = time.time()
                outputs = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                preds = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.squeeze().cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_binary = (all_preds > threshold).astype(int)
        
        # Метрики
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        metrics = {
            'accuracy': accuracy_score(all_labels, pred_binary),
            'precision': precision_score(all_labels, pred_binary, zero_division=0),
            'recall': recall_score(all_labels, pred_binary, zero_division=0),
            'f1': f1_score(all_labels, pred_binary, zero_division=0),
            'auc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5,
            'inference_time_avg': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'threshold': threshold,
            'model_params_m': get_model_params(self.model)
        }
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, pred_binary)
        
        # ROC
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        
        # Classification Report
        report = classification_report(all_labels, pred_binary, 
                                       target_names=['Normal', 'Anomaly'])
        
        # Сохраняем результаты
        results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'classification_report': report,
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist()
        }
        
        with open(f"{self.run_dir}/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Визуализация
        self.plot_results(metrics, cm, (fpr, tpr), all_preds, all_labels)
        
        return metrics, cm, (fpr, tpr), report, all_preds, all_labels
    
    def plot_results(self, metrics, cm, roc_data, preds, labels):
        """Визуализация результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Evaluation Results - {self.model_name}', fontsize=16)
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticklabels(['Normal', 'Anomaly'])
        axes[0, 0].set_yticklabels(['Normal', 'Anomaly'])
        
        # 2. ROC Curve
        fpr, tpr = roc_data
        axes[0, 1].plot(fpr, tpr, 'b-', label=f'AUC = {metrics["auc"]:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve (AUC: {metrics["auc"]:.3f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of predictions
        axes[1, 0].hist(preds[labels==0], bins=30, alpha=0.7, label='Normal', color='green')
        axes[1, 0].hist(preds[labels==1], bins=30, alpha=0.7, label='Anomaly', color='red')
        axes[1, 0].axvline(0.5, color='black', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('Prediction Score')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Prediction Score Distribution')
        axes[1, 0].legend()
        
        # 4. Metrics summary
        metrics_text = f"Accuracy:  {metrics['accuracy']:.4f}\n"
        metrics_text += f"Precision: {metrics['precision']:.4f}\n"
        metrics_text += f"Recall:    {metrics['recall']:.4f}\n"
        metrics_text += f"F1 Score:  {metrics['f1']:.4f}\n"
        metrics_text += f"AUC:       {metrics['auc']:.4f}\n"
        metrics_text += f"Inference: {metrics['inference_time_avg']*1000:.1f}ms (±{metrics['inference_time_std']*1000:.1f})"
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=14, va='center', 
                       fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Metrics Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.run_dir}/evaluation_results.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def get_failed_examples(self, preds, labels, n_examples=3):
        """Находит примеры ошибок модели"""
        pred_binary = (preds > 0.5).astype(int)
        
        # False Positives (аномалия предсказана, но её нет)
        fp_indices = np.where((pred_binary == 1) & (labels == 0))[0]
        # False Negatives (аномалия не обнаружена)
        fn_indices = np.where((pred_binary == 0) & (labels == 1))[0]
        
        # Выбираем по n_examples
        fp_examples = fp_indices[:n_examples]
        fn_examples = fn_indices[:n_examples]
        
        # Если недостаточно примеров, дополняем
        if len(fp_examples) < n_examples:
            # Добираем из самых близких к порогу
            close_to_threshold = np.argsort(np.abs(preds[labels == 1] - 0.5))[:n_examples]
            fn_examples = np.concatenate([fn_examples, close_to_threshold])
        
        return fp_examples, fn_examples


def compare_models(results_dict):
    """Сравнение всех моделей"""
    print("\n" + "="*80)
    print("COMPARISON OF ALL MODELS")
    print("="*80)
    
    comparison = []
    for model_name, results in results_dict.items():
        metrics = results['metrics']
        comparison.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'inference_time_ms': metrics['inference_time_avg'] * 1000,
            'model_size_mb': results.get('model_size_mb', metrics.get('model_params_m', 0))
        })
    
    # Сортировка по F1
    comparison = sorted(comparison, key=lambda x: x['f1'], reverse=True)
    
    # Печать таблицы
    print("\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15} {:<12}".format(
        'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Inference (ms)', 'Size (MB)'
    ))
    print("-"*95)
    for row in comparison:
        print("{:<15} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<15.1f} {:<12.1f}".format(
            row['model'], row['accuracy'], row['precision'], row['recall'],
            row['f1'], row['auc'], row['inference_time_ms'], row['model_size_mb']
        ))
    print("="*80)
    print(f"🏆 Best model: {comparison[0]['model']} (F1: {comparison[0]['f1']:.4f})")
    print("="*80)
    
    return comparison