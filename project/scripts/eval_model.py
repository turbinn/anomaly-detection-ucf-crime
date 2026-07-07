#!/usr/bin/env python3
"""Скрипт для тестирования модели."""
import argparse
import sys
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import create_dataloaders
from models import get_model


def main():
    parser = argparse.ArgumentParser(description='Тестирование модели для UCF-Crime')
    parser.add_argument('--model', type=str, required=True,
                        help='Название модели: 3d_resnet, cnn_lstm, x3d_m, slowfast, timesformer')
    parser.add_argument('--data_path', type=str,
                        default='/content/anomaly-detection-ucf-crime/data/ucf_crime',
                        help='Путь к данным')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Размер батча')
    args = parser.parse_args()

    print(f"Тестирование модели: {args.model}")

    _, _, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_frames=16,
        frame_size=224,
        num_workers=2
    )

    if len(test_loader) == 0:
        print("Ошибка: данные не загружены!")
        return

    #проверка, есть ли чекпоинт
    checkpoint_path = f'runs/{args.model}/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Ошибка: чекпоинт {checkpoint_path} не найден!")
        print("Сначала обучите модель командой: python scripts/train_model.py --model " + args.model)
        return

    model = get_model(args.model, num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)

            if outputs.dim() == 2 and outputs.size(1) == 2:
                outputs = outputs[:, 1:2]
            elif outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.squeeze().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(all_labels, pred_binary),
        'precision': precision_score(all_labels, pred_binary, zero_division=0),
        'recall': recall_score(all_labels, pred_binary, zero_division=0),
        'f1': f1_score(all_labels, pred_binary, zero_division=0),
        'auc': roc_auc_score(all_labels, all_preds)
    }

    print(f"\n=== Test Results for {args.model} ===")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    cm = confusion_matrix(all_labels, pred_binary)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")


if __name__ == "__main__":
    main()