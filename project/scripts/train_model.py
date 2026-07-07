#!/usr/bin/env python3
"""Скрипт для обучения модели."""
import argparse
import sys
import os
import torch

#путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import create_dataloaders
from models import get_model
from train import Trainer
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Обучение модели для UCF-Crime')
    parser.add_argument('--model', type=str, required=True,
                        help='Название модели: 3d_resnet, cnn_lstm, x3d_m, slowfast, timesformer')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--data_path', type=str,
                        default='/content/anomaly-detection-ucf-crime/data/ucf_crime',
                        help='Путь к данным')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Количество кадров в клипе')
    parser.add_argument('--frame_size', type=int, default=224,
                        help='Размер кадра')
    args = parser.parse_args()

    set_seed(42)

    print(f"Обучение модели: {args.model}")
    print(f"Эпох: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Путь к данным: {args.data_path}")

    # Проверяем, что данные есть
    if not os.path.exists(args.data_path):
        print(f"Ошибка: папка {args.data_path} не найдена!")
        return

    train_loader, val_loader, _ = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        num_workers=2
    )

    if len(train_loader) == 0:
        print("Ошибка: данные не загружены!")
        return

    model = get_model(args.model, num_classes=2, pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Параметров модели: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Устройство: {device}")

    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 0.0001
    }

    trainer = Trainer(model, train_loader, val_loader, config, args.model, device)
    trainer.train()
    print(f"Обучение {args.model} завершено!")


if __name__ == "__main__":
    main()