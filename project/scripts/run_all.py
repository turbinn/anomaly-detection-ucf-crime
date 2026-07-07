#!/usr/bin/env python3
"""Скрипт для запуска обучения и тестирования всех моделей."""
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Запуск всех моделей')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Количество эпох для обучения')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Размер батча')
    args = parser.parse_args()

    models = [
        '3d_resnet',
        'cnn_lstm',
        'x3d_m',
        'slowfast',
        'timesformer'
    ]

    print("=" * 60)
    print("Запуск обучения и тестирования всех 5 моделей")
    print(f"Эпох: {args.epochs}, Batch size: {args.batch_size}")
    print("=" * 60)

    for model in models:
        print(f"\n{'#' * 60}")
        print(f"Обработка модели: {model}")
        print(f"{'#' * 60}")

        # Обучение
        print(f"Обучение {model}...")
        result = subprocess.run(
            [sys.executable, 'scripts/train_model.py',
             '--model', model,
             '--epochs', str(args.epochs),
             '--batch_size', str(args.batch_size)],
            capture_output=False
        )

        # Тестирование
        print(f" Тестирование {model}...")
        result = subprocess.run(
            [sys.executable, 'scripts/eval_model.py',
             '--model', model],
            capture_output=False
        )

    print("\n" + "=" * 60)
    print("Все 5 моделей обработаны!")
    print("=" * 60)

if __name__ == "__main__":
    main()