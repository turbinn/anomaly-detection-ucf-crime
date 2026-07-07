import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import os
import time

from .models import get_model, get_recommended_batch_size
from .data_loader import create_dataloaders
from .utils import save_checkpoint, AverageMeter


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), inputs.size(0))
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': losses.avg,
            'Acc': f'{accuracy:.2f}%'
        })
    
    return losses.avg, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            losses.update(loss.item(), inputs.size(0))
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': losses.avg,
                'Acc': f'{accuracy:.2f}%'
            })
    
    return losses.avg, accuracy, np.array(all_preds), np.array(all_targets)


def train_model(model_name, data_root, batch_size=None, epochs=3, lr=0.0001,
                frames_per_video=16, device='cuda', save_dir='checkpoints'):
    
    if batch_size is None:
        batch_size = get_recommended_batch_size(model_name)
    
    print(f"Training {model_name} with batch_size={batch_size}")
    
    # создаём даталоадеры
    train_loader, test_loader, train_size, test_size = create_dataloaders(
        data_root=data_root,
        model_name=model_name,
        batch_size=batch_size,
        frames_per_video=frames_per_video,
        train_split=0.8
    )
    
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    
    # модель
    model = get_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)
    
    # подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable params: {trainable_params:,}")
    
    # оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5)
    
    # создаём папку для чекпоинтов
    os.makedirs(save_dir, exist_ok=True)
    
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        # обучение
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # валидация
        val_loss, val_acc, _, _ = validate(
            model, test_loader, criterion, device
        )
        
        # обновляем scheduler
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pth")
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"Best model saved: {checkpoint_path}")
    
    return model, history, best_acc