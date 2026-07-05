# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils import set_seed, get_model_params

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, model_name, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 30)
        )
        
        self.run_dir = f"runs/{model_name}"
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)
        
        self.best_val_f1 = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_auc': [], 'val_auc': []
        }
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{self.config['epochs']}")
        for batch_idx, (data, labels) in enumerate(progress):
            data, labels = data.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            

            if outputs.dim() == 2 and outputs.size(1) == 2:
                outputs = outputs[:, 1:2]
            elif outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            labels = labels.view(-1, 1)
        
            
            loss = self.criterion(outputs, labels.float())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.squeeze().cpu().numpy())
            
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_binary = (all_preds > 0.5).astype(int)
        
        acc = accuracy_score(all_labels, pred_binary)
        precision = precision_score(all_labels, pred_binary, zero_division=0)
        recall = recall_score(all_labels, pred_binary, zero_division=0)
        f1 = f1_score(all_labels, pred_binary, zero_division=0)
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        
        return avg_loss, acc, precision, recall, f1, auc
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc="Validation"):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
            
                if outputs.dim() == 2 and outputs.size(1) == 2:
                    outputs = outputs[:, 1:2]
                elif outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                labels = labels.view(-1, 1)
                
                
                loss = self.criterion(outputs, labels.float())
                
                total_loss += loss.item()
                preds = torch.sigmoid(outputs).detach().cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.squeeze().cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        pred_binary = (all_preds > 0.5).astype(int)
        
        acc = accuracy_score(all_labels, pred_binary)
        precision = precision_score(all_labels, pred_binary, zero_division=0)
        recall = recall_score(all_labels, pred_binary, zero_division=0)
        f1 = f1_score(all_labels, pred_binary, zero_division=0)
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        
        return avg_loss, acc, precision, recall, f1, auc
    
    def train(self, epochs=None):
        if epochs is None:
            epochs = self.config.get('epochs', 30)
        
        print(f"\n{'='*50}")
        print(f"Training {self.model_name}")
        print(f"{'='*50}")
        print(f"Model params: {get_model_params(self.model):.2f}M")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*50}\n")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = self.train_epoch(epoch)
            self.scheduler.step()
            
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = self.validate(epoch)
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/train_acc', train_acc, epoch)
            self.writer.add_scalar('Metrics/val_acc', val_acc, epoch)
            self.writer.add_scalar('Metrics/train_f1', train_f1, epoch)
            self.writer.add_scalar('Metrics/val_f1', val_f1, epoch)
            self.writer.add_scalar('Metrics/train_auc', train_auc, epoch)
            self.writer.add_scalar('Metrics/val_auc', val_auc, epoch)
            
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc,
                    'model_name': self.model_name,
                    'config': self.config
                }, f"{self.run_dir}/best_model.pth")
                print(f"✓ Best model saved (F1: {val_f1:.4f})")
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            with open(f"{self.run_dir}/history.json", 'w') as f:
                json.dump(self.history, f, indent=2)
        
        self.writer.close()
        print(f"\n✓ Training completed. Best val F1: {self.best_val_f1:.4f}")
        return self.history