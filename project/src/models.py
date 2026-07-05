# src/models.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r3d_18, r2plus1d_18
import timm

# 1. 3D ResNet-18
class VideoClassifier3DResNet(nn.Module):
    """3D ResNet для классификации видео"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = r3d_18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        output = self.model(x)
        # Если num_classes=2, возвращаем только второй класс (аномалия)
        if output.size(1) == 2:
            return output[:, 1:2]  # [B, 1]
        return output

# 2. 3D ResNet + 1D (R(2+1)D)
class VideoClassifierR2Plus1D(nn.Module):
    """R(2+1)D для классификации видео"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = r2plus1d_18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# 3. SlowFast (упрощенная версия)
class VideoClassifierSlowFast(nn.Module):
    """SlowFast архитектура"""
    def __init__(self, num_classes=2, pretrained=True, alpha=4):
        super().__init__()
        self.alpha = alpha
        
        # Slow path
        self.slow_path = r3d_18(pretrained=pretrained)
        slow_in_features = self.slow_path.fc.in_features
        self.slow_path.fc = nn.Identity()
        
        # Fast path
        self.fast_path = r3d_18(pretrained=pretrained)
        fast_in_features = self.fast_path.fc.in_features
        self.fast_path.fc = nn.Identity()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(slow_in_features + fast_in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # Slow path: берем каждый alpha-й кадр
        slow_indices = list(range(0, T, self.alpha))
        if len(slow_indices) < 2:
            slow_indices = [0, min(T-1, 1)]
        slow_frames = x[:, :, slow_indices, :, :]
        
        # Fast path: все кадры, но уменьшенное разрешение
        fast_frames = nn.functional.interpolate(
            x.permute(0, 2, 1, 3, 4),
            scale_factor=0.5,
            mode='trilinear'
        ).permute(0, 2, 1, 3, 4)
        
        slow_feat = self.slow_path(slow_frames)
        fast_feat = self.fast_path(fast_frames)
        
        combined = torch.cat([slow_feat, fast_feat], dim=1)
        return self.fusion(combined)

# 4. VideoMAE (эмуляция через ViT)
class VideoClassifierVideoMAE(nn.Module):
    """VideoMAE-подобная архитектура"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        # Усредняем по времени для эмуляции VideoMAE
        x = x.mean(dim=2)
        return self.model(x)

# 5. CNN + LSTM
class VideoClassifierCNNLSTM(nn.Module):
    """CNN + LSTM для видео классификации"""
    def __init__(self, num_classes=2, cnn_backbone='resnet18', lstm_hidden=512, lstm_layers=2):
        super().__init__()
        
        # CNN backbone
        if cnn_backbone == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            cnn_out = 512
        elif cnn_backbone == 'resnet34':
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            cnn_out = 512
        elif cnn_backbone == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            cnn_out = 2048
        else:
            raise ValueError(f"Unknown backbone: {cnn_backbone}")
        
        # Убираем классификатор
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LSTM
        self.lstm = nn.LSTM(cnn_out, lstm_hidden, lstm_layers, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # Обрабатываем каждый кадр через CNN
        frame_features = []
        for t in range(T):
            frame = x[:, :, t, :, :]
            feat = self.cnn(frame)
            feat = self.cnn_pool(feat)
            feat = feat.view(B, -1)
            frame_features.append(feat)
        
        # Стек по времени [B, T, features]
        sequence = torch.stack(frame_features, dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(sequence)
        
        # Используем последний выход
        features = lstm_out[:, -1, :]
        
        return self.classifier(features)

# 6. TimeSformer (эмуляция)
class VideoClassifierTimeSformer(nn.Module):
    """TimeSformer-подобная архитектура"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        
        # Временной embedding
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, 16, 768) * 0.02)
        
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
        # Временная проекция
        self.temporal_proj = nn.Linear(768, 768)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # Обрабатываем каждый кадр
        frame_outputs = []
        for t in range(T):
            frame = x[:, :, t, :, :]
            # Получаем признаки без головы
            out = self.model.forward_features(frame)
            # [B, num_patches+1, 768]
            frame_outputs.append(out)
        
        # [B, T, num_patches+1, 768]
        time_stack = torch.stack(frame_outputs, dim=1)
        
        # Усредняем по патчам
        time_avg = time_stack.mean(dim=2)  # [B, T, 768]
        
        # Добавляем временной embedding
        time_avg = time_avg + self.temporal_pos_embed[:, :T, :]
        
        # Усредняем по времени
        features = time_avg.mean(dim=1)  # [B, 768]
        
        return self.model.head(features)


def get_model(model_name, num_classes=2, pretrained=True):
    """Фабрика моделей"""
    models_dict = {
        '3d_resnet': lambda: VideoClassifier3DResNet(num_classes, pretrained),
        'r2plus1d': lambda: VideoClassifierR2Plus1D(num_classes, pretrained),
        'slowfast': lambda: VideoClassifierSlowFast(num_classes, pretrained),
        'videomae': lambda: VideoClassifierVideoMAE(num_classes, pretrained),
        'cnn_lstm': lambda: VideoClassifierCNNLSTM(num_classes, lstm_hidden=512, lstm_layers=2),
        'timesformer': lambda: VideoClassifierTimeSformer(num_classes, pretrained)
    }
    
    if model_name not in models_dict:
        available = list(models_dict.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    return models_dict[model_name]()