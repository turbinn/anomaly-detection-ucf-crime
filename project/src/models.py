# src/models.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r3d_18, r2plus1d_18
import timm


# 3D ResNet-18

class VideoClassifier3DResNet(nn.Module):
    """3D ResNet для классификации видео"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = r3d_18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        output = self.model(x)
        if output.size(1) == 2:
            return output[:, 1:2]
        return output



# R(2+1)D

class VideoClassifierR2Plus1D(nn.Module):
    """R(2+1)D для классификации видео"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = r2plus1d_18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)



#SlowFast (упрощенная версия)

class VideoClassifierSlowFast(nn.Module):
    """SlowFast архитектура"""
    def __init__(self, num_classes=2, pretrained=True, alpha=4):
        super().__init__()
        self.alpha = alpha
        
        self.slow_path = r3d_18(pretrained=pretrained)
        slow_in_features = self.slow_path.fc.in_features
        self.slow_path.fc = nn.Identity()
        
        self.fast_path = r3d_18(pretrained=pretrained)
        fast_in_features = self.fast_path.fc.in_features
        self.fast_path.fc = nn.Identity()
        
        self.fusion = nn.Sequential(
            nn.Linear(slow_in_features + fast_in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        slow_indices = list(range(0, T, self.alpha))
        if len(slow_indices) < 2:
            slow_indices = [0, min(T-1, 1)]
        slow_frames = x[:, :, slow_indices, :, :]
        
        fast_frames = nn.functional.interpolate(
            x.permute(0, 2, 1, 3, 4),
            scale_factor=0.5,
            mode='trilinear'
        ).permute(0, 2, 1, 3, 4)
        
        slow_feat = self.slow_path(slow_frames)
        fast_feat = self.fast_path(fast_frames)
        
        combined = torch.cat([slow_feat, fast_feat], dim=1)
        return self.fusion(combined)


# VideoMAE (эмуляция через ViT)

class VideoClassifierVideoMAE(nn.Module):
    """VideoMAE-подобная архитектура"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.mean(dim=2)
        return self.model(x)



# CNN + LSTM

class VideoClassifierCNNLSTM(nn.Module):
    """CNN + LSTM для видео классификации"""
    def __init__(self, num_classes=2, cnn_backbone='resnet18', lstm_hidden=512, lstm_layers=2):
        super().__init__()
        
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
        
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.lstm = nn.LSTM(cnn_out, lstm_hidden, lstm_layers, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        frame_features = []
        for t in range(T):
            frame = x[:, :, t, :, :]
            feat = self.cnn(frame)
            feat = self.cnn_pool(feat)
            feat = feat.view(B, -1)
            frame_features.append(feat)
        
        sequence = torch.stack(frame_features, dim=1)
        lstm_out, _ = self.lstm(sequence)
        features = lstm_out[:, -1, :]
        
        return self.classifier(features)



# TimeSformer (эмуляция)

class VideoClassifierTimeSformer(nn.Module):
    """TimeSformer-подобная архитектура"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, 16, 768) * 0.02)
        
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
        self.temporal_proj = nn.Linear(768, 768)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        frame_outputs = []
        for t in range(T):
            frame = x[:, :, t, :, :]
            out = self.model.forward_features(frame)
            frame_outputs.append(out)
        
        time_stack = torch.stack(frame_outputs, dim=1)
        time_avg = time_stack.mean(dim=2)
        time_avg = time_avg + self.temporal_pos_embed[:, :T, :]
        features = time_avg.mean(dim=1)
        
        return self.model.head(features)



# X3D-M

class VideoClassifierX3D(nn.Module):
    """X3D-M для классификации видео"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=pretrained)
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)



# Фабрика моделей

def get_model(model_name, num_classes=2, pretrained=True):
    """Фабрика моделей"""
    models_dict = {
        '3d_resnet': lambda: VideoClassifier3DResNet(num_classes, pretrained),
        'r2plus1d': lambda: VideoClassifierR2Plus1D(num_classes, pretrained),
        'slowfast': lambda: VideoClassifierSlowFast(num_classes, pretrained),
        'videomae': lambda: VideoClassifierVideoMAE(num_classes, pretrained),
        'cnn_lstm': lambda: VideoClassifierCNNLSTM(num_classes, lstm_hidden=512, lstm_layers=2),
        'timesformer': lambda: VideoClassifierTimeSformer(num_classes, pretrained),
        'x3d_m': lambda: VideoClassifierX3D(num_classes, pretrained)
    }
    
    if model_name not in models_dict:
        available = list(models_dict.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    return models_dict[model_name]()