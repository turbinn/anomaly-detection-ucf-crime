import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.video as video_models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(CNNLSTM, self).__init__()
        
        self.cnn = models.resnet18(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in list(self.cnn.parameters())[-4:]:
            param.requires_grad = True
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        batch_size, frames, C, H, W = x.size()
        
        cnn_input = x.view(batch_size * frames, C, H, W)
        cnn_features = self.cnn(cnn_input)
        cnn_features = cnn_features.view(batch_size * frames, -1)
        
        lstm_input = cnn_features.view(batch_size, frames, -1)
        lstm_output, _ = self.lstm(lstm_input)
        last_output = lstm_output[:, -1, :]
        output = self.fc(last_output)
        
        return output


def get_model(model_name, num_classes=2, pretrained=True):
    """
    Поддерживаемые модели из torchvision:
    - r3d_18: 3D ResNet (33M параметров)
    - mc3_18: Mixed 3D/2D Convolution (33M параметров)
    - r2plus1d_18: (2+1)D CNN (33M параметров)
    - mvit: Multiscale Vision Transformer (5M параметров)
    - s3d: Separable 3D CNN (33M параметров)
    - cnn_lstm: гибридная CNN+LSTM (28M параметров)
    """
    
    if model_name == "r3d_18":
        model = video_models.r3d_18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "mc3_18":
        model = video_models.mc3_18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "r2plus1d_18":
        model = video_models.r2plus1d_18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "mvit":
        model = video_models.mvit_v2_s(pretrained=pretrained)
        # у MViT другой классификатор - head вместо fc
        if hasattr(model, 'head'):
            model.head = nn.Linear(model.head.in_features, num_classes)
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "s3d":
        model = video_models.s3d(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "cnn_lstm":
        model = CNNLSTM(num_classes=num_classes, pretrained=pretrained)
        
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: r3d_18, mc3_18, r2plus1d_18, mvit, s3d, cnn_lstm")
    
    return model


def get_model_params(model_name):
    """Возвращает количество параметров для каждой модели"""
    params = {
        "r3d_18": 33_000_000,
        "mc3_18": 33_000_000,
        "r2plus1d_18": 33_000_000,
        "mvit": 5_000_000,
        "s3d": 33_000_000,
        "cnn_lstm": 28_000_000
    }
    return params.get(model_name, 0)


def get_recommended_batch_size(model_name):
    """Рекомендованный batch_size для каждой модели"""
    batch_sizes = {
        "r3d_18": 8,
        "mc3_18": 8,
        "r2plus1d_18": 8,
        "mvit": 12,
        "s3d": 8,
        "cnn_lstm": 8
    }
    return batch_sizes.get(model_name, 8)