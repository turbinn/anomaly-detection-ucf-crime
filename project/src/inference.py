import torch
import cv2
import numpy as np
from pathlib import Path
import time
from src.data_loader import VideoTransform

class VideoAnomalyDetector:
    """Класс для инференса на видео"""
    
    def __init__(self, model, model_name, device=None, num_frames=16, frame_size=224):
        self.model = model
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = VideoTransform(is_train=False, augment=False)
        self.threshold = 0.5
        
    def load_video_clip(self, video_path, start_frame=0):
        """Загружает клип из видео"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
        
        return frames
    
    def predict_clip(self, frames):
        """Предсказание для одного клипа"""
        # Преобразуем в [C, T, H, W]
        frames = np.array(frames, dtype=np.float32) / 255.0
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        # Применяем нормализацию
        frames = self.transform(frames)
        
        # Добавляем batch dimension
        frames = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(frames)
            score = torch.sigmoid(output.squeeze()).item()
        
        return score
    
    def predict_video(self, video_path, stride=8):
        """Предсказание для всего видео с перекрывающимися клипами"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        scores = []
        positions = []
        
        for start in range(0, total_frames - self.num_frames + 1, stride):
            frames = self.load_video_clip(video_path, start)
            score = self.predict_clip(frames)
            scores.append(score)
            positions.append(start)
        
        # Усредняем по всем клипам
        avg_score = np.mean(scores) if scores else 0.5
        prediction = 1 if avg_score > self.threshold else 0
        
        return {
            'video_path': video_path,
            'prediction': prediction,
            'score': avg_score,
            'confidence': abs(avg_score - 0.5) * 2,  # Нормализованная уверенность
            'clip_scores': scores,
            'clip_positions': positions,
            'total_frames': total_frames,
            'num_clips': len(scores)
        }
    
    def predict_video_with_visualization(self, video_path, output_path=None, stride=16):
        """Предсказание с визуализацией на видео"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path is None:
            output_path = f"runs/{self.model_name}/demo_output.mp4"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = {
            'predictions': [],
            'scores': [],
            'timestamps': [],
            'anomaly_detected': False
        }
        
        for start in range(0, total_frames - self.num_frames + 1, stride):
            # Загружаем клип
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(self.num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if len(frames) < self.num_frames:
                break
            
            score = self.predict_clip(frames)
            
            # Определяем аномалию
            is_anomaly = score > self.threshold
            results['predictions'].append(is_anomaly)
            results['scores'].append(score)
            results['timestamps'].append(start / fps)
            
            if is_anomaly:
                results['anomaly_detected'] = True
            
            # Визуализация на каждом кадре клипа
            for i, frame in enumerate(frames):
                frame_copy = frame.copy()
                
                # Цвет рамки в зависимости от предсказания
                color = (0, 0, 255) if is_anomaly else (0, 255, 0)  # Red для аномалии, Green для нормы
                
                # Добавляем рамку и текст
                text = f"ANOMALY {score:.3f}" if is_anomaly else f"NORMAL {score:.3f}"
                cv2.putText(frame_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
                
                # Прогресс бар уверенности
                bar_width = 200
                bar_height = 15
                bar_x = 10
                bar_y = 60
                
                cv2.rectangle(frame_copy, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height), 
                             (100, 100, 100), 1)
                cv2.rectangle(frame_copy, (bar_x, bar_y), 
                             (bar_x + int(bar_width * score), bar_y + bar_height), 
                             color, -1)
                
                cv2.putText(frame_copy, f"Confidence: {score:.2f}", 
                           (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                out.write(frame_copy)
        
        cap.release()
        out.release()
        
        return results, output_path