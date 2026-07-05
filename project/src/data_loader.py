import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
from pathlib import Path
import random

class UCFCrimeVideoDataset(Dataset):
    """Dataset для загрузки видеофрагментов из UCF-Crime"""
    
    def __init__(self, data_path, split='train', num_frames=16, frame_size=224, 
                 transform=None, samples_per_video=3):
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.samples_per_video = samples_per_video
        
        # Сбор всех видеофайлов
        self.video_files = []
        self.labels = []
        
        # Проверяем структуру UCF-Crime
        anomaly_path = self.data_path / 'Anomaly_Videos'
        normal_path = self.data_path / 'Normal_Videos'
        
        if anomaly_path.exists():
            for video in glob.glob(str(anomaly_path / '**/*.mp4'), recursive=True):
                self.video_files.append(video)
                self.labels.append(1)
        
        if normal_path.exists():
            for video in glob.glob(str(normal_path / '**/*.mp4'), recursive=True):
                self.video_files.append(video)
                self.labels.append(0)
        
        # Если не найдено, ищем все mp4
        if not self.video_files:
            for video in glob.glob(str(self.data_path / '**/*.mp4'), recursive=True):
                if 'Anomaly' in video or 'Anomalous' in video:
                    self.video_files.append(video)
                    self.labels.append(1)
                elif 'Normal' in video:
                    self.video_files.append(video)
                    self.labels.append(0)
                else:
                    # Пробуем определить по имени файла
                    name = os.path.basename(video).lower()
                    if any(word in name for word in ['abandon', 'arrest', 'assault', 'burglary', 
                                                      'explosion', 'fighting', 'robbery', 'shooting',
                                                      'shoplifting', 'stealing', 'vandalism']):
                        self.video_files.append(video)
                        self.labels.append(1)
                    else:
                        self.video_files.append(video)
                        self.labels.append(0)
        
        print(f"Найдено видео: {len(self.video_files)}")
        print(f"  Аномалий: {sum(self.labels)}")
        print(f"  Нормальных: {len(self.labels) - sum(self.labels)}")
        
        # Создаём список (видео, метка, номер клипа)
        self.samples = []
        for video_path, label in zip(self.video_files, self.labels):
            # Получаем количество кадров в видео
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if total_frames < self.num_frames:
                # Если видео слишком короткое, используем все кадры
                self.samples.append((video_path, label, 0, total_frames))
            else:
                # Генерируем несколько клипов из одного видео
                for i in range(self.samples_per_video):
                    start_frame = random.randint(0, max(0, total_frames - self.num_frames))
                    self.samples.append((video_path, label, start_frame, total_frames))
        
        print(f"Всего клипов: {len(self.samples)}")
        
        # Разделение на train/val/test
        if split == 'train':
            self.samples, _ = train_test_split(
                self.samples, test_size=0.3, random_state=42, 
                stratify=[l for _, l, _, _ in self.samples]
            )
            # Дополнительное разделение: часть train для валидации
            self.samples, self.val_samples = train_test_split(
                self.samples, test_size=0.2, random_state=42,
                stratify=[l for _, l, _, _ in self.samples]
            )
            if split == 'val':
                self.samples = self.val_samples
        elif split == 'test':
            _, self.samples = train_test_split(
                self.samples, test_size=0.3, random_state=42,
                stratify=[l for _, l, _, _ in self.samples]
            )
        
        print(f"{split}: {len(self.samples)} клипов")
        
    def __len__(self):
        return len(self.samples)
    
    def load_video_clip(self, video_path, start_frame=0):
        """Загружает клип из видео"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Изменяем размер
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frames.append(frame)
        
        cap.release()
        
        # Если кадров меньше чем нужно, дублируем последний
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
        
        return frames
    
    def __getitem__(self, idx):
        video_path, label, start_frame, total_frames = self.samples[idx]
        
        # Загружаем кадры
        frames = self.load_video_clip(video_path, start_frame)
        
        # Преобразуем в numpy array [T, H, W, C]
        frames = np.array(frames, dtype=np.float32) / 255.0
        
        # Преобразуем в [C, T, H, W]
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        # Применяем аугментации для train
        if self.transform is not None:
            frames = self.transform(frames)
        
        return torch.FloatTensor(frames), torch.FloatTensor([label])


class VideoTransform:
    """Преобразования для видео"""
    def __init__(self, is_train=True, augment=True):
        self.is_train = is_train
        self.augment = augment
        
    def __call__(self, video):
        # video: [C, T, H, W]
        if self.is_train and self.augment:
            # Горизонтальный флип
            if np.random.random() > 0.5:
                video = np.flip(video, axis=-1).copy()
            
            # Случайное изменение яркости
            if np.random.random() > 0.7:
                brightness = np.random.uniform(0.8, 1.2)
                video = np.clip(video * brightness, 0, 1)
            
            # Случайный шум
            if np.random.random() > 0.8:
                noise = np.random.normal(0, 0.02, video.shape)
                video = np.clip(video + noise, 0, 1)
        
        # Нормализация (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])[:, None, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None, None]
        video = (video - mean) / std
        
        return video


def create_dataloaders(data_path, batch_size=4, num_frames=16, frame_size=224, 
                       num_workers=2, samples_per_video=3):
    """Создает DataLoader для train/val/test"""
    transform_train = VideoTransform(is_train=True, augment=True)
    transform_val = VideoTransform(is_train=False, augment=False)
    
    train_dataset = UCFCrimeVideoDataset(
        data_path, split='train', 
        num_frames=num_frames, frame_size=frame_size,
        transform=transform_train, samples_per_video=samples_per_video
    )
    
    val_dataset = UCFCrimeVideoDataset(
        data_path, split='val',
        num_frames=num_frames, frame_size=frame_size,
        transform=transform_val, samples_per_video=1
    )
    
    test_dataset = UCFCrimeVideoDataset(
        data_path, split='test',
        num_frames=num_frames, frame_size=frame_size,
        transform=transform_val, samples_per_video=1
    )

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=use_pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=use_pin
    )
    
    return train_loader, val_loader, test_loader