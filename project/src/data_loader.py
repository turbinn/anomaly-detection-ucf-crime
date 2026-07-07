import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, model_name="r3d_18", frames_per_video=16, is_train=True):
        self.video_paths = video_paths
        self.labels = labels
        self.model_name = model_name
        self.frames_per_video = frames_per_video
        self.is_train = is_train
        
        self.resize_size = self._get_resize_size()
        self.transform = self._get_transforms()
        
    def _get_resize_size(self):
        sizes = {
            "r3d_18": 112,
            "mc3_18": 112,
            "r2plus1d_18": 112,
            "mvit": 224,
            "s3d": 112,
            "cnn_lstm": 112
        }
        return sizes.get(self.model_name, 112)
    
    def _get_transforms(self):
        if self.is_train:
            return transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                   std=[0.22803, 0.22145, 0.216989])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                   std=[0.22803, 0.22145, 0.216989])
            ])
    
    def _extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames.append(frame)
            else:
                frames.append(torch.zeros(3, self.resize_size, self.resize_size))
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        return torch.stack(frames)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._extract_frames(video_path)
        
        if frames is None:
            frames = torch.zeros(self.frames_per_video, 3, self.resize_size, self.resize_size)
        
        # для 3D CNN и MViT: [frames, channels, H, W] -> [channels, frames, H, W]
        if self.model_name in ["r3d_18", "mc3_18", "r2plus1d_18", "s3d", "mvit"]:
            frames = frames.permute(1, 0, 2, 3)
        
        return frames, label


def load_ucf_crime_data(data_root, model_name="r3d_18", frames_per_video=16):
    anomaly_dir = os.path.join(data_root, "Anomaly_Videos")
    normal_dir = os.path.join(data_root, "Normal_Videos")
    
    if not os.path.exists(anomaly_dir):
        for item in os.listdir(data_root):
            if 'anomaly' in item.lower():
                anomaly_dir = os.path.join(data_root, item)
                break
    
    if not os.path.exists(normal_dir):
        for item in os.listdir(data_root):
            if 'normal' in item.lower():
                normal_dir = os.path.join(data_root, item)
                break
    
    print(f"Anomaly path: {anomaly_dir}")
    print(f"Normal path: {normal_dir}")
    
    video_paths = []
    labels = []
    
    if os.path.exists(anomaly_dir):
        items = os.listdir(anomaly_dir)
        has_subdirs = any(os.path.isdir(os.path.join(anomaly_dir, item)) for item in items)
        
        if has_subdirs:
            for category in items:
                category_path = os.path.join(anomaly_dir, category)
                if os.path.isdir(category_path):
                    for video_file in os.listdir(category_path):
                        if video_file.endswith(('.mp4', '.avi', '.mkv')):
                            video_paths.append(os.path.join(category_path, video_file))
                            labels.append(1)
        else:
            for video_file in items:
                if video_file.endswith(('.mp4', '.avi', '.mkv')):
                    video_paths.append(os.path.join(anomaly_dir, video_file))
                    labels.append(1)
    
    if os.path.exists(normal_dir):
        for video_file in os.listdir(normal_dir):
            if video_file.endswith(('.mp4', '.avi', '.mkv')):
                video_paths.append(os.path.join(normal_dir, video_file))
                labels.append(0)
    
    print(f"Found {len(video_paths)} videos ({sum(labels)} anomaly, {len(labels)-sum(labels)} normal)")
    
    return video_paths, labels


def create_dataloaders(data_root, model_name="r3d_18", batch_size=8, frames_per_video=16, 
                       train_split=0.8, num_workers=2):
    video_paths, labels = load_ucf_crime_data(data_root, model_name, frames_per_video)
    
    indices = np.random.permutation(len(video_paths))
    split_idx = int(len(video_paths) * train_split)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_paths = [video_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_paths = [video_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    train_dataset = VideoDataset(
        train_paths, train_labels, model_name, frames_per_video, is_train=True
    )
    test_dataset = VideoDataset(
        test_paths, test_labels, model_name, frames_per_video, is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, len(train_dataset), len(test_dataset)