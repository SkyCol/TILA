from torchvision import transforms
import random
from collections import defaultdict
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import torch

class FewShotImageFolder(Dataset):
    def __init__(self, root, num_shots, transform=None, test_transform=None, seed=42):
        super().__init__()
        self.root = root
        self.num_shots = num_shots
        self.transform = transform
        self.test_transform = test_transform
        
        full_dataset = ImageFolder(root)
        self.class_to_idx = full_dataset.class_to_idx

        label_to_samples = defaultdict(list)
        for path, label in full_dataset.samples:
            label_to_samples[label].append(path)
        
        random.seed(seed)
        
        self.train_samples = []
        self.train_targets = []
        self.test_samples = []
        self.test_targets = []

        for label, samples in label_to_samples.items():
            if len(samples) <= num_shots:
                sampled = samples
                remain = []
            else:
                sampled = random.sample(samples, num_shots)
                remain = [s for s in samples if s not in sampled]

            self.train_samples.extend(sampled)
            self.train_targets.extend([label] * len(sampled))
            self.test_samples.extend(remain)
            self.test_targets.extend([label] * len(remain))

        combined = list(zip(self.train_samples, self.train_targets))
        random.shuffle(combined)
        self.samples, self.targets = zip(*combined)
        self.samples = list(self.samples)
        self.targets = list(self.targets)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.targets[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label

    def get_test_set(self):
        return SimpleImageDataset(self.test_samples, self.test_targets,self.test_transform)


class SimpleImageDataset(Dataset):
    def __init__(self, samples, targets, transform):
        self.samples = samples
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.targets[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label)
        return image, label