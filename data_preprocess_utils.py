import os
import numpy as np
import pretty_midi
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=RuntimeWarning)

class MidiLoader:
    @staticmethod
    def load_single(midi_path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                midi = pretty_midi.PrettyMIDI(midi_path)
                return midi if midi.instruments else None
        except Exception:
            return None

    @classmethod
    def load_from_directory(cls, composer_dir):
        files = [f for f in os.listdir(composer_dir) if f.endswith(('.mid', '.midi'))]
        return [cls.load_single(os.path.join(composer_dir, f)) for f in tqdm(files)]

class FeatureExtractor:
    @staticmethod
    def extract(midi, max_length=500):
        features = np.zeros((max_length, 4))
        if not midi or not midi.instruments:
            return features, 0
            
        notes = []
        for instr in midi.instruments:
            notes.extend([(n.pitch, n.start, n.end, n.velocity) for n in instr.notes])
        
        if not notes:
            return features, 0
            
        notes.sort(key=lambda x: x[1])
        notes = np.array(notes)
        notes[:, 0] /= 127.0  # pitch
        notes[:, 1] /= max(notes[-1, 1], 1)  # timing
        notes[:, 2] = (notes[:, 2] - notes[:, 1]) / 10.0  # duration
        notes[:, 3] /= 127.0  # velocity
        
        seq_len = min(len(notes), max_length)
        features[:seq_len] = notes[:seq_len]
        return features, seq_len

class MidiAugmenter:
    def __init__(self):
        self.augmentations = [self._transpose, self._time_stretch, self._adjust_velocity]
    
    def augment(self, midi):
        for aug in self.augmentations:
            if np.random.rand() > 0.5:
                midi = aug(midi)
        return midi
    
    def _transpose(self, midi, max_steps=2):
        steps = np.random.randint(-max_steps, max_steps + 1)
        for instr in midi.instruments:
            for note in instr.notes:
                note.pitch = max(0, min(127, note.pitch + steps))
        return midi
    
    def _time_stretch(self, midi, factor_range=(0.9, 1.1)):
        factor = np.random.uniform(*factor_range)
        for instr in midi.instruments:
            for note in instr.notes:
                note.start *= factor
                note.end *= factor
        return midi
    
    def _adjust_velocity(self, midi, max_change=20):
        change = np.random.randint(-max_change, max_change)
        for instr in midi.instruments:
            for note in instr.notes:
                note.velocity = max(1, min(127, note.velocity + change))
        return midi

class ComposerDataset(Dataset):
    def __init__(self, data_dir, composers, max_length=500, augment=False):
        self.composers = composers
        self.max_length = max_length
        self.augment = augment
        self.augmenter = MidiAugmenter()
        self.samples = self._load_samples(data_dir)
        self._compute_class_weights()
    
    def _load_samples(self, data_dir):
        samples = []
        for label, composer in enumerate(self.composers):
            midis = MidiLoader.load_from_directory(os.path.join(data_dir, composer))
            samples.extend([(midi, label) for midi in midis if midi])
        return samples
    
    def _compute_class_weights(self):
        labels = [label for (_, label) in self.samples]
        self.class_weights = 1. / torch.bincount(torch.tensor(labels)).float()
        self.class_weights /= self.class_weights.sum()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        midi, label = self.samples[idx]
        if self.augment:
            midi = self.augmenter.augment(midi)
        features, length = FeatureExtractor.extract(midi, self.max_length)
        return {
            'features': torch.FloatTensor(features),
            'length': torch.tensor(length),
            'label': torch.tensor(label)
        }

def create_stratified_splits(dataset, test_size=0.2, val_size=0.1, random_state=42):
    """Create stratified train/val/test splits"""
    indices = np.arange(len(dataset))
    labels = [label for (_, label) in dataset.samples]
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: train vs val
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size/(1-test_size),
        stratify=train_val_labels,
        random_state=random_state
    )
    
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx)
    )

def plot_feature_distribution(features, composer):
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(['Pitch', 'Timing', 'Duration', 'Velocity']):
        plt.subplot(2, 2, i+1)
        plt.hist(features[:, i], bins=50, alpha=0.7)
        plt.title(f"{composer} - {name}")
    plt.tight_layout()
    plt.show()

def print_split_stats(dataset, split_set, name):
    """Print class distribution statistics for a dataset split"""
    labels = [dataset.samples[i][1] for i in split_set.indices]
    counts = torch.bincount(torch.tensor(labels))
    
    print(f"\n{name.upper()} SET CLASS DISTRIBUTION:")
    for i, composer in enumerate(dataset.composers):
        print(f"- {composer.capitalize()}: {counts[i]} samples")