import os
import numpy as np
import pretty_midi
import copy
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


# ---------------------- MIDI LOADING ----------------------
class MidiLoader:
    @staticmethod
    def load_single(midi_path):
        """Load a single MIDI file, return PrettyMIDI object or None if invalid."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                midi = pretty_midi.PrettyMIDI(midi_path)
                return midi if midi.instruments else None
        except Exception:
            return None

    @classmethod
    def load_from_directory(cls, composer_dir):
        """Load all MIDI files from a directory."""
        files = [f for f in os.listdir(composer_dir) if f.endswith(('.mid', '.midi'))]
        return [cls.load_single(os.path.join(composer_dir, f)) for f in tqdm(files, desc=f"Loading {composer_dir}")]


# ---------------------- FEATURE EXTRACTION ----------------------
class FeatureExtractor:
    @staticmethod
    def extract(midi, max_length=500):
        """
        Extract features: [Pitch, StartTime, Duration, Velocity]
        Normalized for LSTM/CNN usage.
        """
        features = np.zeros((max_length, 4))
        if not midi or not midi.instruments:
            return features, 0

        notes = []
        for instr in midi.instruments:
            notes.extend([(n.pitch, n.start, n.end, n.velocity) for n in instr.notes])

        if not notes:
            return features, 0

        notes.sort(key=lambda x: x[1])  # sort by start time
        notes = np.array(notes, dtype=float)

        # Normalize features
        notes[:, 0] /= 127.0  # Pitch [0,1]
        notes[:, 1] /= max(notes[-1, 1], 1)  # Timing relative to last note
        durations = notes[:, 2] - notes[:, 1]
        durations /= max(durations.max(), 1)  # Normalize duration
        notes[:, 2] = durations
        notes[:, 3] /= 127.0  # Velocity [0,1]

        seq_len = min(len(notes), max_length)
        features[:seq_len] = notes[:seq_len]
        return features, seq_len


# ---------------------- DATA AUGMENTATION ----------------------
class MidiAugmenter:
    def __init__(self):
        self.augmentations = [self._transpose, self._time_stretch, self._adjust_velocity]

    def augment(self, midi):
        """Apply random augmentations to a copy of the MIDI."""
        midi = copy.deepcopy(midi)
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


# ---------------------- COMPOSER DATASET ----------------------
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
            composer_dir = os.path.join(data_dir, composer)
            midis = MidiLoader.load_from_directory(composer_dir)
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


# ---------------------- DATA SPLITTING & VISUALIZATION ----------------------
class SplitData:
    def __init__(self, dataset):
        self.dataset = dataset

    def create_stratified_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        """Create stratified train/val/test splits."""
        indices = np.arange(len(self.dataset))
        labels = [label for (_, label) in self.dataset.samples]

        # Train+Val vs Test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=random_state
        )

        # Train vs Val
        train_val_labels = [labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size/(1-test_size),
            stratify=train_val_labels,
            random_state=random_state
        )

        return (
            Subset(self.dataset, train_idx),
            Subset(self.dataset, val_idx),
            Subset(self.dataset, test_idx)
        )

    def plot_feature_distribution(self, composer_idx, max_samples=10):
        """Plot feature distribution for a specific composer."""
        composer_samples = [
            i for i, (_, label) in enumerate(self.dataset.samples)
            if label == composer_idx
        ][:max_samples]

        features = torch.cat([
            self.dataset[i]['features'][:self.dataset[i]['length']]
            for i in composer_samples
        ]).numpy()

        plt.figure(figsize=(12, 8))
        for i, name in enumerate(['Pitch', 'Timing', 'Duration', 'Velocity']):
            plt.subplot(2, 2, i + 1)
            sns.histplot(features[:, i], bins=50, kde=True)
            plt.title(f"{self.dataset.composers[composer_idx]} - {name}")
        plt.tight_layout()
        plt.show()

    def print_split_stats(self, split_set, name):
        """Print class distribution statistics for a dataset split."""
        labels = [self.dataset.samples[i][1] for i in split_set.indices]
        counts = torch.bincount(torch.tensor(labels))

        print(f"\n{name.upper()} SET CLASS DISTRIBUTION:")
        for i, composer in enumerate(self.dataset.composers):
            print(f"- {composer.capitalize()}: {counts[i]} samples")


# ---------------------- RESAMPLED DATASET ----------------------
class Resampled(Dataset):
    def __init__(self, dataset, method='interpolate', random_state=42):
        """
        Custom resampler for sequence data.
        """
        self.dataset = dataset
        self.method = method.lower()
        self.random_state = random_state
        self.samples = self._balanced_resampling()

    def _balanced_resampling(self):
        """Perform balanced resampling while preserving sequence structure."""
        class_samples = defaultdict(list)
        for sample in self.dataset:
            class_samples[sample['label'].item()].append(sample)

        max_count = max(len(samples) for samples in class_samples.values())
        resampled_samples = []

        for label, samples in class_samples.items():
            current_count = len(samples)
            if current_count < max_count:
                num_to_generate = max_count - current_count
                synthetic_samples = self._interpolate_sequences(samples, num_to_generate)
                resampled_samples.extend(samples + synthetic_samples)
            else:
                resampled_samples.extend(samples)

        return resampled_samples

    def _interpolate_sequences(self, samples, num_to_generate):
        """Generate synthetic sequences via interpolation."""
        synthetic_samples = []
        max_length = max(s['length'].item() for s in samples)

        for _ in range(num_to_generate):
            idx1, idx2 = np.random.choice(len(samples), 2, replace=False)
            sample1 = samples[idx1]
            sample2 = samples[idx2]

            seq1 = sample1['features'][:sample1['length']].numpy()
            seq2 = sample2['features'][:sample2['length']].numpy()

            alpha = np.random.uniform(0.2, 0.8)
            min_len = min(len(seq1), len(seq2))
            mixed_seq = alpha * seq1[:min_len] + (1 - alpha) * seq2[:min_len]

            if len(mixed_seq) < max_length:
                pad_width = ((0, max_length - len(mixed_seq)), (0, 0))
                mixed_seq = np.pad(mixed_seq, pad_width, mode='constant')

            synthetic_samples.append({
                'features': torch.FloatTensor(mixed_seq),
                'length': torch.tensor(min_len),
                'label': torch.tensor(sample1['label'].item())
            })

        return synthetic_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def plot_distribution(self, class_names=None):
        """Visualize class distribution."""
        labels = [s['label'].item() for s in self.samples]
        counts = np.bincount(labels)

        plt.figure(figsize=(10, 5))
        if class_names and len(class_names) == len(counts):
            plt.bar(class_names, counts)
        else:
            plt.bar(range(len(counts)), counts)
        plt.title(f"Class Distribution ({self.method} balanced)")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
