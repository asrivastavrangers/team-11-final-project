import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import kagglehub
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Suppress pkg_resources warning
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

# Constants
SELECTED_COMPOSERS = ['Bach', 'Beethoven', 'Chopin', 'Mozart']
SEED = 42
DATASET_NAME = "blanderbuss/midi-classic-music"

def download_dataset():
    """
    Download the dataset from Kaggle using kagglehub
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    print("Downloading dataset from Kaggle...")
    try:
        # Download the dataset
        path = kagglehub.dataset_download(DATASET_NAME)
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def load_dataset(dataset_path):
    """
    Load the dataset from the downloaded directory
    
    Args:
        dataset_path (str): Path to the downloaded dataset directory
        
    Returns:
        list: List of tuples containing (file_path, composer_name)
    """
    data = []
    if not dataset_path:
        return data
        
    print("Loading MIDI files...")
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                # Extract composer name from path
                composer = os.path.basename(root)
                data.append((os.path.join(root, file), composer))
    
    print(f"Loaded {len(data)} MIDI files")
    return data

def filter_composers(data, composers=SELECTED_COMPOSERS):
    """
    Filter the dataset to only include selected composers.
    
    Args:
        data (list): List of tuples from load_dataset
        composers (list): List of composer names to include
        
    Returns:
        list: Filtered list of tuples
    """
    return [(path, comp) for path, comp in data if comp in composers]

def visualize_data_distribution(data):
    """
    Visualize the distribution of composers in the dataset.
    
    Args:
        data (list): List of tuples from load_dataset or filter_composers
    """
    composers = [comp for _, comp in data]
    counts = pd.Series(composers).value_counts().reset_index()
    counts.columns = ['Composer', 'Count']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=counts, 
                x='Composer', 
                y='Count',
                hue='Composer',  
                palette='viridis',
                legend=False)  
    
    plt.title('Distribution of Compositions by Composer')
    plt.xlabel('Composer')
    plt.ylabel('Number of Compositions')
    plt.xticks(rotation=45)
    plt.show()

def extract_features(midi_path, max_length=100, verbose=False):
    """
    Extract features from a MIDI file with robust error handling.
    """
    try:
        # Modified line - removed strict parameter
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Rest of the function remains the same
        if not midi_data.instruments:
            if verbose:
                print(f"Warning: No instruments found in {midi_path}")
            return None
            
        features = []
        notes_processed = 0
        
        for instrument in midi_data.instruments:
            # Skip drum instruments (channel 9)
            if instrument.is_drum:
                if verbose:
                    print(f"Skipping drum instrument in {midi_path}")
                continue
                
            for note in instrument.notes:
                features.append([
                    note.pitch,            # MIDI pitch value (0-127)
                    note.velocity,         # Note velocity (0-127)
                    note.end - note.start  # Duration in seconds
                ])
                notes_processed += 1
        
        # Handle empty files after processing
        if not features:
            if verbose:
                print(f"Warning: No valid notes found in {midi_path}")
            return None
        
        # Pad or truncate to fixed length
        if len(features) > max_length:
            features = features[:max_length]
        else:
            padding = [[0, 0, 0]] * (max_length - len(features))
            features.extend(padding)
            
        if verbose:
            print(f"Processed {midi_path} - extracted {notes_processed} notes")
            
        return np.array(features)
    
    except Exception as e:
        if verbose:
            print(f"Error processing {midi_path}: {str(e)}")
        return None

def split_dataset(data, test_size=0.2, val_size=0.1):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data (list): List of tuples (features, label)
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set (from remaining after test)
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # First split into train+val and test
    train_val, test = train_test_split(
        data, test_size=test_size, random_state=SEED, stratify=[d[1] for d in data]
    )
    
    # Then split train+val into train and val
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=SEED, 
        stratify=[d[1] for d in train_val]
    )
    
    return train, val, test

def create_tensor_datasets(train_data, val_data, test_data):
    """
    Convert datasets into PyTorch tensors.
    
    Args:
        train_data (list): Training data
        val_data (list): Validation data
        test_data (list): Test data
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, label_encoder)
    """
    # Extract features and labels
    X_train = np.array([d[0] for d in train_data])
    y_train = [d[1] for d in train_data]
    
    X_val = np.array([d[0] for d in val_data])
    y_val = [d[1] for d in val_data]
    
    X_test = np.array([d[0] for d in test_data])
    y_test = [d[1] for d in test_data]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val_encoded)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test_encoded)
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, val_dataset, test_dataset, label_encoder

class MidiDataset(Dataset):
    """Custom PyTorch Dataset for MIDI files"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        features, label = self.data[idx]
        return torch.FloatTensor(features), torch.LongTensor([label])

def load_processed_data(data_dir="preprocess"):
    """Load previously processed datasets"""
    train_dataset = torch.load(os.path.join(data_dir, 'train_dataset.pt'))
    val_dataset = torch.load(os.path.join(data_dir, 'val_dataset.pt'))
    test_dataset = torch.load(os.path.join(data_dir, 'test_dataset.pt'))
    
    # Recreate label encoder
    class_labels = np.load(os.path.join(data_dir, 'label_encoder_classes.npy'))
    label_encoder = LabelEncoder()
    label_encoder.classes_ = class_labels
    
    return train_dataset, val_dataset, test_dataset, label_encoder