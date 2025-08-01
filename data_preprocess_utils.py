import os
import pretty_midi
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
import matplotlib.pyplot as plt
import seaborn as sns

def count_midi_files(data_dir="midiclassics"):
    """
    Count MIDI files per composer and return a dictionary.
    Args:
        data_dir: Path to the folder containing composer subfolders.
    Returns:
        Dict of {composer: file_count}.
    """
    composers = ["bach", "beethoven", "chopin", "mozart"]
    counts = {}
    
    for composer in composers:
        composer_dir = os.path.join(data_dir, composer)
        midi_files = [
            f for f in os.listdir(composer_dir) 
            if f.endswith(('.mid', '.midi'))
        ]
        counts[composer] = len(midi_files)
    
    return counts

def plot_composer_counts(counts):
    """
    Plot a bar graph of MIDI file counts per composer.
    Args:
        counts: Dict from count_midi_files().
        save_path: If provided, saves the plot to this path.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(counts.keys()),
        y=list(counts.values()),
        hue=list(counts.keys()), 
        palette="viridis"
    )
    plt.title("Number of MIDI Files per Composer", fontsize=16)
    plt.xlabel("Composer", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    
    # Add value labels
    for i, count in enumerate(counts.values()):
        plt.text(i, count + 5, str(count), ha='center', fontsize=12)
    plt.show()
