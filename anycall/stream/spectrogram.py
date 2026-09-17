import numpy as np
import librosa
from scipy.ndimage import binary_closing, label, find_objects
from typing import List, Tuple

class SpectrogramEventDetector:
    """Detects acoustic events (ROIs) from raw audio using morphological operations on Mel spectrograms."""
    
    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 64,
        fmax: int = 12000,
        hop_length: int = 512,
        energy_threshold: float = 0.85,
        min_duration: float = 0.05,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmax = fmax
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.min_duration = min_duration
        
    def find_events(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Analyze audio and return a list of (start_time, end_time) events in seconds.
        Uses Harmonic-Percussive Source Separation to isolate bird streaks.
        """
        # 1. Compute STFT
        D = librosa.stft(audio)
        
        # 2. HPSS - isolate the harmonic streaks (margin > 1.0 reduces percussive bleed)
        D_harmonic, _ = librosa.decompose.hpss(D, margin=1.2)
        
        # 3. Mel Spectrogram of the harmonic component
        S = librosa.feature.melspectrogram(
            S=np.abs(D_harmonic)**2, sr=self.sample_rate, n_mels=self.n_mels, fmax=self.fmax, hop_length=self.hop_length
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalize to 0-1
        s_min = S_db.min()
        s_max = S_db.max()
        if s_max == s_min:
            return []
            
        S_norm = (S_db - s_min) / (s_max - s_min)
        
        # Thresholding
        binary = S_norm > self.energy_threshold
        
        # Morphological closing to connect fragmented streaks 
        structure = np.ones((3, 5))
        closed = binary_closing(binary, structure=structure)
        
        # Find distinct blobs
        labeled, num_features = label(closed)
        objects = find_objects(labeled)
        
        events = []
        for obj in objects:
            freq_slice, time_slice = obj
            
            start_time = time_slice.start * self.hop_length / self.sample_rate
            end_time = time_slice.stop * self.hop_length / self.sample_rate
            duration = end_time - start_time
            
            if duration >= self.min_duration:
                events.append((start_time, end_time))
                
        # Merge overlapping or touching events in time domain
        if not events:
            return []
            
        events.sort(key=lambda x: x[0])
        merged = [events[0]]
        
        for current in events[1:]:
            previous = merged[-1]
            if current[0] <= previous[1] + 0.2:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
                
        return merged
