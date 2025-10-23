from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import librosa

def extract_frame_level_features(dataset, frame_size, hop_size, n_jobs=-1):
    """
    Iterates through all clips in the dataset and extracts frame-level features.
    Returns a list of dictionaries, one per frame across all clips.
    """

    # Parallel processing with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_clip)( # delay function calls per clip so parallel can run them concurrently
            dataset, cid, frame_size, hop_size # function parameters for process_clip
        )
        for cid in tqdm(dataset.clip_ids, desc="Processing clips") # loop over all clip_ids in our dataset from loader
    )

    """
    Gives something like:
    results = [
        [frame_dict_1, frame_dict_2, ...],   # clip 1
        [frame_dict_a, frame_dict_b, ...],   # clip 2
        ...
        ]
    So we flatten:
    """
    all_frames = []
    for clip_frames in results:
        for frame in clip_frames:
            all_frames.append(frame)
    return all_frames

def process_clip(dataset, cid, frame_size, hop_size):
    """Helper to allow clean parallelisation"""
    clip = dataset.clip(cid)
    audio, sr = clip.audio

    return compute_frame_features(
        audio=audio,
        sr=sr,
        frame_size=frame_size,
        hop_size=hop_size,
        cid=cid,
        fold=clip.fold,
        label=clip.target,
    )

def compute_frame_features(audio, sr, frame_size, hop_size, cid, fold, label):
    """
    Splits one audio clip into overlapping frames and computes metrics per frame.
    Returns a list of dicts (one per frame).
    """
    frames = [] # each entry corresponds to one frame

    # iterate over sliding window. We make sure our final frame is complete and does not extend past the final audio value
    for start_idx in range(0, len(audio) - frame_size + 1, hop_size):
        end_idx = start_idx + frame_size
        frame = audio[start_idx:end_idx]

        # compute metrics for this frame
        feats = compute_frame_metrics(frame, sr)

        # attach metadata, e.g. so we can identify which clip each frame came from
        feats.update({
            "cid": cid,
            "fold": fold,
            "label": label,
            "frame_start": start_idx,
            "frame_end": end_idx
        })

        frames.append(feats)

    return frames

def compute_frame_metrics(frame, sr):
    """
    Computes a set of acoustic features for a single frame.
    Each metric returns a scalar (or small vector like MFCCs).
    """
    feats = {}
    n_fft = len(frame) # incase len(frame) < default 2048

    # Basic features (same as in base model but now computed per frame instead of per clip)
    feats["rms"] = np.sqrt(np.mean(frame ** 2))
    feats["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y=frame))
    feats["spec_cent"] = np.mean(librosa.feature.spectral_centroid(y=frame, sr=sr, n_fft=n_fft))
    feats["spec_bw"] = np.mean(librosa.feature.spectral_bandwidth(y=frame, sr=sr, n_fft=n_fft))

    # Mel-Frequency Cepstral Coefficients. Each mfcc[i] is a vector as 1 value per internal subframe so we take the mean
    mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13, n_fft=n_fft)
    feats.update({f"mfcc{i+1}": np.mean(mfcc[i]) for i in range(13)})

    return feats