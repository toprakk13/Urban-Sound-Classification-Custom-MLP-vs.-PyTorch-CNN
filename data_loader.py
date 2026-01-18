import os
import numpy as np
import librosa
from tqdm import tqdm
import glob

def get_file_paths(dataset_path):
    
    train_files = []
    test_files = []
    
    #1-8 train
    for i in range(1, 9):
        folder = os.path.join(dataset_path, f"fold{i}")
        train_files.extend(glob.glob(os.path.join(folder, "*.wav")))
        
    # 9-10 test
    for i in range(9, 11):
        folder = os.path.join(dataset_path, f"fold{i}")
        test_files.extend(glob.glob(os.path.join(folder, "*.wav")))
        
    return train_files, test_files

def extract_mel_spectrogram(file_path, n_mels=128, fixed_width=128):
   
    try:
        y, sr = librosa.load(file_path, sr=None)
        # Mel Spec
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_spec_db.shape[1] < fixed_width:
            #padding 
            pad_width = fixed_width - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            #trimming
            mel_spec_db = mel_spec_db[:, :fixed_width]
            
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_mfcc(file_path, n_mfcc=40, fixed_width=128):
    
    
    y, sr = librosa.load(file_path, sr=None)
        # mfcc 40,128
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
    if mfcc.shape[1] < fixed_width:
        pad_width = fixed_width - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :fixed_width]
            
    return mfcc

def load_and_process_data(file_paths,feature_type="mel", is_flatten=True,save_path=None):
    
    if save_path:
        x_path = f"{save_path}_X.npy"
        y_path = f"{save_path}_y.npy"
        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"Loading data from cached .npy files: {x_path}")
            return np.load(x_path), np.load(y_path)
    X = []
    y = []
    
    for path in tqdm(file_paths, desc="Loading Audio"):
        if feature_type == 'mel':
            spec = extract_mel_spectrogram(path)
        elif feature_type == 'mfcc':
            spec = extract_mfcc(path)
        if spec is not None:
            # Normalize 
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
            
            if is_flatten:
                X.append(spec.flatten())
            else:
                X.append(spec[np.newaxis, ...]) # (1, 128, 128)
            
           
            filename = os.path.basename(path)
            class_id = int(filename.split('-')[1])
            y.append(class_id)
            
    X,y=np.array(X), np.array(y)
    if save_path:
        np.save(f"{save_path}_X.npy", X)
        np.save(f"{save_path}_y.npy", y)
        print(f"Data saved to {save_path}_X.npy and {save_path}_y.npy")
        
    return X, y

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]