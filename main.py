import os
import sys
import pandas as pd
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_loader import get_file_paths, load_and_process_data, one_hot_encode
from mlp_model import train_mlp, MLP
from cnn_model import train_cnn, UrbanSoundDataset, CNN
def set_seed(seed=42):
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")

set_seed(42)

def main():
   
    project_root = os.path.dirname(current_dir)
    DATASET_PATH = os.path.join(project_root, "archive")
    
    
    cache_dir = os.path.join(current_dir, "processed_data")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    train_files, test_files = get_file_paths(DATASET_PATH)

    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    # 1. MLP data loading(Flattened)
    print("Loading Data for MLP (Flattened)... This may take time.")
    X_train_mlp, y_train_raw = load_and_process_data(train_files, is_flatten=True,save_path=os.path.join(cache_dir, "mlp_train_mel"))
    X_test_mlp, y_test_raw = load_and_process_data(test_files, is_flatten=True,save_path=os.path.join(cache_dir, "mlp_test_mel"))

    #one-hot encode
    Y_train_mlp = one_hot_encode(y_train_raw)
    Y_test_mlp = one_hot_encode(y_test_raw)

    print(f"MLP Train Shape: {X_train_mlp.shape}")

    # 2. CNN data loading (2D Image Shape)
    X_train_cnn, y_train_cnn_raw = load_and_process_data(train_files, is_flatten=False,save_path=os.path.join(cache_dir, "cnn_train_mel"))
    X_test_cnn, y_test_cnn_raw = load_and_process_data(test_files, is_flatten=False,save_path=os.path.join(cache_dir, "cnn_test_mel"))
    y_train_cnn = y_train_cnn_raw
    y_test_cnn = y_test_cnn_raw
    print(f"CNN Train Shape: {X_train_cnn.shape}")


   

    # Parameteres Learning rate: 0.005-0.02, Batch size: 16-128)
    LR = 0.01
    BATCH_SIZE = 64
    EPOCHS = 20
    INPUT_DIM = 128 * 128
    OUTPUT_DIM = 10

    # 1: Single Layer (No Hidden) 
    print("\n--- MLP Experiment 1: Single Layer (No Hidden) ---")
    layer_sizes_1 = [INPUT_DIM, OUTPUT_DIM] 
    mlp1, loss1, acc1 = train_mlp(X_train_mlp, Y_train_mlp, X_test_mlp, Y_test_mlp, 
                                  layer_sizes_1, EPOCHS, LR, BATCH_SIZE)

    # 2: Two Hidden Layers
    print("\n--- MLP Experiment 2: Two Hidden Layers ---")
    layer_sizes_2 = [INPUT_DIM, 128, 64, OUTPUT_DIM] 
    mlp2, loss2, acc2 = train_mlp(X_train_mlp, Y_train_mlp, X_test_mlp, Y_test_mlp, 
                                  layer_sizes_2, EPOCHS, LR, BATCH_SIZE)

    # visualiziation    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss1, label="Single Layer")
    plt.plot(loss2, label="2 Hidden Layers")
    plt.title("MLP Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc1, label="Single Layer")
    plt.plot(acc2, label="2 Hidden Layers")
    plt.title("MLP Test Accuracy")
    plt.legend()
    plt.show()

    # weight visualization
    print("Visualizing Weights for MLP Model 2 (First Layer):")
    mlp2.visualize_weights()


    

    # Parameters 
    learning_rates = [0.005, 0.01, 0.02]
    batch_sizes = [16, 32, 64, 128]

    results_list = []

    INPUT_DIM = 128 * 128
    OUTPUT_DIM = 10
    EPOCHS = 20 

    print("--- Starting Hyperparameter Experiments for MLP ---")

    # combinations
    for lr in learning_rates:
        for batch in batch_sizes:
            print(f"\nTesting LR: {lr}, Batch Size: {batch} ...")
            
            # Model: 2 Hidden Layers
            layer_sizes = [INPUT_DIM, 128, 64, OUTPUT_DIM]
            
            
            model, loss_hist, acc_hist = train_mlp(
                X_train_mlp, Y_train_mlp, X_test_mlp, Y_test_mlp, 
                layer_sizes, EPOCHS, lr, batch
            )
            
            
            results_list.append({
                "Learning Rate": lr,
                "Batch Size": batch,
                "Final Loss": loss_hist[-1],
                "Final Test Accuracy": acc_hist[-1]
            })


    results_df = pd.DataFrame(results_list)


    print("\n--- MLP Hyperparameter Experiment Results ---")
    
    print(results_df)

    # best parameter
    best_result = results_df.loc[results_df['Final Test Accuracy'].idxmax()]
    print(f"\nBest Parameters: LR={best_result['Learning Rate']}, Batch={best_result['Batch Size']}")


    

    # PyTorch DataLoader 
    train_dataset = UrbanSoundDataset(X_train_cnn, y_train_cnn)
    test_dataset = UrbanSoundDataset(X_test_cnn, y_test_cnn)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Parameters
    CNN_EPOCHS = 20
    CNN_LR = 0.01
    DECAY_RATE = 0.95 # Reduction at each epoch

    # 3: 1 Conv + 1 Fully Connected 
    print("\n--- CNN Experiment 1: 1 Conv + 1 FC ---")
    cnn1 = CNN(num_conv_layers=1, num_fc_layers=1, num_classes=10)
    cnn1, cnn_loss1, cnn_acc1 = train_cnn(cnn1, train_loader, test_loader, CNN_EPOCHS, CNN_LR, DECAY_RATE)

    # 4: 2 Conv + 2 Fully Connected 
    print("\n--- CNN Experiment 2: 2 Conv + 2 FC ---")
    cnn2 = CNN(num_conv_layers=2, num_fc_layers=2, num_classes=10)
    cnn2, cnn_loss2, cnn_acc2 = train_cnn(cnn2, train_loader, test_loader, CNN_EPOCHS, CNN_LR, DECAY_RATE)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cnn_loss1, label="1 Conv + 1 FC")
    plt.plot(cnn_loss2, label="2 Conv + 2 FC")
    plt.title("CNN Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cnn_acc1, label="1 Conv + 1 FC")
    plt.plot(cnn_acc2, label="2 Conv + 2 FC")
    plt.title("CNN Test Accuracy")
    plt.legend()
    plt.show()


    

    #  BONUS: CNN MFCC 

    print("\n--- BONUS Experiment: CNN with MFCC ---")

    # 1. load data (N, 1, 40, 128)

    if 'X_train_cnn_mfcc' not in locals():
        print("Loading MFCC Data for CNN...")
        X_train_cnn_mfcc, y_train_mfcc_raw = load_and_process_data(train_files, feature_type='mfcc', is_flatten=False,save_path=os.path.join(cache_dir, "cnn_train_mffc"))
        X_test_cnn_mfcc, y_test_mfcc_raw = load_and_process_data(test_files, feature_type='mfcc', is_flatten=False,save_path=os.path.join(cache_dir, "cnn_test_mffc"))

    print(f"MFCC Input Shape: {X_train_cnn_mfcc.shape}") 
    # (7079, 1, 40, 128)


    train_dataset_mfcc = UrbanSoundDataset(X_train_cnn_mfcc, y_train_mfcc_raw)
    test_dataset_mfcc = UrbanSoundDataset(X_test_cnn_mfcc, y_test_mfcc_raw)

    train_loader_mfcc = DataLoader(train_dataset_mfcc, batch_size=64, shuffle=True)
    test_loader_mfcc = DataLoader(test_dataset_mfcc, batch_size=64, shuffle=False)


    #  input_shape  (1, 40, 128)
    input_shape_mfcc = X_train_cnn_mfcc.shape[1:] # (1, 40, 128)

    cnn_mfcc = CNN(num_conv_layers=2, num_fc_layers=2, num_classes=10, input_shape=input_shape_mfcc)

    print(f"Training CNN on MFCC data with shape {input_shape_mfcc}...")
    cnn_mfcc, loss_mfcc, acc_mfcc = train_cnn(cnn_mfcc, train_loader_mfcc, test_loader_mfcc, epochs=15, lr=0.01, decay_rate=0.95)


    plt.figure(figsize=(10, 5))
    plt.plot(acc_mfcc, label="CNN (MFCC) - 2 Conv")
    plt.plot(cnn_acc2, label="CNN (Mel Spec) - 2 Conv") 
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Feature Comparison: MFCC vs Mel Spectrogram (CNN)")
    plt.legend()
    plt.show()


    new_row = {
        "Model": "CNN (MFCC Features)",
        "Final Test Accuracy": acc_mfcc[-1],
        "Final Training Loss": loss_mfcc[-1]
    }

    new_df = pd.DataFrame([new_row])


    print("\nUpdated Results Table:")
    # display(new_df)
    print(new_df)




    
    models_dir = os.path.join(project_root, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("Saving the best model...")
    torch.save(cnn2.state_dict(), os.path.join(models_dir, 'best_cnn_model.pth'))
    print("Saving the best model...")
    torch.save(cnn_mfcc.state_dict(), os.path.join(models_dir, 'best_cnn_model_mfcc.pth'))




    results = {
        "Model": ["MLP (1 Layer)", "MLP (3 Layer)", "CNN (1 Conv)", "CNN (2 Conv)"],
        "Final Test Accuracy": [acc1[-1], acc2[-1], cnn_acc1[-1], cnn_acc2[-1]],
        "Final Training Loss": [loss1[-1], loss2[-1], cnn_loss1[-1], cnn_loss2[-1]]
    }

    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()