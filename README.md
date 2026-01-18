Urban Sound Classification: Custom MLP vs. PyTorch CNN

This project is a comprehensive deep learning study developed to classify urban sounds using the Urban Sound Dataset. The primary goal is to analyze different audio feature extraction methods (Mel Spectrogram vs. MFCC) and to compare a from-scratch neural network with a modern CNN architecture implemented in PyTorch.

Project Features

Key technical aspects that make this project unique:
	•	MLP from Scratch with NumPy:
A Multi-Layer Perceptron engine is implemented using only mathematical formulations (Gradient Descent, Backpropagation, ReLU, Softmax) without relying on sklearn or torch.
	•	Dynamic CNN with PyTorch:
A flexible Convolutional Neural Network (CNN) is built to process audio spectrograms using a dynamic layer structure.
	•	Audio Feature Analysis:
Raw audio signals are processed to extract both Mel Spectrogram and MFCC representations, and their impact on classification performance is evaluated.

📂 File Structure
	•	main.py: Main execution script. Loads data, trains MLP and CNN models, compares results, and saves the best-performing models.
	•	data_loader.py: Reads audio files (WAV), performs Mel Spectrogram and MFCC transformations using librosa, and prepares the dataset (Folds 1–8 for training, Folds 9–10 for testing).
	•	mlp_model.py: From-scratch implementation. NumPy-based class containing the mathematical backbone of the neural network (forward and backward passes).
	•	cnn_model.py: CNN architecture built using PyTorch nn.Module, including Conv2D and pooling layers.

Methodology

1. Data Preprocessing
	•	Audio files are loaded using librosa.
	•	Mel Spectrogram: Visualizes the temporal evolution of frequency components (processed as 2D images for CNNs).
	•	MFCC: Extracts features that are closer to human auditory perception.
	•	Data is normalized using methods such as Standard Scaling or Min-Max Scaling.

2. Models

A. Custom MLP (NumPy)
	•	Activation Functions: ReLU (hidden layers), Softmax (output layer).
	•	Optimization: Stochastic Gradient Descent (SGD).
	•	Architecture: Flexible layer sizes (e.g., [Input, 512, 256, 10]).

B. CNN (PyTorch)
	•	Architecture: Two or more convolutional blocks followed by fully connected layers.
	•	Training: CrossEntropyLoss with the Adam optimizer.
	•	Feature Support: Dynamic structure supporting both Mel Spectrogram (128×128) and MFCC (40×128) inputs.

Results

Key findings from the experiments:
	•	CNNs outperform MLPs by effectively capturing spatial patterns in spectrograms.
	•	Mel Spectrograms generally yield better performance than MFCCs in deep learning models due to their richer representation of audio texture.
