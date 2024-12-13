import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, LSTM, Flatten, Dense, Dropout, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping


# load audio
def load_audio(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)  # Load audio at specified sampling rate
    return y

# Generate Mel-Spectrograms
def extract_mel_spectrogram(audio, sr=16000, n_mels=128, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    # print(f"Extracted Mel-Spectrogram shape: {log_mel_spectrogram.shape}")  # Debugging line
    return log_mel_spectrogram

# Split Spectrograms into Time-Slices
def split_spectrogram(spectrogram, slice_length=32):
    slices = []
    if spectrogram.shape[1] < slice_length:
        print("Warning: Spectrogram width is less than slice length. No slices will be generated.")  # Debugging line
        return np.array(slices)  # Return empty array if no slices can be generated
    for i in range(0, spectrogram.shape[1] - slice_length + 1, slice_length // 2):  # 50% overlap
        slice_ = spectrogram[:, i:i + slice_length]
        slices.append(slice_)
    print(f"Number of slices generated: {len(slices)}")  # Debugging line
    return np.array(slices)

# Normalize and Reshape Data
def preprocess_slices(slices):
    if slices.size == 0:  # Check if slices is empty
        print("Warning: Received empty slices array.")  # Debugging line
        return slices  # Return empty array without processing
    slices = slices / np.max(slices)  # Normalize to [0, 1]
    slices = np.expand_dims(slices, axis=-1)  # Add channel dimension
    return slices

# Convert class labels (real, fake) into numerical values (0, 1):
def assign_labels(files, label):
    return [(file, label) for file in files]

# Prepare Dataset for Training
# For each audio file:

# Extract its mel-spectrogram.
# Split the spectrogram into slices.
# Assign the corresponding label to each slice.
# Combine the features and labels into arrays:

def prepare_dataset(dataset_path):
    features, labels = [], []
    for label_name in ['real', 'fake']:
        label = 0 if label_name == 'real' else 1
        folder = os.path.join(dataset_path, label_name)
        print(f"Checking folder: {folder}")  # Debugging line
        if not os.path.exists(folder):  # Check if the folder exists
            print(f"Folder does not exist: {folder}")  # Debugging line
            continue  # Skip to the next label if the folder doesn't exist
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            try:
                audio = load_audio(file_path)

                spectrogram = extract_mel_spectrogram(audio)
                print(f"Spectrogram shape: {spectrogram.shape}")
                slices = split_spectrogram(spectrogram)
                print(f"Number of slices generated: {len(slices)}") 
                slices = preprocess_slices(slices)                
                features.extend(slices)
                labels.extend([label] * len(slices))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return np.array(features), np.array(labels)

# Get the current working directory
cwd = os.getcwd()

# Define the folder or file name you want to concatenate
train_folder_name = "dataset\\train\\"
test_folder_name = "dataset\\test\\"

# Concatenate the CWD with the folder name
train_full_path = os.path.join(cwd, train_folder_name)
test_full_path = os.path.join(cwd, test_folder_name)

# Prepare training and testing datasets
X_train, y_train = prepare_dataset(train_full_path)
X_test, y_test = prepare_dataset(test_full_path)


# Split the training data into training and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check shapes
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# X_train and X_test: Arrays of shape (num_samples, 128, 128, 1) where:
# num_samples: Total slices.
# 128, 128, 1: Height, width, and channels of spectrogram slices.
# y_train and y_test: Arrays of shape (num_samples,) containing binary labels (0 for real, 1 for fake).

with open('train_data.pkl', 'wb') as f:
    pickle.dump((X_train, y_train), f)

with open('test_data.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)

with open('train_data.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

# Balancing the Dataset
# If your dataset is imbalanced (e.g., more real samples than fake), apply techniques like:

# Oversampling: Duplicate slices from the minority class.
# Undersampling: Reduce samples from the majority class.
# Class Weights: Adjust loss function weights to penalize minority misclassification.

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Define model architecture
def build_hybrid_model(input_shape):
    # input_shape = (128, 128, 1)  # Spectrogram size

    # Input layer
    inputs = Input(shape=input_shape)

    # CNN layers
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Flatten and prepare for RNN input
    x = TimeDistributed(Flatten())(x)

    # RNN layers
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    # Fully connected layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build the model
def train_function():
    input_shape = (128, 32, 1)  # Spectrogram dimensions
    model = build_hybrid_model(input_shape)
    model.summary()
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks help in improving training and preventing overfitting. Use:
    # ModelCheckpoint: Save the best model during training.
    # EarlyStopping: Stop training when validation performance stops improving.

    checkpoint = ModelCheckpoint(
        filepath='emma_model.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        verbose=1
    )

    # Train the model on the processed dataset:
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=25,  # Increase as needed
        batch_size=32, 
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    # Load the best model
    best_model = tf.keras.models.load_model('emma_model.h5')

    # Evaluate on test data
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save the Model
    model.save('final_hybrid_model.h5')

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    