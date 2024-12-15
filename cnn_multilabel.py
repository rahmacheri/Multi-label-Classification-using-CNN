

import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from scipy.io import savemat, loadmat

def main(scenario_name, snr_value, top_k=20):
    train_X = loadmat(f'/path/{scenario_name}-{snr_value}-train/data.mat')['datat']
    train_Y = loadmat(f'/path/{scenario_name}-{snr_value}-train-labels/labels.mat')['Labels']
    valid_X = loadmat(f'/path/{scenario_name}-{snr_value}-valid/data.mat')['datat']
    valid_Y = loadmat(f'/path/{scenario_name}-{snr_value}-valid-labels/labels.mat')['Labels']
    test_X = loadmat(f'/path/{scenario_name}-{snr_value}-test/data.mat')['datat']
    test_Y = loadmat(f'/path/{scenario_name}-{snr_value}-test-labels/labels.mat')['Labels']

    # Reshape the data
    train_X = train_X.reshape((6000, 64, 4, 1))
    valid_X = valid_X.reshape((2000, 64, 4, 1))
    test_X = test_X.reshape((2000, 64, 4, 1))

    # Define model
    model = Sequential([
        Conv2D(filters=16, kernel_size=(4, 4), strides=1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Flatten(),
        Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(units=64, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(train_X, train_Y, epochs=50, validation_data=(valid_X, valid_Y), callbacks=[early_stopping])

    start_time = time.time()

    predicted_labels = model.predict(test_X)

    # Sort predictions for each sample in descending order
    sorted_indices = np.argsort(-predicted_labels, axis=1)

    # Initialize array to store top k labels for each sample
    selected_labels = np.zeros_like(test_Y)

    for i in range(len(test_Y)):
        top_indices = sorted_indices[i, :top_k]
        # Sort the top indices to maintain the original order
        top_indices = np.sort(top_indices)
        selected_labels[i, top_indices] = 1

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(elapsed_time)

    print(selected_labels[:10])

    directory = "/kaggle/working/"

    # Ensure that the directory exists, create it if necessary
    os.makedirs(directory, exist_ok=True)

    # Define the filename for the MATLAB file
    filename = f"{scenario_name}_{snr_value}_predicted_labelcnn.mat"

    # Concatenate the directory path and filename
    filepath = os.path.join(directory, filename)

    # Save the predicted labels to the MATLAB file
    try:
        savemat(filepath, {"y_predcnn": selected_labels})
    except Exception as e:
        print("Error saving file:", e)

    # Calculate the F1 score
    f1 = f1_score(test_Y, selected_labels, average='samples')  # Assuming 'y_test' contains the true labels
    print("F1 Score:", f1)


if __name__ == "__main__":
    scenario_name = "20"
    snr_value = "snr0"
    main(scenario_name, snr_value, top_k=20)
