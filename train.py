
import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import numpy as np

def main(args):
    # Load the MobileNetV2 model without the top classification layer
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global Average Pooling
    x = Dense(198, activation='linear')(x)  # Dense layer with 198 linear units

    # Create the new model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss='mean_squared_error', metrics=['mae'])

    # Load the data
    data = np.load(args.data_path)/255
    label = np.load(args.label_path)

    # Split the data into training and validation sets (80% train, 20% validation)
    x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.2, random_state=42)

    # Create model directory if it doesn't exist
    model_dir = "./model/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)
    model_checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_weights.h5'), monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
              callbacks=[reduce_lr, model_checkpoint])
    
    model.load_weights(os.path.join(model_dir, 'best_weights.h5'))
    model.save(os.path.join(model_dir, 'best_model.h5'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 model for fish keypoint detection.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the numpy file containing the training images.")
    parser.add_argument('--label_path', type=str, required=True, help="Path to the numpy file containing the training labels.")
    
    args = parser.parse_args()
    main(args)
