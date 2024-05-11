import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model Definition
def build_model(num_classes=3):  # Adjust num_classes for heavy, light, and none
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Data Loading and Preprocessing
def load_data(data_dir, target_size=(150, 150), batch_size=20):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 20% data for validation
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Use 'training' subset for training data
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Use 'validation' subset for validation data
    )
    return train_generator, validation_generator

# Training the Model
def train_model(data_dir, epochs=15):
    train_gen, val_gen = load_data(data_dir)
    model = build_model(num_classes=train_gen.num_classes)  # Set the number of classes based on the training generator
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,  # Calculate steps per epoch based on dataset size
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size  # Calculate validation steps based on dataset size
    )
    model.save('vct_shields_model.keras')  # Save the trained model
    return history

# Main Execution Logic
if __name__ == "__main__":
    data_dir = 'vctshield_training/'  # Specific dataset path for the shield images
    history = train_model(data_dir)
