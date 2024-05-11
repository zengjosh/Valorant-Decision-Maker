import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model Definition
def build_model(num_classes=18):  # Updated num_classes to 16 to match the number of weapon categories
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
        Dense(num_classes, activation='softmax')  # Output layer adjusted to 16 classes
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
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator

# Training the Model
def train_model(data_dir, epochs=15):
    train_gen, val_gen = load_data(data_dir)
    model = build_model(num_classes=train_gen.num_classes)  # Ensures that the model's last layer matches the number of classes
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,  # Dynamic adjustment based on actual dataset size
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size  # Dynamic adjustment based on validation set size
    )
    model.save('vct_guns_model.keras')  # Saving the model under a new name to reflect the dataset
    return history

# Main Execution Logic
if __name__ == "__main__":
    data_dir = 'vctguns_training/'  # Adjust this to the actual path
    history = train_model(data_dir)
