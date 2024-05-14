import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner import RandomSearch, Objective
from joblib import dump

# Load and preprocess the data
data = pd.read_csv('vct_data.csv')
encoder_shield = OneHotEncoder(sparse=False)
shield_encoded = encoder_shield.fit_transform(data[['shield']])
encoder_gun = OneHotEncoder(sparse=False)
gun_encoded = encoder_gun.fit_transform(data[['gun']])
scaler = StandardScaler()
features = ['u_credits', 't1_credits', 't2_credits', 't3_credits', 't4_credits']
X_scaled = scaler.fit_transform(data[features])
X = X_scaled
y_shield = shield_encoded
y_gun = gun_encoded
X_train, X_test, y_shield_train, y_shield_test, y_gun_train, y_gun_test = train_test_split(X, y_shield, y_gun, test_size=0.2, random_state=42)

# Define the model-building function
def build_model(hp):
    inputs = Input(shape=(X_train.shape[1],))
    x = Dense(hp.Int('units1', min_value=32, max_value=512, step=32), activation='relu')(inputs)
    x = Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Int('units2', min_value=16, max_value=256, step=16), activation='relu')(x)
    shield_output = Dense(3, activation='softmax', name='shield_output')(x)
    gun_output = Dense(14, activation='softmax', name='gun_output')(x)
    model = Model(inputs=inputs, outputs=[shield_output, gun_output])
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss={'shield_output': 'categorical_crossentropy', 'gun_output': 'categorical_crossentropy'},
                  metrics={'shield_output': 'accuracy', 'gun_output': 'accuracy'})
    return model

# Set up hyperparameter tuner with explicit objective
tuner = RandomSearch(
    build_model,
    objective=Objective('val_shield_output_accuracy', direction='max'),  # Explicit objective with direction
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='keras_tuner_vct'
)

# Early Stopping and Model Checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_shield_output_accuracy', mode='max')

# Start hyperparameter search
tuner.search(X_train, {'shield_output': y_shield_train, 'gun_output': y_gun_train}, epochs=50, 
             validation_data=(X_test, {'shield_output': y_shield_test, 'gun_output': y_gun_test}),
             callbacks=[early_stopping, model_checkpoint])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model and preprocessors
best_model.save('gun_shield_model.keras')
dump(scaler, 'scaler.joblib')
dump(encoder_shield, 'encoder_shield.joblib')
dump(encoder_gun, 'encoder_gun.joblib')