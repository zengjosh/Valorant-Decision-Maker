import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
# print("Unique values in 'gun' column:", data['gun'].unique())
# Define and compile the neural network
inputs = Input(shape=(X_train.shape[1],))
x = Dense(100, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

# Define output layers with correct number of neurons
shield_output = Dense(3, activation='softmax', name='shield_output')(x)  # 3 unique classes for 'shield'
gun_output = Dense(14, activation='softmax', name='gun_output')(x)  # 'n' unique classes for 'gun'

model = Model(inputs=inputs, outputs=[shield_output, gun_output])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'shield_output': 'categorical_crossentropy', 'gun_output': 'categorical_crossentropy'},
              metrics={'shield_output': 'accuracy', 'gun_output': 'accuracy'})

# Train the model
model.fit(X_train, [y_shield_train, y_gun_train], validation_data=(X_test, [y_shield_test, y_gun_test]), epochs=50, batch_size=32)

# Save the model
model.save('gun_shield_model.keras')
