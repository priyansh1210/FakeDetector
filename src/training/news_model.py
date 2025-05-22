import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load preprocessed data
X_train = np.load('D:\\FakeDetector\\data\\news_data\\tfidf_train.npy')
X_test = np.load('D:\\FakeDetector\\data\\news_data\\tfidf_test.npy')
y_train = np.load('D:\\FakeDetector\\data\\news_data\\y_train.npy')
y_test = np.load('D:\\FakeDetector\\data\\news_data\\y_test.npy')

# Build the model
input_dim = X_train.shape[1]  # Number of features

model = Sequential([
    Dense(256, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: real vs fake
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint(
    filepath='D:\\FakeDetector\\models\\news_model\\best_model',
    save_best_only=True,
    monitor='val_accuracy'
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the final model
model.save('D:\\FakeDetector\\models\\news_model\\final_model')
