import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = np.load('D:\\FakeDetector\\data\\image_data\\X_train.npy')
X_test = np.load('D:\\FakeDetector\\data\\image_data\\X_test.npy')
y_train = np.load('D:\\FakeDetector\\data\\image_data\\y_train.npy')
y_test = np.load('D:\\FakeDetector\\data\\image_data\\y_test.npy')

# Build the model using EfficientNet as base
def build_model(input_shape=(128, 128, 3)):
    # Create base model with EfficientNetB0
    input_tensor = Input(shape=input_shape)
    base_model = EfficientNetB0(include_top=False, 
                               input_tensor=input_tensor,
                               weights='imagenet')
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers on top
    x = GlobalMaxPooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=input_tensor, outputs=output)
    return model

# Create model
model = build_model()

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(
    filepath='D:\\FakeDetector\\models\\image_model\\best_model',
    save_best_only=True,
    monitor='val_accuracy'
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the final model
model.save('D:\\FakeDetector\\models\\image_model\\final_model')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.savefig('D:\\FakeDetector\\results\\image_model_training_history.png')
