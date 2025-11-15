# Bhai ka pehla ML project - Handwritten Digit Recognition
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

print("Bhai ne ML shuru kar diya! 🚀")

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
print("Training shuru...")
model.fit(x_train, y_train, epochs=3)

# Test
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nAccuracy: {test_acc*100:.2f}% 🎯")

# Save model
model.save('digit_model.h5')
print("Model saved! Ab job pakki! 💼")
