import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

# Load IMDB dataset
max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each review

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(f"{len(x_train)} training sequences")
print(f"{len(x_test)} validation sequences")

# Pad sequences to same length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Transformer Components
def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

# Transformer Model for Text Classification
class TransformerClassifier(keras.Model):
    def __init__(self, max_features, embed_dim, num_heads, dense_dim, maxlen, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(max_features, embed_dim)
        self.pos_encoding = positional_encoding(length=maxlen, depth=embed_dim)
        self.encoders = [TransformerEncoder(embed_dim, dense_dim, num_heads)
                        for _ in range(num_layers)]
        self.dropout1 = layers.Dropout(0.7)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout2 = layers.Dropout(0.5)
        self.classifier = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        # Create padding mask
        mask = tf.math.not_equal(inputs, 0)

        x = self.embedding(inputs)
        x = x + self.pos_encoding[tf.newaxis, :maxlen, :]
        x = self.dropout1(x, training=training)

        for encoder in self.encoders:
            x = encoder(x, mask=mask)

        x = self.global_pool(x)
        x = self.dropout2(x, training=training)
        return self.classifier(x)

# Enhanced parameters
embed_dim = 128
num_heads = 4
dense_dim = 128
num_layers = 2

model = TransformerClassifier(
    max_features=max_features,
    embed_dim=embed_dim,
    num_heads=num_heads,
    dense_dim=dense_dim,
    maxlen=maxlen,
    num_layers=num_layers)

# Add learning rate scheduling and early stopping
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping])

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], color='purple',label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], color='purple', label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

plot_history(history)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Make predictions
sample_text = "This movie was absolutely fantastic! The acting was superb and the plot was engaging."
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index.get(word, 0) for word in tokens]
    return keras.preprocessing.sequence.pad_sequences([tokens], maxlen=maxlen)[0]

sample_encoded = encode_text(sample_text)
prediction = model.predict(np.array([sample_encoded]))
print(f"Prediction: {'Positive' if prediction[0] > 0.5 else 'Negative'} with confidence {prediction[0][0]:.2f}")
