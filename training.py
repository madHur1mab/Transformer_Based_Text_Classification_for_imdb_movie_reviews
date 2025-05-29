import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sample Dataset (Simple addition problem as a sequence-to-sequence task)
def generate_data(num_samples, seq_length):
    x = np.random.randint(1, 10, size=(num_samples, seq_length))
    y = np.sum(x, axis=1, keepdims=True)
    y = np.concatenate([np.zeros((num_samples, seq_length - 1)), y], axis=1)
    return x, y

num_samples = 10000
seq_length = 5
input_vocab_size = 10  # Digits 1-9 plus padding (0)
output_vocab_size = 50  # Sums up to 45 max, plus padding, etc.

x_train, y_train = generate_data(num_samples, seq_length)
x_test, y_test = generate_data(1000, seq_length)

# One-hot encode the input and output
def one_hot_encode(data, vocab_size):
    return tf.one_hot(data, depth=vocab_size)

x_train_encoded = one_hot_encode(x_train, input_vocab_size)
y_train_encoded = one_hot_encode(y_train, output_vocab_size)
x_test_encoded = one_hot_encode(x_test, input_vocab_size)
y_test_encoded = one_hot_encode(y_test, output_vocab_size)

# Transformer Components (from scratch)
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

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                          for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Expand dimensions of pos_encoding to match batch size
        pos_encoding = self.pos_encoding[tf.newaxis, :seq_len, :]
        x = x + pos_encoding

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

# Custom layer to convert one-hot to indices
class OneHotToIndices(layers.Layer):
    def call(self, inputs):
        return tf.argmax(inputs, axis=-1)

# Create the Transformer Model
def create_transformer(num_layers, d_model, num_heads, dff, input_vocab_size, output_vocab_size, pe_input, pe_target, rate=0.1):
    inputs = layers.Input(shape=(None, input_vocab_size))
    x = OneHotToIndices()(inputs)  # Use custom layer instead of tf.argmax
    enc_output = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input)(x, training=False)
    final_layer = layers.Dense(output_vocab_size, activation='softmax')(enc_output)
    model = keras.Model(inputs=inputs, outputs=final_layer)
    return model

# Model Parameters
num_layers = 2
d_model = 64
num_heads = 4
dff = 128
pe_input = seq_length
pe_target = seq_length

# Create and Compile the Model
transformer = create_transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    output_vocab_size=output_vocab_size,
    pe_input=pe_input,
    pe_target=pe_target
)

transformer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
transformer.fit(x_train_encoded, y_train_encoded, epochs=20, batch_size=32, validation_data=(x_test_encoded, y_test_encoded))

# Evaluate the Model
loss, accuracy = transformer.evaluate(x_test_encoded, y_test_encoded)
print(f"Test Accuracy: {accuracy}")
