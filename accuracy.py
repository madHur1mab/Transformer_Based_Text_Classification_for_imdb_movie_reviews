import matplotlib.pyplot as plt
import seaborn as sns

# Plot training & validation accuracy/loss
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Visualize sample predictions
def visualize_predictions(model, x_test, y_test, num_samples=5):
    sample_indices = np.random.choice(len(x_test), num_samples)
    # Convert sample_indices to a TensorFlow tensor
    sample_indices = tf.convert_to_tensor(sample_indices, dtype=tf.int32)
    x_sample = tf.gather(x_test, sample_indices)  # Use tf.gather for indexing
    y_sample = tf.gather(y_test, sample_indices)

    predictions = model.predict(x_sample)
    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(y_sample, axis=-1)

    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)

        # Plot input sequence
        input_seq = np.argmax(x_sample[i], axis=-1)
        plt.bar(np.arange(len(input_seq))-0.2, input_seq, width=0.4, label='Input Numbers')

        # Plot true and predicted sums
        true_sum = true_classes[i][-1]  # Last element is the sum
        pred_sum = predicted_classes[i][-1]

        plt.bar(len(input_seq)-0.2, true_sum, width=0.4, color='g', label='True Sum')
        plt.bar(len(input_seq)+0.2, pred_sum, width=0.4, color='r', label='Predicted Sum')

        plt.title(f'Sample {i+1} - True Sum: {true_sum}, Predicted Sum: {pred_sum}')
        plt.ylabel('Value')
        plt.xlabel('Position in Sequence')
        plt.legend()

    plt.tight_layout()
    plt.show()

# After training, call these visualization functions
history = transformer.fit(x_train_encoded, y_train_encoded,
                         epochs=20, batch_size=32,
                         validation_data=(x_test_encoded, y_test_encoded))

# Plot training history
plot_training_history(history)

# Visualize some predictions
visualize_predictions(transformer, x_test_encoded, y_test_encoded, num_samples=5)

# Confusion matrix for the sums
def plot_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_true = np.argmax(y_test, axis=-1)[:, -1]  # Get just the sums
    y_pred = np.argmax(y_pred, axis=-1)[:, -1]

    # Create confusion matrix
    cm = tf.math.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Sum Predictions')
    plt.xlabel('Predicted Sum')
    plt.ylabel('True Sum')
    plt.show()

plot_confusion_matrix(transformer, x_test_encoded, y_test_encoded)
