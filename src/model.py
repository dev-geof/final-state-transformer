import os
import re
import numpy as np
import tensorflow as tf
from keras.layers import Layer


class FloatEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(FloatEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.embedding_weights = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        # inputs: (batch_size, seq_length, input_dim)
        # Reshape to flatten the last dimension
        inputs_reshaped = tf.reshape(inputs, (-1, self.input_dim))

        # Perform the embedding lookup
        embedded = tf.matmul(inputs_reshaped, self.embedding_weights)

        # Reshape back to the original shape
        embedded = tf.reshape(embedded, (-1, tf.shape(inputs)[1], self.output_dim))
        return embedded


def build_transformer(
    nparticles: int,
    nfeatures: int,
    nheads: int,
    nMHAlayers: int,
    nDlayers: int,
    vdropout: float,
    act_fn: str,
    nclass: int,
    training_mode: str,
    embedding: bool,
    embedding_dim: int,
) -> tf.keras.Model:

    """
    Build a transformer model for sequence classification or regression.

    Parameters:
    - nparticles (int): Number of particles in the input sequence.
    - nfeatures (int): Number of features for each particle.
    - nheads (int): Number of attention heads in the multi-head attention layers.
    - nMHAlayers (int): Number of multi-head attention layers.
    - nDlayers (int): Number of dense layers in the feedforward block.
    - vdropout (float): Dropout rate for dropout layers.
    - act_fn (str): Activation function for dense layers.
    - nclass (int): Number of classes for classification tasks.
    - training_mode (str): Mode of training, either "classification" or "regression".
    - embedding (bool): Whether to include an embedding layer.
    - embedding_dim (int): Dimension of the embedding layer.

    Returns:
    - tf.keras.Model: The constructed transformer model.

    This function constructs a transformer model for sequence classification or regression based on the
    specified parameters. It includes multi-head attention layers, dense layers, dropout for regularization,
    and an output layer with appropriate activation function based on the training mode.
    """

    # Instantiate a Keras input tensors
    input_tensor = tf.keras.Input(shape=(nparticles, nfeatures))

    # Include embedding - To be completed
    if embedding == True:

        # Create embedding layer
        embedding_layer = FloatEmbedding(input_dim=nfeatures, output_dim=embedding_dim)

        # Reshape input tensor for embedding layer
        embedded_input = tf.keras.layers.Reshape((nparticles, nfeatures))(input_tensor)

        # Apply embedding layer
        x = embedding_layer(embedded_input)

    else:
        x = input_tensor

    # Add multi-head attention layers
    for i in range(nMHAlayers):
        # Add a multi-head attention layer with the specified number of attention heads and key dimensions
        x = tf.keras.layers.MultiHeadAttention(
            num_heads=nheads,
            key_dim=nparticles,
        )(query=x, value=x, key=x)

    # Flatten the output of the multi-head attention layers
    MHA_output_flattened = tf.keras.layers.Flatten(data_format=None)(x)

    # Define the number of nodes in the first dense layer as the product of the number of particles and features
    nodes_per_layer = nparticles * nfeatures

    # Return MHA output as dense layer
    h = tf.keras.layers.Dense(
        nodes_per_layer,
        activation=act_fn,
    )(MHA_output_flattened)

    # Add n-1 hidden layers
    for _ in range(nDlayers - 1):
        nodes_per_layer = nodes_per_layer  
        # Add dropout to prevent overfitting
        h = tf.keras.layers.Dropout(vdropout)(h)
        h = tf.keras.layers.Dense(
            nodes_per_layer,
            activation=act_fn,
        )(h)

    # Add dropout to the final dense layer
    h = tf.keras.layers.Dropout(vdropout)(h)

    # Add the output layer with a sigmoid activation function for classification and no activation function for regression
    if training_mode == "regression":
        out = tf.keras.layers.Dense(1)(h)
    else:
        out = tf.keras.layers.Dense(nclass, activation="softmax")(h)

    # Create and retrun the model
    model = tf.keras.Model(inputs=input_tensor, outputs=out)

    return model


def compile_model(model, prime_service, mode):
    """
    Compile the TensorFlow model.

    Parameters:
    - model: The TensorFlow model to compile.
    - prime_service (dict): Dictionary containing configuration parameters.
    - mode (str): The training mode ("classification" or "regression").

    Returns:
    None
    """
    opt_params = prime_service[f"transformer_{mode}_parameters"]
    opt = tf.keras.optimizers.Adam(learning_rate=opt_params["learning_rate"])

    if mode == "classification":
        model.compile(
            opt,
            loss="categorical_crossentropy",
            weighted_metrics=["accuracy", tf.keras.metrics.AUC()],
        )
    elif mode == "regression":
        model.compile(
            opt,
            loss="mean_squared_error",
            weighted_metrics=[tf.keras.metrics.MeanSquaredError()],
        )


def get_latest_checkpoint(checkpoint_dir):
    """
    Get the latest model checkpoint file.

    Parameters:
    - checkpoint_dir (str): Directory containing model checkpoints.

    Returns:
    - str or None: The path to the latest checkpoint file, or None if no checkpoints are found.
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")]
    if not checkpoints:
        return None
    latest_checkpoint = max(
        checkpoints, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )
    return os.path.join(checkpoint_dir, latest_checkpoint)


def get_best_checkpoint(dir_path):
    """
    Get the best model checkpoint based on validation accuracy.

    Parameters:
    - dir_path (str): The path to the directory containing checkpoint files.

    Returns:
    - str: The file path of the best checkpoint, or an empty string if none is found.
    """
    best_checkpoint = None
    best_val_acc = 0.0

    for filename in os.listdir(dir_path):
        # Check if the file is a checkpoint file
        if not filename.endswith(".h5"):
            continue

        # Extract the validation accuracy from the file name
        match = re.search(r"acc-(\d+\.\d+)-(\d+\.\d+)", filename)
        if not match:
            continue

        val_acc = float(match.group(2))
        if val_acc > best_val_acc:
            best_checkpoint = os.path.join(dir_path, filename)
            best_val_acc = val_acc

    print("Selected model checkpoint: ", best_checkpoint)

    return best_checkpoint if best_checkpoint else ""


def get_best_checkpoint_regression(dir_path):
    """
    Get the best model checkpoint for regression based on validation mean squared error.

    Parameters:
    - dir_path (str): The path to the directory containing checkpoint files.

    Returns:
    - str: The file path of the best checkpoint, or an empty string if none is found.
    """
    best_checkpoint = None
    best_val_mse = 999999.0

    for filename in os.listdir(dir_path):
        # Check if the file is a checkpoint file
        if not filename.endswith(".h5"):
            continue

        # Extract the validation accuracy from the file name
        match = re.search(r"mse-(\d+\.\d+)-(\d+\.\d+)", filename)
        if not match:
            continue

        val_mse = float(match.group(2))
        if val_mse < best_val_mse:
            best_checkpoint = os.path.join(dir_path, filename)
            best_val_mse = val_mse

    print("Selected model checkpoint: ", best_checkpoint)

    return best_checkpoint if best_checkpoint else ""
