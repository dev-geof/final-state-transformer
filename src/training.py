import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore")
import datetime
import re
import sys
from argparse import ArgumentParser
import tensorflow as tf
import yaml
from termcolor import colored
import logging

from configuration import configure_logging, load_config
from model import build_transformer, compile_model, get_latest_checkpoint
from data import get_nfeatures, get_training_datasets
from callbacks import TrainingPlot, TrainingPlot_Regression

# To check GPU vs. CPU usage
# tf.debugging.set_log_device_placement(True)


def transformer_training(
    configfile: str,
):
    """
    Train a Transformer model for either classification or regression based on the configuration.

    Parameters:""
    - configfile (str): Path to the YAML configuration file.

    Returns:
    None
    """

    # configure logging module
    configure_logging()

    # Loading config file
    prime_service = load_config(configfile)

    training_mode = prime_service["general_configuration"]["training_mode"]
    if training_mode != "classification" and training_mode != "regression":
        logging.info(
            colored(
                "Training mode not supported (accepted mode: classification, regression)",
                "red",
            )
        )
        sys.exit(0)

    outputdir = prime_service["general_configuration"]["output_directory"]
    use_gpu = prime_service["general_configuration"]["use_gpu"]

    nparticles = prime_service["preparation_configuration"]["nparticles"]
    nfeatures = get_nfeatures(configfile)
    samples = prime_service[f"training_samples"]
    nclass = len(samples) if training_mode == "classification" else 1

    nheads = prime_service[f"transformer_{training_mode}_parameters"]["nheads"]
    nMHAlayers = prime_service[f"transformer_{training_mode}_parameters"]["nMHAlayers"]
    nDlayers = prime_service[f"transformer_{training_mode}_parameters"]["nDlayers"]
    vdropout = prime_service[f"transformer_{training_mode}_parameters"]["vdropout"]
    act_fn = prime_service[f"transformer_{training_mode}_parameters"]["act_fn"]
    embedding = prime_service[f"transformer_{training_mode}_parameters"]["embedding"]
    embedding_dim = prime_service[f"transformer_{training_mode}_parameters"][
        "embedding_dim"
    ]
    # embedding_discrete_threshold = prime_service[f"transformer_{training_mode}_parameters"][
    #    "embedding_discrete_threshold"
    # ]
    nepochs = prime_service[f"transformer_{training_mode}_parameters"]["nepochs"]
    verbose = prime_service[f"transformer_{training_mode}_parameters"]["verbose"]

    logging.info(colored("TRANSFORMER TRAINING & TEST", "green"))

    # Use GPU for opening training/test dataset and building the model
    if use_gpu == True:

        # List available physical devices (CPUs and GPUs)
        devices = tf.config.experimental.list_physical_devices()

        # Filter GPUs from the list of devices
        gpu_devices = [device.name for device in devices if "GPU" in device.name]

        # Print the list of GPU devices
        logging.info(colored(f"Available GPU devices: {gpu_devices}", "yellow"))

        # Getting training and test datasets
        datasetdir = str(outputdir) + "/dataset_training"
        logging.debug(
            colored(f"Opening training dataset located in {datasetdir}", "yellow")
        )
        tf_dataset_train, tf_dataset_test = get_training_datasets(
            prime_service, training_mode, datasetdir
        )

        logging.info(colored("Building classification Transformer model", "yellow"))

        # building transformer model
        with tf.device("/device:GPU:0"):
            model_transformer = build_transformer(
                nparticles=nparticles,
                nfeatures=nfeatures,
                nheads=nheads,
                nMHAlayers=nMHAlayers,
                nDlayers=nDlayers,
                vdropout=vdropout,
                act_fn=act_fn,
                nclass=nclass,
                training_mode=training_mode,
                embedding=embedding,
                embedding_dim=embedding_dim,
                # embedding_discrete_threshold=embedding_discrete_threshold,
            )
    else:

        # Getting training and test datasets
        datasetdir = str(outputdir) + "/dataset_training"
        logging.debug(
            colored(f"Opening training dataset located in {datasetdir}", "yellow")
        )
        tf_dataset_train, tf_dataset_test = get_training_datasets(
            prime_service, training_mode, datasetdir
        )

        logging.info(colored("Building classification Transformer model", "yellow"))

        # building transformer model
        model_transformer = build_transformer(
            nparticles=nparticles,
            nfeatures=nfeatures,
            nheads=nheads,
            nMHAlayers=nMHAlayers,
            nDlayers=nDlayers,
            vdropout=vdropout,
            act_fn=act_fn,
            nclass=nclass,
            training_mode=training_mode,
            embedding=embedding,
            embedding_dim=embedding_dim,
            # embedding_discrete_threshold=embedding_discrete_threshold,
        )

    # print model summary
    model_transformer.summary()

    # Compiling model
    logging.info(colored(f"Compiling model", "yellow"))
    compile_model(model_transformer, prime_service, training_mode)

    # Check if there are existing checkpoints
    logging.info(colored(f"Checking existing checkpoints", "yellow"))
    checkpoint_dir = f"{outputdir}/{training_mode}_training/checkpoints/"

    if os.path.exists(checkpoint_dir):
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

        if latest_checkpoint:
            logging.info(
                colored(
                    f"Resuming training from checkpoint: {latest_checkpoint}",
                    "yellow",
                )
            )
            model_transformer = tf.keras.models.load_model(latest_checkpoint)
    else:
        logging.info(
            colored(
                f"No existing checkpoints found. Starting training from scratch.",
                "yellow",
            )
        )

    # fit the model, input and target to be provided
    log_dir = f"{outputdir}/{training_mode}_training/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')} "

    plot_losses = (
        TrainingPlot(f"{outputdir}/{training_mode}_training")
        if training_mode == "classification"
        else TrainingPlot_Regression(f"{outputdir}/{training_mode}_training")
    )
    LRMonitor = (
        "val_accuracy"
        if training_mode == "classification"
        else "val_mean_squared_error"
    )

    class_CPname = (
        outputdir
        + "/classification_training/checkpoints/model.{epoch:02d}-loss-{loss:.5f}-{val_loss:.5f}-acc-{accuracy:.5f}-{val_accuracy:.5f}-auc-{auc:.5f}-{val_auc:.5f}.keras"
    )
    reg_CPname = filepath = (
        outputdir
        + "/regression_training/checkpoints/model-{epoch:02d}-loss-{loss:.5f}-{val_loss:.5f}-mse-{mean_squared_error:.5f}-{val_mean_squared_error:.5f}.keras"
    )
    CPoutput = class_CPname if training_mode == "classification" else reg_CPname

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=LRMonitor, factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath=CPoutput),
        plot_losses,
    ]

    # Use GPU for fitting the model
    if use_gpu == True:
        with tf.device("/device:GPU:0"):
            model_history = model_transformer.fit(
                tf_dataset_train,
                epochs=nepochs,
                callbacks=my_callbacks,
                validation_data=tf_dataset_test,
                verbose=verbose,
            )
    else:
        model_history = model_transformer.fit(
            tf_dataset_train,
            epochs=nepochs,
            callbacks=my_callbacks,
            validation_data=tf_dataset_test,
            verbose=verbose,
        )

    logging.info(colored("Training completed", "yellow"))


def main():
    """
    Entry point for running Transformer training.

    Parses command-line arguments and invokes the transformer_training function.
    """
    parser = ArgumentParser(description="run Transformer training")
    parser.add_argument(
        "--configfile",
        action="store",
        dest="configfile",
        default="config/config.yaml",
        help="Configuration file",
    )
    args = vars(parser.parse_args())
    transformer_training(**args)

if __name__ == "__main__":
    main()


