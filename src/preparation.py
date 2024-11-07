import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from termcolor import colored
import logging
import sys

from configuration import configure_logging, load_config, create_directory
from data import (
    weights_extraction,
    data_extraction_transformer,
    get_nfeatures,
    apply_sample_normalization,
)
from plotting import (
    plotter_preparation,
    plotter_correlation,
    plotter_correlation_to_target,
)


def training_data_extraction(
    configfile: str,
):
    """
    Training Dataset Extraction

    Parameters
    ----------
    configfile : str
        configuration file path

    Returns
    -------

    """

    # configure logging module
    configure_logging()
    logging.info(colored("TRAINING DATASET PREPARTION", "green"))
    logging.info(colored("Running training data extraction", "yellow"))

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

    nparticles = prime_service["preparation_configuration"]["nparticles"]
    nfeatures = get_nfeatures(configfile)
    feature_names = prime_service["input_variables"]
    regression_target = prime_service["preparation_configuration"]["regression_target"]
    norm = prime_service["preparation_configuration"]["norm"]
    duplicate = prime_service["preparation_configuration"]["duplicate"]
    batch_size = prime_service["preparation_configuration"]["batch_size"]
    outputdir = prime_service["general_configuration"]["output_directory"]
    validation_plots = prime_service["preparation_configuration"]["validation_plots"]
    validation_plots_log = prime_service["preparation_configuration"][
        "validation_plots_log"
    ]

    # Load training sample list
    sample_list = prime_service["training_samples"]
    keyname_list = [list(dictionary.keys())[0] for dictionary in sample_list]

    # extract training sample dataset and weights
    datasets_list = []
    weights_list = []
    nevents_list = []
    legend_list = []
    colour_list = []
    regression_target_list = []

    for item in sample_list:
        for value in item.values():

            # extract dataset
            datasets_list.append(
                data_extraction_transformer(
                    filepath=value["path"],
                    dataset=value["particle_dataset"],
                    nevents=value["nevents"],
                    fields_of_interest=feature_names,
                    nparticles=nparticles,
                    isvalidation=False,
                )
            )

            # extract weights
            weights_list.append(
                weights_extraction(
                    filepath=value["path"],
                    dataset=value["event_dataset"],
                    weights=value["weights"],
                    nevents=value["nevents"],
                    isvalidation=False,
                )
            )
            # get number of events and colours
            nevents_list.append(value["nevents"] * len(value["path"].split(":")))
            legend_list.append(value["legend"])
            colour_list.append(value["colour"])

            # extract regression target if needed
            if training_mode == "regression":
                regression_target_list.append(
                    weights_extraction(
                        filepath=value["path"],
                        dataset=value["event_dataset"],
                        weights=regression_target,
                        nevents=value["nevents"],
                        isvalidation=False,
                    )
                )

    # Remove everall normalisation difference between signal and background while preserving shape differences.
    weights_list = apply_sample_normalization(weights_list, norm)

    # concatenate input datasets into X_train and Y_train
    X_train = np.concatenate(datasets_list, axis=0)
    X_weights = np.concatenate(weights_list, axis=0)

    # produce hot labels for classification purpose or select target in case of regression
    if training_mode == "regression":
        Y_train = np.concatenate(regression_target_list, axis=0)
    else:
        logging.info(
            colored("Producing hot labels for classification purpose", "yellow")
        )
        hot_labels = np.eye(len(sample_list))
        Y_train = np.repeat(hot_labels, nevents_list, axis=0)

    logging.info(
        colored(
            f"Extracted dataset shapes: events {X_train.shape}, weights {X_weights.shape}, labels {Y_train.shape}",
            "yellow",
        )
    )

    # Instanciate tensorflow dataset and apply reshuffling and batching before splitting it into training and test datasets
    logging.info(colored("Preparation of TensorFlow datasets", "yellow"))
    tf_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train, X_weights))

    # Duplicate the dataset as an option to increase statistics artificially if needed
    if duplicate == True:
        tf_dataset = tf_dataset.concatenate(tf_dataset)
        tf_dataset_batched = tf_dataset.shuffle(
            buffer_size=2 * sum(np.array(nevents_list)),
            reshuffle_each_iteration=True,
        ).batch(batch_size)
        logging.info(colored("Training statistics has been duplicated", "yellow"))
    else:
        tf_dataset_batched = tf_dataset.shuffle(
            buffer_size=sum(np.array(nevents_list)),
            reshuffle_each_iteration=True,
        ).batch(batch_size)

    # Save TensorFlow datasets
    dataset_training_outputdir = outputdir + "/dataset_training"
    tf.data.Dataset.save(tf_dataset_batched, dataset_training_outputdir)
    logging.info(
        colored(
            f"Prepared training dataset saved as TensorFlow dataset in {dataset_training_outputdir}",
            "yellow",
        )
    )

    # Create validation directory
    outputdir_val = str(outputdir) + "/validation"
    create_directory(outputdir_val)

    # Produce a first set of validation plots
    if validation_plots == True:
        logging.info(
            colored(
                f"Producing first set of validation plots",
                "yellow",
            )
        )

        plotter_preparation(
            datasets_list=datasets_list,
            sample_list=keyname_list,
            weights_list=weights_list,
            legend_list=legend_list,
            colour_list=colour_list,
            nparticles=nparticles,
            fields_of_interest=feature_names,
            log_scale=validation_plots_log,
            outputdir=outputdir_val,
        )

        # Create input to target correleation plot for regression
        if training_mode == "regression":
            plotter_correlation_to_target(
                input_dataset=X_train,
                target_dataset=Y_train,
                nfeatures=nfeatures,
                nparticles=nparticles,
                feature_names=feature_names,
                outputdir=outputdir_val,
            )

    else:
        logging.info(
            colored(
                f"No validation plots considered",
                "yellow",
            )
        )

    logging.info(colored("Training sample preparation completed", "yellow"))

def main():
    """
    Entry point for running training data preparation.

    Parses command-line arguments and invokes the training_data_extraction function.
    """
    parser = ArgumentParser(description="Run training data prepration")
    parser.add_argument(
        "--configfile",
        action="store",
        dest="configfile",
        default="config/config.yaml",
        help="Configuration file path",
    )
    args = vars(parser.parse_args())
    training_data_extraction(**args)


if __name__ == "__main__":
    main()

