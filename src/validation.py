import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import sys
import h5py
import logging
import datetime
import numpy as np
import tensorflow as tf

# import tf2onnx
from termcolor import colored
from sklearn.metrics import confusion_matrix

from configuration import (
    configure_logging,
    load_config,
)
from model import (
    get_best_checkpoint,
    get_best_checkpoint_regression,
    FloatEmbedding,
)
from plotting import (
    plotter_confusion_matrix,
    plotter_scores,
    plotter_distributions_processes,
    plotter_distributions_signal_vs_background,
    Plotter_ROC,
    Plotter_Efficiency,
    plotter_residuals,
    plotter_regression_ratio,
    plotter_regression_prediction,
    visualize_embedding_space,
)

from data import (
    weights_extraction,
    ghost_extraction,
    data_extraction_transformer,
    building_new,
)


def transformer_validation(
    configfile: str,
):

    """
    Perform series of validation steps for trained model model based on the configuration.

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
    model_path = prime_service["general_configuration"]["output_directory"]
    analysis_title = prime_service["general_configuration"]["analysis_title"]

    regression_target = prime_service["preparation_configuration"]["regression_target"]
    nparticles = prime_service["preparation_configuration"]["nparticles"]
    regression_target_label = prime_service["preparation_configuration"][
        "regression_target_label"
    ]

    luminosity_scaling = prime_service["validation_configuration"]["luminosity_scaling"]
    save_predictions = prime_service["validation_configuration"]["save_predictions"]
    save_onnx_model = prime_service["validation_configuration"]["save_onnx_model"]
    plot_proba = prime_service["validation_configuration"]["plot_proba"]
    plot_discriminant = prime_service["validation_configuration"]["plot_discriminant"]
    plot_roc = prime_service["validation_configuration"]["plot_roc"]
    plot_confusion = prime_service["validation_configuration"]["plot_confusion"]
    plot_model = prime_service["validation_configuration"]["plot_model"]
    plot_scores = prime_service["validation_configuration"]["plot_scores"]
    plot_embedding = prime_service["validation_configuration"]["plot_embedding"]
    logy = prime_service["validation_configuration"]["plot_log_probabilities"]

    feature_names = prime_service["input_variables"]
    ghost_names = prime_service["ghost_variables"]

    logging.info(colored("FINAL STATE TRANSFORMER VALIDATION", "green"))

    # Loading models
    logging.info(colored("Loading trained model", "yellow"))

    # Load the model from the checkpoint with custom object scope (for embedding)
    with tf.keras.utils.custom_object_scope({"FloatEmbedding": FloatEmbedding}):

        if training_mode == "classification":
            classifier = tf.keras.models.load_model(
                get_best_checkpoint(
                    str(model_path) + "/classification_training/checkpoints/"
                ),
            )
        elif training_mode == "regression":
            classifier = tf.keras.models.load_model(
                get_best_checkpoint_regression(
                    str(model_path) + "/regression_training/checkpoints/"
                )
            )
        else:
            logging.info(
                colored(
                    "Training mode not supported (accepted mode: classification or regression)",
                    "red",
                )
            )
            sys.exit(0)

    # Plot the model architecture and save it to a file
    # os.makedirs(model_path + "/validation", exist_ok=True)
    outputdir = str(model_path) + "/validation"
    if os.path.exists(outputdir) == False:
        os.mkdir(outputdir)

    # if save_onnx_model == True:
    #    # Convert the TensorFlow model to ONNX
    #    onnx_model, _ = tf2onnx.convert.from_keras(classifier)
    #    # Save the ONNX model to a file
    #    onnx_file_path = (
    #        model_path
    #        + "/validation/validation_model_"
    #        + training_mode
    #        + ".onnx"
    #    )
    #    tf2onnx.save_model(onnx_file_path, onnx_model)

    if plot_model == True:
        arch_name = (
            model_path
            + "/validation/validation_model_architecture_"
            + training_mode
            + ".pdf"
        )
        tf.keras.utils.plot_model(
            classifier,
            to_file=arch_name,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=False,
            expand_nested=True,
            layer_range=None,
            show_layer_activations=True,
        )

    # Get validation samples and run predictions
    logging.info(
        colored(
            f"Opening training and validation datasets and running predictions",
            "yellow",
        )
    )

    # Load sample lists
    sample_list = prime_service["training_samples"]
    keyname_list = [list(dictionary.keys())[0] for dictionary in sample_list]
    predictions_list = []
    inputs_list = []
    weights_list = []
    legend_list = []
    colour_list = []
    nevents_list = []
    type_list = []
    regression_target_list = []

    val_sample_list = prime_service["validation_samples"]
    val_keyname_list = [list(dictionary.keys())[0] for dictionary in val_sample_list]
    val_predictions_list = []
    val_inputs_list = []
    val_weights_list = []
    val_legend_list = []
    val_colour_list = []
    val_nevents_list = []
    val_type_list = []
    val_regression_target_list = []
    val_dsname_list = []
    val_ghost_list = []

    # Loop over training samples list
    for item in sample_list:
        for value in item.values():

            # extract dataset
            validation_sample = data_extraction_transformer(
                filepath=value["path"],
                dataset=value["particle_dataset"],
                nevents=value["nevents"],
                fields_of_interest=feature_names,
                nparticles=nparticles,
                isvalidation=False,
            )
            inputs_list.append(validation_sample)
            predictions_list.append(classifier.predict(validation_sample, verbose=1))

            # extract weights
            weight_original = weights_extraction(
                filepath=value["path"],
                dataset=value["event_dataset"],
                weights=value["weights"],
                nevents=value["nevents"],
                isvalidation=False,
            )
            weight_new = (
                weight_original
                * luminosity_scaling
                / (
                    np.sum(weight_original)
                    / (value["cross_section"] * value["branching_ratio"])
                )
            )
            weights_list.append(weight_new)

            # get legends and colours
            legend_list.append(value["legend"])
            colour_list.append(value["colour"])
            nevents_list.append(int(value["nevents"]) * len(value["path"].split(":")))
            type_list.append(value["type"])

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

    # Loop over validation samples list
    v = 0
    for item in val_sample_list:
        for value in item.values():

            # extract dataset
            validation_sample = data_extraction_transformer(
                filepath=value["path"],
                dataset=value["particle_dataset"],
                nevents=value["nevents"],
                fields_of_interest=feature_names,
                nparticles=nparticles,
                isvalidation=False,
            )
            val_inputs_list.append(validation_sample)
            val_predictions_list.append(
                classifier.predict(validation_sample, verbose=1)
            )

            # extract weights
            weight_original = weights_extraction(
                filepath=value["path"],
                dataset=value["event_dataset"],
                weights=value["weights"],
                nevents=value["nevents"],
                isvalidation=False,
            )

            print(
                "ORI -> sum of weights:",
                np.sum(weight_original),
                "xs:",
                (value["cross_section"] * value["branching_ratio"]),
                "Lumi:",
                np.sum(weight_original)
                / (value["cross_section"] * value["branching_ratio"]),
            )

            weight_new = (
                weight_original
                * luminosity_scaling
                / (
                    np.sum(weight_original)
                    / (value["cross_section"] * value["branching_ratio"])
                )
            )

            print(
                "NEW -> sum of weights:",
                np.sum(weight_new),
                "xs:",
                (value["cross_section"] * value["branching_ratio"]),
                "Lumi:",
                np.sum(weight_new)
                / (value["cross_section"] * value["branching_ratio"]),
            )

            val_weights_list.append(weight_new)

            # ghost feature extraction
            ghost = ghost_extraction(
                filepath=value["path"],
                dataset=value["event_dataset"],
                nevents=value["nevents"],
                fields_of_interest=ghost_names,
                isvalidation=False,
            )
            val_ghost_list.append(ghost)

            # get legends and colours
            val_legend_list.append(value["legend"])
            val_colour_list.append(value["colour"])
            val_nevents_list.append(
                int(value["nevents"]) * len(value["path"].split(":"))
            )
            val_type_list.append(value["type"])

            # extract regression target if needed
            if training_mode == "regression":
                val_regression_target_list.append(
                    weights_extraction(
                        filepath=value["path"],
                        dataset=value["event_dataset"],
                        weights=regression_target,
                        nevents=value["nevents"],
                        isvalidation=False,
                    )
                )
            # get dataset names
            val_dsname_list.append(val_keyname_list[v])
        v += 1

    ######################

    # Save predictions into hdf5
    if save_predictions == True:

        f = h5py.File(str(model_path) + "/validation/prediction.keras", "w")

        for d in range(len(val_sample_list)):

            # Include weights
            DS = np.array(val_weights_list[d], dtype=[("weights", "<f4")])

            # Include ghost features
            for name, g in zip(ghost_names, np.transpose(val_ghost_list[d])):
                DS = building_new(g, name, DS, val_nevents_list[d])

            # Include output probabilities
            for p in range(len(sample_list)):
                DS = building_new(
                    val_predictions_list[d][:, p], f"p{p}", DS, val_nevents_list[d]
                )

            f.create_dataset(val_dsname_list[d], data=DS)

        f.close()
        logging.info(
            colored(f"Validation sample predictions saved in HDF5 file", "yellow")
        )

    ######################

    # Visualizing embedding space
    logging.info(colored("Visualizing embedding space ", "yellow"))

    if plot_embedding == True:

        inputs = []
        embeddings = []

        for s in range(len(inputs_list)):
            temp_input = classifier.get_layer("input_1")(inputs_list[s][:100])
            batch_size_in, sequence_length_in, embedding_dim_in = temp_input.shape
            temp_input = np.reshape(
                temp_input, (batch_size_in * sequence_length_in, embedding_dim_in)
            )
            inputs.append(temp_input)

            temp_embedding = classifier.get_layer("float_embedding")(
                inputs_list[s][:100]
            )
            batch_size, sequence_length, embedding_dim = temp_embedding.shape
            temp_embedding = np.reshape(
                temp_embedding, (batch_size * sequence_length, embedding_dim)
            )
            embeddings.append(temp_embedding)

        visualize_embedding_space(inputs, embeddings, legend_list, model_path)

    ######################

    if training_mode == "regression":

        # Plotting residuals
        logging.info(colored("Plotting regression residuals", "yellow"))

        plotter_residuals(
            predictions_list=predictions_list,
            regression_target_list=regression_target_list,
            weights_list=weights_list,
            legend_list=legend_list,
            colour_list=colour_list,
            outputdir=model_path,
        )

        # Plotting ratio
        logging.info(colored("Plotting regression ratio", "yellow"))

        plotter_regression_ratio(
            predictions_list=predictions_list,
            regression_target_list=regression_target_list,
            weights_list=weights_list,
            legend_list=legend_list,
            colour_list=colour_list,
            outputdir=model_path,
        )

        # Plotting predicted distribution
        logging.info(colored("Plotting regression predicted distribution", "yellow"))

        plotter_regression_prediction(
            predictions_list=predictions_list,
            regression_target_list=regression_target_list,
            weights_list=weights_list,
            xlabel=regression_target_label,
            logy=logy,
            legend_list=legend_list,
            colour_list=colour_list,
            outputdir=model_path,
        )

    ######################

    else:
        # Plotting confusion matrix
        logging.info(
            colored("Computing confusion matrix based on training samples", "yellow")
        )

        rounded_prediction_validation = np.argmax(
            np.concatenate(predictions_list), axis=1
        )
        rounded_true_validation = np.argmax(
            np.repeat(np.eye(len(sample_list)), nevents_list, axis=0), axis=1
        )

        cm = confusion_matrix(
            y_true=rounded_true_validation,
            y_pred=rounded_prediction_validation,
            sample_weight=np.concatenate(weights_list),
        )

        if plot_confusion == True:
            plotter_confusion_matrix(
                cm=cm,
                x_classes=legend_list,
                y_classes=legend_list,
                normalize=True,
                title="FINAL STATE TRANSFORMER - Confusion Matrix",
                outputdir=model_path
                + "/validation/validation_confusion_martix_train_"
                + training_mode
                + ".pdf",
            )

        logging.info(
            colored("Computing confusion matrix based on validation samples", "yellow")
        )

        val_rounded_prediction_validation = np.argmax(
            np.concatenate(val_predictions_list), axis=1
        )
        val_rounded_true_validation = np.argmax(
            np.repeat(np.eye(len(val_sample_list)), val_nevents_list, axis=0), axis=1
        )

        val_cm = confusion_matrix(
            y_true=val_rounded_true_validation,
            y_pred=val_rounded_prediction_validation,
            sample_weight=np.concatenate(val_weights_list),
        )

        if plot_confusion == True:
            plotter_confusion_matrix(
                cm=val_cm,
                x_classes=legend_list,
                y_classes=val_legend_list,
                normalize=True,
                title="FINAL STATE TRANSFORMER - Confusion Matrix",
                outputdir=model_path
                + "/validation/validation_confusion_martix_val_"
                + training_mode
                + ".pdf",
            )

        #####################
        # Calculate Accuracy, Precision, Recall, and F1-Score per process:
        logging.info(
            colored(
                "Computing Accuracy, Precision, Recall, and F1-Score per process",
                "yellow",
            )
        )

        misclass_dif = plotter_scores(
            cm=cm,
            classes=legend_list,
            normalize=True,
            title="FINAL STATE TRANSFORMER",
            plot=plot_scores,
            outputdir=model_path
            + "/validation/validation_scores_"
            + training_mode
            + ".pdf",
        )

        # compute background class weights
        bkg_class_score = []
        for x in range(len(sample_list)):
            if type_list[x] == "background":
                bkg_class_score.append(misclass_dif[x] * np.sum(weights_list[x]))
            else:
                bkg_class_score.append(0.0)

        bkg_class_weights = []
        for x in range(len(sample_list)):
            bkg_class_weights.append(
                bkg_class_score[x] / np.sum(np.array(bkg_class_score))
            )
            print(
                "class weight:",
                "{:.4f}".format(bkg_class_score[x] / np.sum(np.array(bkg_class_score))),
                keyname_list[x],
            )

        ####################
        # Plotting sample weighted distribution
        logging.info(colored("Building classifier discriminant", "yellow"))

        disc = []
        disc_weighted = []

        # compute classifier discriminant for each validation sample
        for x in range(len(val_sample_list)):

            psig = None
            pbkg = None
            pbkg_weights = None

            # Check signal and background probabilities for a given process
            for y in range(len(sample_list)):

                if type_list[y] == "signal":

                    if psig is None:
                        psig = val_predictions_list[x][:, y]
                    else:
                        psig += val_predictions_list[x][:, y]

                elif type_list[y] == "background":

                    if pbkg is None:
                        pbkg = val_predictions_list[x][:, y]
                    else:
                        pbkg += val_predictions_list[x][:, y]

                    if pbkg_weights is None:
                        pbkg_weights = (
                            bkg_class_weights[y] * val_predictions_list[x][:, y]
                        )
                    else:
                        pbkg_weights += (
                            bkg_class_weights[y] * val_predictions_list[x][:, y]
                        )

            disc.append(np.log(psig / pbkg))
            disc_weighted.append(np.log(psig / pbkg_weights))

        # Plotting sample weight distribution
        if plot_discriminant == True:
            logging.info(
                colored("Plotting classifier discriminant distributions", "yellow")
            )

            if os.path.exists(str(model_path) + "/validation/dist") == False:
                os.mkdir(str(model_path) + "/validation/dist")

            plotter_distributions_processes(
                inputs=disc,
                weights=val_weights_list,
                legend=val_legend_list,
                colour=val_colour_list,
                xlabel="Discriminant",
                logy=logy,
                title="FINAL STATE TRANSFORMER",
                subtitle=analysis_title,
                outputdir=model_path
                + "/validation/dist/validation_discriminant_"
                + training_mode
                + ".pdf",
            )

            plotter_distributions_processes(
                inputs=disc_weighted,
                weights=val_weights_list,
                legend=val_legend_list,
                colour=val_colour_list,
                xlabel="Discriminant",
                logy=logy,
                title="FINAL STATE TRANSFORMER",
                subtitle=analysis_title,
                outputdir=model_path
                + "/validation/dist/validation_discriminant_weighted_"
                + training_mode
                + ".pdf",
            )

            plotter_distributions_signal_vs_background(
                inputs=disc,
                weights=val_weights_list,
                type=val_type_list,
                xlabel="Discriminant",
                logy=logy,
                title="FINAL STATE TRANSFORMER",
                subtitle=analysis_title,
                outputdir=model_path
                + "/validation/dist/validation_discriminant_sig_vs_bkg_"
                + training_mode
                + ".pdf",
            )

        if plot_proba == True:
            logging.info(
                colored("Plotting classifier probability distributions", "yellow")
            )

            if os.path.exists(str(model_path) + "/validation/dist") == False:
                os.mkdir(str(model_path) + "/validation/dist")

            for x in range(len(predictions_list)):

                pred = [array[:, x] for array in val_predictions_list]

                plotter_distributions_processes(
                    inputs=pred,
                    weights=val_weights_list,
                    legend=val_legend_list,
                    colour=val_colour_list,
                    xlabel=legend_list[x] + " output probability",
                    logy=logy,
                    title="FINAL STATE TRANSFORMER",
                    subtitle=analysis_title,
                    outputdir=model_path
                    + "/validation/dist/validation_probability_"
                    + keyname_list[x]
                    + "_"
                    + training_mode
                    + ".pdf",
                )

                plotter_distributions_signal_vs_background(
                    inputs=pred,
                    weights=val_weights_list,
                    type=val_type_list,
                    xlabel=legend_list[x] + " output probability",
                    logy=logy,
                    title="FINAL STATE TRANSFORMER",
                    subtitle=analysis_title,
                    outputdir=model_path
                    + "/validation/dist/validation_probability_sig_vs_bkg_"
                    + keyname_list[x]
                    + "_"
                    + training_mode
                    + ".pdf",
                )

        #######################
        # Plotting ROC curves
        if plot_roc == True:
            logging.info(colored("Plotting ROC curves", "yellow"))

            if os.path.exists(str(model_path) + "/validation/roc") == False:
                os.mkdir(str(model_path) + "/validation/roc")

            if plot_discriminant == True:

                Plotter_ROC(
                    inputs=disc,
                    weights=val_weights_list,
                    legend=val_legend_list,
                    colour=val_colour_list,
                    title="FINAL STATE TRANSFORMER",
                    keyname=val_keyname_list,
                    training_mode=training_mode,
                    outputdir=model_path
                    + "/validation/roc/validation_discriminant_ROC_",
                )

                Plotter_ROC(
                    inputs=disc_weighted,
                    weights=val_weights_list,
                    legend=val_legend_list,
                    colour=val_colour_list,
                    title="FINAL STATE TRANSFORMER",
                    keyname=val_keyname_list,
                    training_mode=training_mode,
                    outputdir=model_path
                    + "/validation/roc/validation_discriminant_weighted_ROC_",
                )

            if plot_proba == True:
                for x in range(len(predictions_list)):

                    pred = [array[:, x] for array in val_predictions_list]

                    Plotter_ROC(
                        inputs=pred,
                        weights=val_weights_list,
                        legend=val_legend_list,
                        colour=val_colour_list,
                        title="FINAL STATE TRANSFORMER",
                        keyname=val_keyname_list,
                        training_mode=training_mode,
                        outputdir=model_path
                        + "/validation/roc/validation_probability_"
                        + keyname_list[x]
                        + "_ROC_",
                    )

        #######################
        # Plotting Efficiency curves
        if plot_roc == True:
            logging.info(colored("Plotting efficiency curves", "yellow"))

            if os.path.exists(str(model_path) + "/validation/efficiency") == False:
                os.mkdir(str(model_path) + "/validation/efficiency")

            if plot_discriminant == True:
                Plotter_Efficiency(
                    inputs=disc_weighted,
                    weights=val_weights_list,
                    legend=val_legend_list,
                    colour=val_colour_list,
                    xlabel="Discriminant",
                    title="FINAL STATE TRANSFORMER",
                    keyname=val_keyname_list,
                    training_mode=training_mode,
                    outputdir=model_path
                    + "/validation/efficiency/validation_discriminant_weighted_efficiency_",
                )

                Plotter_Efficiency(
                    inputs=disc,
                    weights=val_weights_list,
                    legend=val_legend_list,
                    colour=val_colour_list,
                    xlabel="Discriminant",
                    title="FINAL STATE TRANSFORMER",
                    keyname=val_keyname_list,
                    training_mode=training_mode,
                    outputdir=model_path
                    + "/validation/efficiency/validation_discriminant_efficiency_",
                )

            if plot_proba == True:
                for x in range(len(predictions_list)):

                    pred = [array[:, x] for array in val_predictions_list]

                    Plotter_Efficiency(
                        inputs=pred,
                        weights=val_weights_list,
                        legend=val_legend_list,
                        colour=val_colour_list,
                        xlabel=val_legend_list[x] + " output probability",
                        title="FINAL STATE TRANSFORMER",
                        keyname=val_keyname_list,
                        training_mode=training_mode,
                        outputdir=model_path
                        + "/validation/efficiency/validation_probability_"
                        + keyname_list[x]
                        + "_efficiency_",
                    )

    #######################
    logging.info(colored("Validation completed", "yellow"))


def main():
    """
    Entry point for running Transformer validation.

    Parses command-line arguments and invokes the transformer_validation function.
    """
    parser = ArgumentParser(description="run Transformer validation")
    parser.add_argument(
        "--configfile",
        action="store",
        dest="configfile",
        default="config/config.yaml",
        help="Configuration file path",
    )

    args = vars(parser.parse_args())

    transformer_validation(**args)

if __name__ == "__main__":
    main()

