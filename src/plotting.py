import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from puma import Histogram, HistogramPlot


def plotter_preparation(
    datasets_list,
    weights_list,
    legend_list,
    colour_list,
    nparticles,
    fields_of_interest,
    log_scale,
    outputdir,
):

    # loop over the input variables
    for v in range(len(fields_of_interest)):
        var = str(fields_of_interest[v])

        for p in range(nparticles):

            array_stack = np.concatenate(datasets_list, axis=0)
            xmin = np.min(array_stack[:, p, v])
            xmax = np.max(array_stack[:, p, v])

            # initialise Histogram plot
            plot = HistogramPlot(
                xlabel=f"{var} [{p}] ",
                ylabel="Normalised number of final state objects",
                bins=40,
                bins_range=(xmin, xmax),
                norm=False,
                logy=log_scale,
                figsize=(6, 5),
                n_ratio_panels=1,
                atlas_first_tag="FINAL STATE TRANSFORMER",
                atlas_second_tag="Input feature validation",
                atlas_brand=None,
            )

            for d in range(len(datasets_list)):
                if d == 0:
                    plot.add(
                        Histogram(
                            datasets_list[d][:, p, v],
                            weights=weights_list[d],
                            label=legend_list[d],
                            colour=colour_list[d],
                            linestyle="solid",
                        ),
                        reference=True,
                    )
                else:
                    plot.add(
                        Histogram(
                            datasets_list[d][:, p, v],
                            weights=weights_list[d],
                            label=legend_list[d],
                            colour=colour_list[d],
                            linestyle="solid",
                        ),
                    )
            plot.draw()
            plot.savefig(
                f"{outputdir}/{var}_{v}_{p}.pdf",
                transparent=True,
            )


def plotter_correlation(
    data,
    outputdir,
    process,
    nparticles,
    nfeatures,
):

    corr_matrix = [[0 for _ in range(nfeatures)] for _ in range(nparticles)]
    for i in range(nparticles):
        for j in range(nfeatures):
            cov = np.corrcoef(data[:, i, j].flatten(), data[:, i, j].flatten())
            corr_matrix[i][j] = cov[1, 0]

    plt.figure(figsize=(5, 5))
    boundary = (np.absolute(corr_matrix)).max() * 1.2
    plt.imshow(corr_matrix, cmap="jet", vmin=-boundary, vmax=boundary, origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("Linear correlation")
    plt.title(f"FINAL STATE TRANSFORMER", loc="left")
    plt.title(f"Correlation Matrix - {process}")
    plt.xlabel("Feature index")
    plt.ylabel("Feature index")
    plt.tight_layout()
    plt.draw()
    plt.savefig(
        f"{outputdir}/Correlation_{process}.pdf",
        transparent=True,
    )
    plt.clf()


def plotter_correlation_to_target(
    input_dataset,
    target_dataset,
    nfeatures,
    nparticles,
    feature_names,
    outputdir,
):

    # Loop through the input features and create scatterplots
    corr_matrix = [[0 for _ in range(nfeatures)] for _ in range(nparticles)]
    for i in range(nparticles):
        for j in range(nfeatures):
            cov = np.corrcoef(input_dataset[:, i, j].flatten(), target_dataset.flatten())
            corr_matrix[i][j] = cov[1, 0]

    plt.figure(figsize=(5, 5))
    boundary = (np.absolute(corr_matrix)).max() * 1.2
    plt.imshow(corr_matrix, cmap="jet", vmin=-boundary, vmax=boundary, origin="lower")
    cbar = plt.colorbar()
    cbar.set_label("Linear correlation to target")
    plt.title(f"FINAL STATE TRANSFORMER", loc="left")
    plt.ylabel("Paricle index")
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.tight_layout()
    plt.draw()
    plt.savefig(
        f"{outputdir}/correlations_to_target.pdf",
        transparent=True,
    )
    plt.clf()


def plotter_confusion_matrix(
    cm,
    x_classes,
    y_classes,
    normalize=True,
    title="Confusion matrix",
    outputdir=".",
    cmap=plt.cm.Blues,
):
    """
    Plot a confusion matrix.

    Parameters:
    - cm (numpy.ndarray): Confusion matrix.
    - x_classes (list): Labels for the x-axis.
    - y_classes (list): Labels for the y-axis.
    - normalize (bool): Whether to normalize the matrix.
    - title (str): Title of the plot.
    - outputdir (str): Directory to save the plot.
    - cmap: Colormap for the plot.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(x_classes))
    y_tick_marks = np.arange(len(y_classes))
    plt.xticks(x_tick_marks, x_classes, rotation=90)
    plt.yticks(y_tick_marks, y_classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    plt.savefig(
        outputdir,
        transparent=True,
    )
    plt.close()


def plotter_scores(
    cm,
    classes,
    normalize=True,
    title="Scores",
    plot=True,
    outputdir=".",
):
    """
    Plot classification scores based on a confusion matrix.

    Parameters:
    - cm (numpy.ndarray): Confusion matrix.
    - classes (list): Class labels.
    - normalize (bool): Whether to normalize the scores.
    - title (str): Title of the plot.
    - plot (bool): Whether to plot the scores.
    - outputdir (str): Directory to save the plot.

    Returns:
    - list: List of misclassification differences.
    """

    process = []

    accuracy = (
        []
    )  # Accuracy measures the proportion of correctly classified instances out of the total instances.
    precision = (
        []
    )  # Precision measures the proportion of true positive predictions out of all (true or false) positive predictions.
    recall = (
        []
    )  # Recall measures the proportion of true positive predictions out of all actual positive (true positive or false negatice) instances.
    f1_score = (
        []
    )  # F1 score is the harmonic mean of precision and recall, providing a balance between the two metrics.
    misclass_dif = (
        []
    )  # This metric represents the proportion of instances from class i that were misclassified as other classes.

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    for c in range(len(classes)):

        process.append(classes[c])

        val_accuracy = cm[c, c]
        val_precision = cm[c, c] / sum(cm[:, c])
        val_recall = cm[c, c] / sum(cm[c, :])
        val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        val_misclass_dif = 1 - val_recall

        accuracy.append(val_accuracy)
        precision.append(val_precision)
        recall.append(val_recall)
        f1_score.append(val_f1_score)
        misclass_dif.append(val_misclass_dif)

    if plot == True:
        score_dict = {
            "Accuracy": np.array(accuracy),
            "Precision": np.array(precision),
            "Recall": np.array(recall),
            "F1": np.array(f1_score),
            "Misclass. Dif.": np.array(misclass_dif),
        }

        x = np.arange(len(process))  # the label locations
        width = 0.15  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout="constrained")

        for attribute, measurement in score_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3, rotation=90, fmt="%.2f")
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Score")
        ax.set_xticks(x + width, np.array(process))
        ax.legend(loc="upper right", ncol=2, frameon=False)
        ax.set_ylim(0, 1.2)
        plt.text(
            0.12,
            0.92,
            title,
            fontsize=12,
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gcf().transFigure,
        )
        plt.savefig(outputdir, transparent=True)
        plt.close()

    return misclass_dif


def plotter_distributions_processes(
    inputs,
    weights,
    legend,
    colour,
    xlabel,
    logy,
    title,
    subtitle,
    outputdir,
):
    """
    Plot distributions of different processes.

    Parameters:
    - inputs (list): List of input data arrays.
    - weights (list): List of weights for each input.
    - legend (list): Legend labels.
    - colour (list): Colors for each process.
    - xlabel (str): Label for the x-axis.
    - logy (bool): Whether to use a logarithmic scale on the y-axis.
    - title (str): Title of the plot.
    - subtitle (str): Subtitle of the plot.
    - outputdir (str): Directory to save the plot.
    """
    dist_weight = HistogramPlot(
        n_ratio_panels=1,
        ylabel="Normalized number of events",
        xlabel=xlabel,
        logy=logy,
        norm=True,
        figsize=(5.5, 5),
        bins=np.linspace(
            np.min(np.concatenate(inputs)), np.max(np.concatenate(inputs)), 40
        ),
        y_scale=1.5,
        atlas_first_tag="FINAL STATE TRANSFORMER",
        atlas_second_tag="Validation",
        atlas_brand=None,
    )

    for y in range(len(inputs)):
        if y == 0:
            dist_weight.add(
                Histogram(
                    inputs[y],
                    weights=weights[y],
                    label=legend[y],
                    colour=colour[y],
                    linestyle="solid",
                ),
                reference=True,
            )
        else:
            dist_weight.add(
                Histogram(
                    inputs[y],
                    weights=weights[y],
                    label=legend[y],
                    colour=colour[y],
                    linestyle="solid",
                ),
            )

    dist_weight.draw()
    dist_weight.savefig(outputdir, transparent=True)


def plotter_distributions_signal_vs_background(
    inputs,
    weights,
    type,
    xlabel,
    logy,
    title,
    subtitle,
    outputdir,
):
    """
    Plot distributions of different processes.

    Parameters:
    - inputs (list): List of input data arrays.
    - weights (list): List of weights for each input.
    - type (list): List of process type.
    - xlabel (str): Label for the x-axis.
    - logy (bool): Whether to use a logarithmic scale on the y-axis.
    - title (str): Title of the plot.
    - subtitle (str): Subtitle of the plot.
    - outputdir (str): Directory to save the plot.
    """

    dist_weight = HistogramPlot(
        n_ratio_panels=1,
        ylabel="Normalized number of events",
        xlabel=xlabel,
        logy=logy,
        norm=True,
        figsize=(5.5, 5),
        bins=np.linspace(
            np.min(np.concatenate(inputs)), np.max(np.concatenate(inputs)), 40
        ),
        y_scale=1.5,
        atlas_first_tag="FINAL STATE TRANSFORMER",
        atlas_second_tag="Validation",
        atlas_brand=None,
    )

    sig = []
    bkg = []
    sig_weights = []
    bkg_weights = []

    for x in range(len(inputs)):
        if type[x] == "signal":
            sig.append(inputs[x])
            sig_weights.append(weights[x])
        elif type[x] == "background":
            bkg.append(inputs[x])
            bkg_weights.append(weights[x])

    dist_weight.add(
        Histogram(
            np.concatenate(bkg),
            weights=np.concatenate(bkg_weights),
            label="background",
            colour="crimson",
            linestyle="solid",
        ),
        reference=True,
    )

    dist_weight.add(
        Histogram(
            np.concatenate(sig),
            weights=np.concatenate(sig_weights),
            label="signal",
            colour="dodgerblue",
            linestyle="solid",
        ),
    )

    dist_weight.draw()
    dist_weight.savefig(outputdir, transparent=True)


def Plotter_ROC(
    inputs, 
    weights, 
    legend, 
    colour, 
    title, 
    keyname, 
    training_mode, 
    outputdir
):
    """
    Plot ROC Curves.

    Parameters:
    - inputs (list): List of input data arrays.
    - weights (list): List of weights for each input.
    - legend (list): Legend labels.
    - colour (list): Colors for each process.
    - xlabel (str): Label for the x-axis.
    - logy (bool): Whether to use a logarithmic scale on the y-axis.
    - title (str): Title of the plot.
    - keyname (str): name of reference process
    - training_mode (str): Training mode. 
    - outputdir (str): Directory to save the plot.
    """

    rejections = []
    efficiencies = []

    for i in range(len(inputs)):
        class_rejections = []
        class_efficiencies = []

        # Compute the histogram for signal events
        signal_counts, signal_bins = np.histogram(
            inputs[i],
            bins=500,
            range=(np.min(np.concatenate(inputs)), np.max(np.concatenate(inputs))),
            weights=weights[i],
        )

        sum_background_counts = None
        for j in range(len(inputs)):
            if i != j:

                # Compute the histogram for background events
                background_counts, background_bins = np.histogram(
                    inputs[j],
                    bins=500,
                    range=(
                        np.min(np.concatenate(inputs)),
                        np.max(np.concatenate(inputs)),
                    ),
                    weights=weights[j],
                )
                # include sum of all background
                if sum_background_counts is None:
                    sum_background_counts = background_counts
                else:
                    sum_background_counts += background_counts

                # Calculate cumulative sums
                signal_cumsum = np.cumsum(signal_counts[::-1])[::-1]
                background_cumsum = np.cumsum(background_counts[::-1])[::-1]

                # Normalize cumulative sums
                signal_efficiency = signal_cumsum / np.sum(signal_counts)
                background_rejection = 1.0 - (
                    background_cumsum / np.sum(background_counts)
                )

                # plot ROC curve
                plt.plot(
                    signal_efficiency,
                    background_rejection,
                    color=colour[j],
                    lw=2,
                    linestyle="--",
                    label=legend[j],
                )

        # include sum of all background
        sum_background_cumsum = np.cumsum(sum_background_counts[::-1])[::-1]
        sum_background_rejection = 1.0 - (
            sum_background_cumsum / np.sum(sum_background_counts)
        )
        plt.plot(
            signal_efficiency,
            sum_background_rejection,
            color="black",
            lw=2,
            label="Total background",
        )

        plt.text(
            0.15,
            0.85,
            title,
            fontsize=12,
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gcf().transFigure,
        )
        plt.text(
            0.15,
            0.80,
            "Validation",
            fontsize=10,
            horizontalalignment="left",
            verticalalignment="top",
            transform=plt.gcf().transFigure,
        )
        plt.plot([0, 1], [1, 0], color="gray", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.20])
        plt.xlabel(legend[i] + " efficiency")
        plt.ylabel("Background rejection")
        plt.tick_params(axis="x", direction="in", length=10)
        plt.tick_params(axis="y", direction="in", length=10)
        plt.legend()
        plt.grid(True)
        plt.savefig(
            outputdir + keyname[i] + "_" + training_mode + ".pdf",
            transparent=True,
        )
        plt.close()


def Plotter_Efficiency(
    inputs, 
    weights, 
    legend, 
    colour, 
    xlabel, 
    title, 
    keyname, 
    training_mode, 
    outputdir
):
    """
    Plot Efficiency Curves.

    Parameters:
    - inputs (list): List of input data arrays.
    - weights (list): List of weights for each input.
    - legend (list): Legend labels.
    - colour (list): Colors for each process.
    - xlabel (str): Label for the x-axis.
    - title (str): Title of the plot.
    - keyname (str): name of reference process
    - training_mode (str): Training mode. 
    - outputdir (str): Directory to save the plot.
    """

    for i in range(len(inputs)):

        # Compute the histogram for signal events
        signal_counts, signal_bins = np.histogram(
            inputs[i],
            bins=100,
            range=(np.min(np.concatenate(inputs)), np.max(np.concatenate(inputs))),
            weights=weights[i],
        )
        signal_cumsum = np.cumsum(signal_counts[::-1])[::-1]
        signal_efficiency = signal_cumsum / np.sum(signal_counts)

        plt.plot(
            signal_bins[:-1], signal_efficiency, color=colour[i], lw=2, label=legend[i]
        )

    plt.text(
        0.15,
        0.85,
        title,
        fontsize=12,
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gcf().transFigure,
    )
    plt.text(
        0.15,
        0.80,
        "Validation",
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gcf().transFigure,
    )
    plt.ylim([0.0, 1.20])
    plt.xlabel(xlabel)
    plt.ylabel("Efficiency")
    plt.tick_params(axis="x", direction="in", length=10)
    plt.tick_params(axis="y", direction="in", length=10)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        outputdir + training_mode + ".pdf",
        transparent=True,
    )
    plt.close()


def plotter_residuals(
    predictions_list,
    regression_target_list,
    weights_list,
    legend_list,
    colour_list,
    outputdir,
):
    residuals = HistogramPlot(
        n_ratio_panels=1,
        ylabel="Normalized number of events",
        xlabel="Residuals",
        logy=False,
        norm=True,
        figsize=(5, 5),
        bins=np.linspace(-2, 2, 50),
        y_scale=1.5,
        atlas_first_tag="FINAL STATE TRANSFORMER",
        atlas_second_tag="Regression",
        atlas_brand=None,
    )
    for y in range(len(predictions_list)):
        if y == 0:
            residuals.add(
                Histogram(
                    predictions_list[y].flatten() - regression_target_list[y],
                    weights=weights_list[y],
                    label=legend_list[y],
                    colour=colour_list[y],
                    linestyle="solid",
                ),
                reference=True,
            )
        else:
            residuals.add(
                Histogram(
                    predictions_list[y].flatten() - regression_target_list[y],
                    weights=weights_list[y],
                    label=legend_list[y],
                    colour=colour_list[y],
                    linestyle="solid",
                ),
            )
    # probabilities.add_bin_width_to_ylabel()
    residuals.draw()
    residuals.savefig(
        outputdir + "/validation/regression_residuals.pdf", transparent=False
    )


def plotter_regression_ratio(
    predictions_list,
    regression_target_list,
    weights_list,
    legend_list,
    colour_list,
    outputdir,
):
    ratio = HistogramPlot(
        n_ratio_panels=1,
        ylabel="Normalized number of events",
        xlabel="Predicted value / True value",
        logy=False,
        norm=True,
        figsize=(5, 5),
        bins=np.linspace(0, 2, 50),
        y_scale=1.5,
        atlas_first_tag="FINAL STATE TRANSFORMER",
        atlas_second_tag="Regression",
        atlas_brand=None,
    )
    for y in range(len(predictions_list)):
        if y == 0:
            ratio.add(
                Histogram(
                    predictions_list[y].flatten() / regression_target_list[y],
                    weights=weights_list[y],
                    label=legend_list[y],
                    colour=colour_list[y],
                    linestyle="solid",
                ),
                reference=True,
            )
        else:
            ratio.add(
                Histogram(
                    predictions_list[y].flatten() / regression_target_list[y],
                    weights=weights_list[y],
                    label=legend_list[y],
                    colour=colour_list[y],
                    linestyle="solid",
                ),
            )
    # probabilities.add_bin_width_to_ylabel()
    ratio.draw()
    ratio.savefig(outputdir + "/validation/regression_ratio.pdf", transparent=False)


def plotter_regression_prediction(
    predictions_list,
    regression_target_list,
    weights_list,
    xlabel,
    logy,
    legend_list,
    colour_list,
    outputdir,
):

    dist = HistogramPlot(
        n_ratio_panels=0,
        ylabel="Normalized number of events",
        xlabel=xlabel,
        logy=logy,
        norm=True,
        figsize=(6.5, 5),
        bins=np.linspace(np.min(predictions_list), np.max(predictions_list), 80),
        y_scale=1.5,
        atlas_first_tag="FINAL STATE TRANSFORMER",
        atlas_second_tag="Regression",
        atlas_brand=None,
    )

    for y in range(len(predictions_list)):
        dist.add(
            Histogram(
                regression_target_list[y],
                weights=weights_list[y],
                label=None,
                colour=colour_list[y],
                linestyle="--",
            ),
        )
        dist.add(
            Histogram(
                predictions_list[y].flatten(),
                weights=weights_list[y],
                label=legend_list[y],
                colour=colour_list[y],
                linestyle="solid",
            ),
        )

    # probabilities.add_bin_width_to_ylabel()
    dist.draw()
    dist.make_linestyle_legend(
        linestyles=["solid", "--"],
        labels=["Predicted value", "True value"],
        bbox_to_anchor=(0.55, 1),
    )
    dist.savefig(
        outputdir + "/validation/regression_distribution.pdf", transparent=False
    )

# Function to visualize the embedding space
def visualize_embedding_space(
    inputs,
    embeddings,
    legend,
    model_path,
):

    # Plot the embeddings in 3D space
    fig = plt.figure(figsize=(10, 5))

    # Plot 0 vs 1
    ax1 = fig.add_subplot(121)
    ax1.set_title("Input Space Visualization")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")

    # Plot 0 vs 2
    ax2 = fig.add_subplot(122)
    ax2.set_title("Embedding Space Visualization")
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")

    # Loop over processed
    for s in range(len(inputs)):

        # Perform PCA to reduce dimensionality
        tsne_inputs = TSNE(n_components=2, random_state=42)
        inputs_tsne = tsne_inputs.fit_transform(inputs[s])
        ax1.scatter(
            inputs_tsne[:, 0], inputs_tsne[:, 1], marker=".", alpha=0.5, label=legend[s]
        )

    # Loop over processed
    for s in range(len(embeddings)):

        # Perform PCA to reduce dimensionality
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_tsne = tsne.fit_transform(embeddings[s])
        ax2.scatter(
            embeddings_tsne[:, 0],
            embeddings_tsne[:, 1],
            marker=".",
            alpha=0.5,
            label=legend[s],
        )

    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    plt.savefig(model_path + "/validation/embedding_space_tsne.pdf", transparent=False)
    plt.close()
    
