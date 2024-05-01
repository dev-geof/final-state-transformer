import os
import tensorflow as tf
import matplotlib.pyplot as plt


class TrainingPlot(tf.keras.callbacks.Callback):
    def __init__(self, outputdir):
        """
        Callback for plotting training metrics during training.

        Parameters:
        - outputdir (str): Output directory for saving the training plots.

        Returns:
        None
        """
        self.outputdir = outputdir

    def on_train_begin(self, logs={}):

        """
        Called at the beginning of training.

        Parameters:
        - logs (dict): Dictionary containing the training metrics.

        Returns:
        None
        """
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
        self.aucs = []
        self.val_aucs = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Called at the end of each epoch.

        Parameters:
        - epoch (int): The current epoch.
        - logs (dict): Dictionary containing the training metrics.

        Returns:
        None
        """
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.accs.append(logs.get("accuracy"))
        self.val_accs.append(logs.get("val_accuracy"))
        self.aucs.append(logs.get("auc"))
        self.val_aucs.append(logs.get("val_auc"))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

            # Plot Training and Validation Loss
            ax1.plot(self.losses, label="Training Loss")
            ax1.plot(self.val_losses, label="Validation Loss")
            ax1.tick_params(axis="x", direction="in")
            ax1.tick_params(axis="y", direction="in")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend(frameon=False)
            ax1.set_title("Loss")
            ax1.legend()

            # Plot Training and Validation Accuracy
            ax2.plot(self.accs, label="Training Accuracy")
            ax2.plot(self.val_accs, label="Validation Accuracy")
            ax2.tick_params(axis="x", direction="in")
            ax2.tick_params(axis="y", direction="in")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend(frameon=False)
            ax2.set_title("Accuracy")
            ax2.legend()

            # Plot Training and Validation AUC
            ax3.plot(self.aucs, label="Training AUC")
            ax3.plot(self.val_aucs, label="Validation AUC")
            ax3.tick_params(axis="x", direction="in")
            ax3.tick_params(axis="y", direction="in")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("AUC")
            ax3.legend(frameon=False)
            ax3.set_title("AUC")
            ax3.legend()

            plt.savefig(os.path.join(self.outputdir, "training_plot.pdf"))
            plt.close()


class TrainingPlot_Regression(tf.keras.callbacks.Callback):
    def __init__(self, outputdir):
        """
        Callback for plotting training metrics during regression training.

        Parameters:
        - outputdir (str): Output directory for saving the training plots.

        Returns:
        None
        """
        self.outputdir = outputdir

    def on_train_begin(self, logs={}):
        """
        Called at the beginning of training.

        Parameters:
        - logs (dict): Dictionary containing the training metrics.

        Returns:
        None
        """
        self.losses = []
        self.val_losses = []
        self.metrics = []
        self.val_metrics = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Called at the end of each epoch.

        Parameters:
        - epoch (int): The current epoch.
        - logs (dict): Dictionary containing the training metrics.

        Returns:
        None
        """
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.metrics.append(
            logs.get("mean_squared_error")
        )  # Update with your desired metric
        self.val_metrics.append(
            logs.get("val_mean_squared_error")
        )  # Update with your desired metric

        if len(self.losses) > 1:

            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

            # Plot Training and Validation Loss
            ax1.plot(self.losses, label="Training Loss")
            ax1.plot(self.val_losses, label="Validation Loss")
            ax1.tick_params(axis="x", direction="in")
            ax1.tick_params(axis="y", direction="in")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend(frameon=False)
            ax1.set_title("Loss")
            ax1.legend()

            # Plot Training and Validation Metric (Mean Squared Error)
            ax2.plot(
                self.metrics, label="Training Metric"
            )  # Update with your desired metric
            ax2.plot(
                self.val_metrics, label="Validation Metric"
            )  # Update with your desired metric
            ax2.tick_params(axis="x", direction="in")
            ax2.tick_params(axis="y", direction="in")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Mean squared error")
            ax2.legend(frameon=False)
            ax2.set_title("Mean squared error")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.outputdir, "training_plot.pdf"))
            plt.close()
