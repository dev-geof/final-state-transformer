import os
import numpy as np
import h5py
import yaml
from termcolor import colored
import logging
import tensorflow as tf


def weights_extraction(
    filepath: str,
    dataset: str,
    weights: str,
    nevents: int,
    isvalidation: bool,
) -> np.ndarray:
    """
    Extracts weights from event dataset in HDF5 files.

    Parameters:
        filepath (str): Path(s) to the HDF5 file(s), separated by colon if multiple.
        dataset (str): Name of the dataset to extract weights from.
        weights (str): Name of the weights to extract.
        nevents (int): Number of events to extract from each file.
        isvalidation (bool): If True, reduces the number of events by 10 for validation.

    Returns:
        np.ndarray: Concatenated array of weights from all specified files.

    """

    if isvalidation == True:
        nevents = int(nevents / 10)

    filepath_list = filepath.split(":")
    df = []

    for f in range(len(filepath_list)):
        # read h5 file
        with h5py.File(filepath_list[f], "r") as h5file:
            df.append(h5file.get(dataset)[:nevents][weights])

    df_full = np.concatenate(df, axis=0)

    return df_full


def ghost_extraction(
    filepath: str,
    dataset: str,
    nevents: int,
    fields_of_interest: str,
    isvalidation: bool,
) -> np.ndarray:
    """
    Extracts ghost data of interest from event dataset in HDF5 files. Only for validation purpose.

    Parameters:
        filepath (str): Path(s) to the HDF5 file(s), separated by colon if multiple.
        dataset (str): Name of the dataset to extract data from.
        nevents (int): Number of events to extract from each file.
        fields_of_interest (str): Name of the fields of interest to extract.
        isvalidation (bool): If True, reduces the number of events by 10 for validation.

    Returns:
        np.ndarray: Concatenated array of data of interest from all specified files.

    """
    if isvalidation == True:
        nevents = int(nevents / 10)

    filepath_list = filepath.split(":")
    df = []

    for f in range(len(filepath_list)):
        # read h5 file
        with h5py.File(filepath_list[f], "r") as h5file:
            # declare list of var subdataset
            var_list = []
            # loop over event dataset
            for x in range(len(fields_of_interest)):
                # print(x,fields_of_interest[x])
                var = h5file.get(dataset)[:nevents][fields_of_interest[x]]
                var_list.append(var)

    return np.stack(var_list, axis=1)


def data_extraction_transformer(
    filepath: str,
    dataset: str,
    nevents: int,
    fields_of_interest: list,
    nparticles: int,
    isvalidation: bool,
) -> np.ndarray:
    """
    Extract a selection of features of interest from the particle dataset

    Parameters:
        filepath (str): Path(s) to the HDF5 file(s), separated by colon if multiple.
        dataset (str): Name of the dataset to extract data from.
        nevents (int): Number of events to extract from each file.
        fields_of_interest (list): List of field names to extract from the dataset.
        nparticles (int): Number of particles to consider per event.
        isvalidation (bool): If True, reduces the number of events by 10 for validation.

    Returns:
        np.ndarray: Concatenated array of transformed data from all specified files.

    """

    if isvalidation == True:
        nevents = int(nevents / 10)

    filepath_list = filepath.split(":")
    df = []

    for f in range(len(filepath_list)):
        # read h5 file
        print("Processed sample: ", filepath_list[f])
        with h5py.File(filepath_list[f], "r") as h5file:
            # declare list of var subdataset
            var_list = []
            # loop over event dataset
            for x in range(len(fields_of_interest)):
                # print(x,fields_of_interest[x])
                var = h5file.get(dataset)[:nevents, :nparticles][fields_of_interest[x]]
                var_list.append(var)

            # Satck all individual variable container and assign dtype
            df.append(np.dstack(var_list))

    return np.concatenate(df, axis=0)


def get_nfeatures(configfile: str) -> int:

    """
    Extract number of track feature

    Parameters
    ----------
    configfile : str
        config. file

    Returns
    -------
    int
        Number of features
    """
    with open(configfile, "r") as file:
        prime_service = yaml.safe_load(file)
        fields = prime_service["input_variables"]

    return len(fields)


def apply_sample_normalization(
    weights_list,
    norm,
):
    """
    Apply sample normalization to a list of weights.

    Parameters:
    - weights_list (list): A list of weights to be normalized.
    - norm (bool): A flag indicating whether normalization should be applied.

    Returns:
    - list: The normalized list of weights.

    If the `norm` parameter is True, the function applies sample overall normalization
    by dividing each element in the `weights_list` by the sum of its elements.
    """
    if norm:
        logging.info(colored("Applying sample overall normalization", "yellow"))
        weights_list = [item * 1.0 / sum(item) for item in weights_list]
    return weights_list


def get_training_datasets(prime_service, mode, datasetdir):
    """
    Get training and test datasets.

    Parameters:
    - prime_service (dict): Dictionary containing configuration parameters.
    - mode (str): The training mode ("classification" or "regression").
    - outputdir (str): Output directory for datasets.

    Returns:
    - tf_dataset_train: Training dataset.
    - tf_dataset_test: Test dataset.
    """
    logging.debug(colored(f"Opening {mode} dataset located in {datasetdir}", "yellow"))
    tf_dataset = tf.data.Dataset.load(datasetdir)

    # Splitting input dataset into training (50%) and test (50%) dataset
    tf_dataset_train = tf_dataset.shard(num_shards=2, index=0)
    tf_dataset_test = tf_dataset.shard(num_shards=2, index=1)

    return tf_dataset_train, tf_dataset_test


def building_new(var, label, first_array, nevents):
    """
    Build a new NumPy array by adding a new variable with the given label.

    Parameters:
    - var: The variable to be added to the new array.
    - label (str): The label for the new variable.
    - first_array (numpy.ndarray): The original array from which to extract existing data.
    - nevents (int): The number of events in the new array.

    Returns:
    - numpy.ndarray: A new array with the specified variable added.

    This function creates a new NumPy array with the same structure as `first_array` but adds
    a new variable with the specified `label` and values from the `var` parameter. The number of
    events in the new array is determined by the `nevents` parameter.
    """
    new_array = np.zeros(nevents, dtype=(first_array.dtype.descr + [(label, "<f4")]))
    existing_keys = list(first_array.dtype.fields.keys())
    new_array[existing_keys] = first_array[existing_keys]
    new_array[label] = var

    return new_array
