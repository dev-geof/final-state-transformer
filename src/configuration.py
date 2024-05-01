import os
import sys
import yaml
import logging


def configure_logging():
    """
    Configure the logging module.

    Parameters:
    None

    Returns:
    None
    """
    logging.basicConfig(level=logging.INFO, format="--- INFO --- %(message)s")


def load_config(configfile):
    """
    Load configuration from a YAML file.

    Parameters:
    - configfile (str): The path to the YAML configuration file.

    Returns:
    - dict: A dictionary containing the configuration.

    Raises:
    - Exception: If there is an error loading the configuration file.
    """
    try:
        with open(configfile, "r") as file:
            prime_service = yaml.safe_load(file)
        return prime_service
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


def create_directory(directory):
    """
    Create a directory if it doesn't exist.

    Parameters:
    - directory (str): The path of the directory to be created.
    """
    os.makedirs(directory, exist_ok=True)
