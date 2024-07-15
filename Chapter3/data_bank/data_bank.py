# Description: This file contains the data loader selector function.

from .Data_loader_shades2 import Data_loader_shades2
from .Data_loader_existing import Data_loader_existing


def data_selector(data_name, arguments):
    """Select a data loader based on `data_name` (str).
    Arguments
    ---------
    data_name (str): Name of the data loader
    arguments (dict): Dictionary given to the constructor of the data loader

    Returns
    -------
    Data loader with name `data_name`. If not found, an error message is printed
    and it returns None.
    """
    if data_name.lower() == "existing":
        return Data_loader_existing(arguments)
    if data_name.lower() == "shades2":
        return Data_loader_shades2(arguments)
    else:
        print("Error: Could not find data loader with name %s" % (data_name))
        return None
