from .Data_loader import Data_loader
import numpy as np
import random


class Data_loader_existing(Data_loader):
    """
    load data from a saved file


    Attributes
    ----------
    images (string): The name of the file of images to read from
    labels (string): The name of the file of labels to read from
    difference (string): The name of the file of differences to read from
    epsilons (string): The name of the file of epsilons to read from
    model (int): The model number

    Methods
    -------
    load_data(): Loads the training data.

    Example arguments
    -----------------

    images: "/mn/kadingir/vegardantun_000000/nobackup/NN_security_data/data/shades_train_images"
    labels: "/mn/kadingir/vegardantun_000000/nobackup/NN_security_data/data/shades_train_labels"
    difference: "/mn/kadingir/vegardantun_000000/nobackup/NN_security_data/data/shades_train_difference"
    epsilons: "/mn/kadingir/vegardantun_000000/nobackup/NN_security_data/data/shades_train_epsilon"
    model: 1

    """

    def __init__(self, arguments):
        super(Data_loader_existing, self).__init__(arguments)
        required_keys = ["images", "labels", "difference", "epsilons", "model"]

        self._check_for_valid_arguments(required_keys, arguments)
        self.images = arguments["images"]
        self.labels = arguments["labels"]
        self.difference = arguments["difference"]
        self.epsilons = arguments["epsilons"]
        self.model = arguments["model"]

    def load_data(self):

        data, label, diff, epsilons = self._generate_set()

        return data, label, diff, epsilons

    def __str__(self):
        class_str = """ Existing data

Images : %s
Labels : %s
Difference: %s
Epsilons: %s
Model Number: %f

""" % (
            self.images,
            self.labels,
            self.difference,
            self.epsilons,
            self.model,
        )

        return class_str

    def _generate_set(self, shuffle=True):

        images = self.images
        labels = self.labels
        diff = self.difference
        epsilons = self.epsilons
        model = self.model

        image_link = images + str(model) + ".npy"
        labels_link = labels + str(model) + ".npy"
        diff_link = diff + str(model) + ".npy"
        epsilon_link = epsilons + str(model) + ".npy"

        data = np.load(image_link)
        labels_data = np.load(labels_link)
        diff_data = np.load(diff_link)
        epsilon_data = np.load(epsilon_link)

        return data, labels_data, diff_data, epsilon_data
