import random
import os
import string

import numpy as np

from .google_cloud_utils import append_to_spreadsheet

SPREADSHEET = "1BOFQdSvPN73Ofy7GhVTMxSVn7aEyIGNYHToTzoaMiv0"

def set_output_directory(output_directory):
    """Create all the necessary folders inside the output directory to save all
    data from current experiment.
    """
    # Generate a folder inside the output directory for the current
    # experiment
    experiment_tag = ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(20)
    )

    print "Create %r folder for current experiment" % (experiment_tag,)
    experiment_directory = os.path.join(output_directory, experiment_tag)
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Generate all necessary folders if they don't already exist
    if not os.path.exists(os.path.join(experiment_directory, "weights")):
        os.makedirs(os.path.join(experiment_directory, "weights"))
    if not os.path.exists(os.path.join(experiment_directory, "plots")):
        os.makedirs(os.path.join(experiment_directory, "plots"))

    return experiment_directory, experiment_tag


class Metrics(object):
    def __init__(self, train_data=None, validation_data=None):
        self.train_data = train_data
        self.validation_data = validation_data

    @staticmethod
    def _parse_from_file(filename):
        """ Given an input filename read the data from the file and generate a
        dictionary containing all data """
        with open(filename, "r") as f:
            lines = f.readlines()
        l = [x.strip().split(" ") for x in lines]

        # Create a dictionary that will keep the indices of each key in the
        # list l. The first list of the filename should be something like
        # `acc batch loss size` or `acc batch loss mae mde size`
        key_idxs = {}
        for li in l[0]:
            key_idxs[li] = l[0].index(li)

        # Dictionary that will hold the parsed data
        if "val_acc" in key_idxs.keys():
            data = {"val_acc": [], "val_loss": [], "val_mde": [], "val_mae": []}
        else:
            data = {"acc": [], "loss": [], "mde": [], "mae": []}
        for li in l[1:]:
            for k in data.keys():
                # Check if we have and index for that key
                if k in key_idxs.keys():
                    data[k].append(float(li[key_idxs[k]]))

        return data

    @classmethod
    def from_file(cls, training_file, validation_file):
        print training_file
        print validation_file

        train_data = cls._parse_from_file(training_file)
        validation_data = cls._parse_from_file(validation_file)

        return cls(train_data, validation_data)
        
    def to_list(self, n_samples=500):
        l = [None]*8

        key_idxs = {"loss": 0, "acc": 1, "mde": 2, "mae": 3}
        for k in self.train_data.keys():
            # Get the values for that key
            d = self.train_data[k]
            if len(d) > 0:
                l[key_idxs[k]] = "%.6f" % np.mean(d[:n_samples]) + " - " + "%.6f" % np.mean(d[-n_samples:])

        key_idxs = {"val_loss": 4, "val_acc": 5, "val_mde": 6, "val_mae": 7}
        for k in self.validation_data.keys():
            # Get the values for that key
            d = self.validation_data[k]
            if len(d) > 0:
                l[key_idxs[k]] = "%.6f" % d[0] + " - " + "%.6f" % d[-1]

        return [x for x in l if x is not None]


def register_experiment_with_data(data_file, credentials, sheet="Sheet1"):
    # Append results to the spreadsheet
    append_to_spreadsheet(
        SPREADSHEET, sheet,
        [data_file],
        credential_path=credentials
    )


def register_experiment(
    params_ordering,
    params,
    metrics,
    experiment_tag,
    credentials,
    sheet="Sheet1"
):
    # Generate a list with all the informations for the current experiment
    data = [experiment_tag]
    for i in params_ordering:
        data.append(params[i])
    data.extend(metrics)

    register_experiment_with_data(data, credentials, sheet=sheet)


def save_experiment_locally(
    params_ordering,
    params,
    metrics,
    experiment_tag,
    output_file
):
    # Generate a list with all the informations for the current experiment
    data = [experiment_tag]
    for i in params_ordering:
        data.append(params[i])
    data.extend(metrics)
    
    np.save(output_file, data)
