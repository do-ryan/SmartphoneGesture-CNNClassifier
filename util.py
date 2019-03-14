import json

def load_config(path):
    """ Loads the configuration file

     Args:
         path: A string indicating the path to the configuration file
     Returns:
         config: A Python dictionary of hyperparameter name-value pairs
         learning rate: The learning rate of the optimzer
         batch_size: Batch size used during training
         num_epochs: Number of epochs to train the network for
         target_classes: A list of strings denoting the classes to
                        build the classifer for
     """
    with open(path) as file:
        config = json.load(file)

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    eval_every = config["eval_every"]
    val_set_portion = config["val_set_portion"]

    return config, batch_size, learning_rate, num_epochs, eval_every, val_set_portion

def get_model_name(config):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_"
    path += "epoch{}_".format(config["num_epochs"])
    path += "bs{}_".format(config["batch_size"])
    path += "lr{}".format(config["learning_rate"])

    return path