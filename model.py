from shallow_conv_models import Conv2, Conv4, Conv6
from vgg19 import VGG19
from lenet import Lenet300100


def get_model(model_type, method, batch_norm=False, num_classes=10):
    """
    Instantiate a model object for training

    :param model_type: string, the name of the model
    :param method: string, the selection method being considered (greedy, prob, learned)
    :param batch_norm: bool, indicate whether to use batch normalization or not
    :param num_classes: int, the number of classes

    :return: A neural network for optimization
    """
    if model_type == "conv2":
        model = Conv2
    elif model_type == "conv4":
        model = Conv4
    elif model_type == "conv6":
        model = Conv6
    elif model_type == "vgg19":
        model = VGG19
    elif model_type == "lenet":
        model = Lenet300100
    else:
        raise ValueError(f"unknown type {model_type}")

    if method == "greedy":
        greedy_selection = True
    else:
        greedy_selection = False

    return model(num_classes, greedy_selection, batch_norm)