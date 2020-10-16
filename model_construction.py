from conv2d import Conv2d
from linear import Linear
from torch import nn


def construct_features(cfg,
                       padding,
                       module_names,
                       batch_norm=False,
                       slot_machine=False,
                       k=8,
                       greedy_selection=True
                       ):
    """
    Constructs convolutional layers for a neural network
    :param cfg: The configuration of the model. A list of the layers
    :param padding: The padding to apply to layers
    :param module_names: The module names
    :param batch_norm: (bool), Whether to apply batch normalization or not
    :param slot_machine: (bool), constructs a module for weight updates or slot_machines
    :param k: (int), The number of options per weight
    :param greedy_selection:(bool) the selection method for if model is slot machine
    :return: model: a sequential module of convolutional layers
    """
    model = nn.Sequential()
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            model.add_module(module_names[i], nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            if slot_machine:
                model.add_module(module_names[i],
                                 Conv2d(in_channels, v, k,
                                        kernel_size=3,
                                        padding=padding[i],
                                        greedy_selection=greedy_selection
                                        )
                                 )
            else:
                model.add_module(module_names[i],
                                 nn.Conv2d(in_channels, v,
                                           kernel_size=3,
                                           padding=padding[i],
                                           bias=False)
                                 )

            if batch_norm:
                model.add_module(f"bn_{i}", nn.BatchNorm2d(v, affine=False))

            model.add_module(f"relu_{i}", nn.ReLU(inplace=True))
            in_channels = v

    return model


def construct_classifier(cfg,
                         module_names,
                         in_features,
                         slot_machine=False,
                         k=8,
                         greedy_selection=True
                         ):
    """
    Constructs a sequential model of  fully-connected layers
    :param cfg:(List) The configuration of the model
    :param module_names: (List) The names of the layers
    :param in_features: (int) The number of input features to first fully-connected layer
    :param slot_machine: (bool)  constructs a module for weight updates or slot_machines
    :param k:(int), the number of options per weight if model is a slot machine
    :param greedy_selection: (bool), use greedy selection if model is slot machine
    :return: model: a sequential module of fully-connected layers
    """
    model = nn.Sequential()
    for i, v in enumerate(cfg):
        if v == 'D':
            model.add_module(module_names[i], nn.Dropout(p=0.5))
        elif v == "relu":
            model.add_module(module_names[i], nn.ReLU(inplace=True))
        else:
            if slot_machine:
                model.add_module(module_names[i],Linear(in_features, v, k, greedy_selection))
            else:
                model.add_module(module_names[i], nn.Linear(in_features, v, bias=False))
            in_features = v

    return model
