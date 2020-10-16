from model_construction import construct_features, construct_classifier

from torch import nn


class VGG(nn.Module):
    def __init__(self, features, classifier):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = classifier
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class VGG19:
    def __init__(self, num_classes=10, greedy_selection=True, batch_norm=True):
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                    512, 512, 512, 512, 'M', 512, 512, 512, 512]
        self.names = ["conv_1", "conv_2", "maxpool_1", "conv_3", "conv_4", "maxpool_2",
                      "conv_5", "conv_6", "conv_7", "conv_8", "maxpool_3",
                      "conv_9", "conv_10", "conv_11", "conv_12", "maxpool_4",
                      "conv_13", "conv_14", "conv_15", "conv_16",
                      ]
        self.padding = [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        self.classifier = [num_classes]
        self.classifier_names = ["fc1"]
        self.in_features = 512
        self._greedy_selection = greedy_selection
        self._batch_norm = batch_norm

    def weight_updates(self):
        """
        Construct a model to be trained in the normal way through weight updates
        :return: The model
        """
        features = construct_features(self.cfg,
                                      self.padding,
                                      self.names,
                                      batch_norm=self._batch_norm
                                      )
        classifier = construct_classifier(self.classifier,
                                          self.classifier_names,
                                          self.in_features
                                          )
        return VGG(features, classifier)

    def weight_selections(self, k=8):
        """
        Construct a slot machine
        :param k: The number of options per weight in the slot machine
        :return: The slot machine
        """
        features = construct_features(self.cfg,
                                      self.padding,
                                      self.names,
                                      batch_norm=self._batch_norm,
                                      selection=True, k=k,
                                      greedy_selection=self._greedy_selection)

        classifier = construct_classifier(self.classifier,
                                          self.classifier_names,
                                          self.in_features,
                                          selection=True,
                                          k=k,
                                          greedy_selection=self._greedy_selection
                                          )
        return VGG(features, classifier)
