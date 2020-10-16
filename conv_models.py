from model_construction import construct_features, construct_classifier

from torch import nn


class VGGLike(nn.Module):
    def __init__(self, features, classifier):
        super(VGGLike, self).__init__()
        self.features = features
        self.classifier = classifier
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class ConvWrapper:
    def __init__(self,
                 features_cfg,
                 padding,
                 feature_names,
                 in_features,
                 num_classes=10,
                 greedy_selection=True,
                 batch_norm=False
                 ):
        self._features_cfg = features_cfg
        self._padding = padding
        self._features_names = feature_names
        self._classifier_cfg = ["D", 256, "relu", "D", 256, "relu", num_classes]
        self._classifier_names = ["dropout_1", "fc1", "relu_1", "dropout2", "fc2", "relu_2", "fc3"]
        self._in_features = in_features
        self._greedy_selection = greedy_selection
        self._batch_norm = batch_norm

    def weight_updates(self):
        features = construct_features(self._features_cfg,
                                      self._padding,
                                      self._features_names,
                                      batch_norm=self._batch_norm
                                      )
        classifier = construct_classifier(self._classifier_cfg,
                                          self._classifier_names,
                                          self._in_features
                                          )
        return VGGLike(features, classifier)

    def weight_selections(self, k=8):
        features = construct_features(self._features_cfg,
                                      self._padding,
                                      self._features_names,
                                      batch_norm=self._batch_norm,
                                      slot_machine=True,
                                      k=k,
                                      greedy_selection=self._greedy_selection
                                      )
        classifier = construct_classifier(self._classifier_cfg,
                                          self._classifier_names,
                                          self._in_features,
                                          slot_machine=True,
                                          k=k,
                                          greedy_selection=self._greedy_selection
                                          )

        return VGGLike(features, classifier)


class Conv2(ConvWrapper):
    def __init__(self, num_classes=10, greedy_selection=True, batch_norm=False):
        super(Conv2, self).__init__(features_cfg=[64, 64, "M"],
                                    padding=[0, 0, 0],
                                    feature_names=["conv_1", "conv_2", "maxpool_1"],
                                    in_features=12544,
                                    num_classes=num_classes,
                                    greedy_selection=greedy_selection,
                                    batch_norm=batch_norm
                                    )


class Conv4(ConvWrapper):
    def __init__(self, num_classes=10, greedy_selection=True, batch_norm=False):
        super(Conv4, self).__init__(features_cfg=[64, 64, "M", 128, 128, "M"],
                                    padding=[0, 0, 0, 0, 0, 0],
                                    feature_names=["conv_1", "conv_2", "maxpool_1", "conv_3", "conv_4", "maxpool_2"],
                                    in_features=3200,
                                    num_classes=num_classes,
                                    greedy_selection=greedy_selection,
                                    batch_norm=batch_norm
                                    )


class Conv6(ConvWrapper):
    def __init__(self, num_classes=10, greedy_selection=True, batch_norm=False):
        super(Conv6, self).__init__(features_cfg=[64, 64, "M", 128, 128, "M", 256, 256, "M"],
                                    padding=[0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    feature_names=["conv_1", "conv_2", "maxpool_1",
                                                   "conv_3", "conv_4", "maxpool_2",
                                                   "conv_5", "conv_6", "maxpool_3"],
                                    in_features=1024,
                                    num_classes=num_classes,
                                    greedy_selection=greedy_selection,
                                    batch_norm=batch_norm
                                    )
