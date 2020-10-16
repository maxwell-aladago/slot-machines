from model_construction import construct_classifier

from torch import nn


class LenetBase(nn.Module):
    def __init__(self, classifier):
        super(LenetBase, self).__init__()
        self.classifier = classifier
        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class Lenet300100:
    def __init__(self,
                 num_classes=10, greedy_selection=True, batch_norm=False):
        self._classifier_cfg = [300, "relu", 100, "relu", num_classes]
        self._classifier_names = ["fc1", "relu_1", "fc2", "relu_2", "fc3"]
        self._in_features = 784
        self._greedy_selection = greedy_selection
        self._batch_norm = batch_norm

    def weight_updates(self):
        classifier = construct_classifier(self._classifier_cfg,
                                          self._classifier_names,
                                          self._in_features
                                          )
        return LenetBase(classifier)

    def weight_selections(self, k=8):
        classifier = construct_classifier(self._classifier_cfg,
                                          self._classifier_names,
                                          self._in_features,
                                          slot_machine=True,
                                          k=k,
                                          greedy_selection=self._greedy_selection
                                          )

        return LenetBase(classifier)
