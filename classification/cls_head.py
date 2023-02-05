import torch.nn as nn


class ClassifierHead(nn.Module):
    """Classification head for CSN.
    Args:
        num_classes (int): Number of classes to be classified.
        in_features(int): Number of channels in input feature.
        init_std (float): Std value for Initiation. Default: 0.01.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
    """

    def __init__(self, in_features=2048, num_classes=400, init_std=0.01, dropout_ratio=0.5):
        super(ClassifierHead, self).__init__()
        self.init_std = init_std
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(in_features, num_classes)

    def normal_init(self, module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        self.normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
