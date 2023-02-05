import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, cls_head, depth_head):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.cls_head = cls_head
        self.depth_head = depth_head
        
    def forward(self, x):
        x = self.encoder(x)
        depth = self.depth_head(x)
        cls_score = self.cls_head(x[4])
        return cls_score, depth
