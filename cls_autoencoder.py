import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, cls_head, reconstruct_head):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.cls_head = cls_head
        self.reconstruct_head = reconstruct_head

    def forward(self, x):
        x = self.encoder(x)
        reconstructed = self.reconstruct_head(x)
        cls_score = self.cls_head(x[4])
        return cls_score, reconstructed
