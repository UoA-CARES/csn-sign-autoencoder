import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code)
