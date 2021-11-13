from tensorflow.keras import Model

from . import layers

class RBN(Model):
    def __init__(self, t, k, H, units=32, **kwargs):
        super(RBN, self).__init__(name='Recurrent Bitcoin Network', **kwargs)

        self.encoder = layers.Encoder(t=t, k=k, units=units)
        self.decoder = layers.Decoder(k=k, H=H, units=units)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def call(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x
