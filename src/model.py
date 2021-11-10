from tensorflow.keras import Model

from .src.layers import Encoder
from .src.layers import Decoder

class RBN(Model):
    def __init__(self, t, k, H, units=32, **kwargs):
        super(RBN, self).__init__(name='Recurrent Bitcoin Network', **kwargs)

        self.encoder = Encoder(t=t, k=k, units=units)
        self.decoder = Decoder(k=k, H=H, units=units)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, initial_state):
        return self.decoder(x, initial_state)

    def call(self, x, training=True):
        x = self.encode(x)
        x = self.decode(x[0], x[1:])

        return x
