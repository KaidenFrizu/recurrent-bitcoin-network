from tensorflow.keras import Model

from typing import Optional

from . import layers

class RBN(Model):
    def __init__(
        self,
        k: int,
        H: int = 14,
        units: int = 32,
        encoder_kwargs: Optional[dict] = None,
        decoder_kwargs: Optional[dict] = None,
        **kwargs
        ):

        if encoder_kwargs is None:
            encoder_kwargs = {}

        if decoder_kwargs is None:
            decoder_kwargs = {}

        super(RBN, self).__init__(name='Recurrent Bitcoin Network', **kwargs)

        self.encoder = layers.Encoder(k=k, units=units, **encoder_kwargs)
        self.decoder = layers.Decoder(k=k, H=H, units=units, **decoder_kwargs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, initial_state):
        return self.decoder(x, initial_state)

    def call(self, x):
        x, state_h, state_c = self.encode(x)
        x = self.decode(x, initial_state=[state_h, state_c])

        return x
