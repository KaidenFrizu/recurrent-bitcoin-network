import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer
from tensorflow import convert_to_tensor

from typing import Optional

from . import layers

class RBN(Model):
    def __init__(
        self,
        t: int,
        k: int,
        H: int = 28,
        units: int = 50,
        encoder_kwargs: Optional[dict] = None,
        decoder_kwargs: Optional[dict] = None,
        **kwargs
        ):

        if encoder_kwargs is None:
            encoder_kwargs = {}

        if decoder_kwargs is None:
            decoder_kwargs = {}

        super(RBN, self).__init__(name='Recurrent-Bitcoin-Network', **kwargs)

        self.arr_decode = np.empty((1,H), dtype='float32')
        self.arr_decode = convert_to_tensor(self.arr_decode)

        self.inputlayer = InputLayer(input_shape=(t, k), dtype='float32',
                                     name='InputCheck')

        self.encoder = layers.Encoder(
            units=units,
            rnn_kwargs={'input_shape':(None,t,k)},
            **encoder_kwargs
        )

        self.decoder = layers.Decoder(
            H=H, units=units,
            **decoder_kwargs
        )

    def encode(self, x):
        x = self.inputlayer(x)

        return self.encoder(x)

    def decode(self, initial_state):

        return self.decoder(self.arr_decode, initial_state)

    def call(self, x):
        x, state_h, state_c = self.encode(x)

        return self.decode(initial_state=[state_h, state_c])
