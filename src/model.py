from tensorflow.keras import Model
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed

from typing import Optional


class RBN(Model):
    def __init__(
        self,
        name: str,
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

        super(RBN, self).__init__(name=name, **kwargs)

        # Encoder
        self.normalize = Normalization(axis=-1, input_shape=(t,k),
                                       name='Normalizer')
        self.lstm_encoder = LSTM(units=units, input_shape=(t,k),
                                 name='Encoder',
                                 unroll=True)

        # Decoder
        self.dense = Dense(H, name='Decoder')

    def call(self, x):
        # Encoding Phase
        x = self.normalize(x)
        x = self.lstm_encoder(x)

        # Decoding Phase
        return self.dense(x)
