from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed

from typing import Optional


class Encoder(Layer):
    def __init__(
        self,
        units: int,
        rnn_kwargs: Optional[dict] = None,
        **kwargs
    ):

        if rnn_kwargs is None:
            rnn_kwargs = {}

        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.normalize = Normalization(axis=-1)
        self.lstm_encoder = LSTM(units=units, return_state=True,
                                 **rnn_kwargs)

    def call(self, x):
        x = self.normalize(x)
        return self.lstm_encoder(x)


class Decoder(Layer):
    def __init__(
        self,
        H: int,
        units: int,
        rnn_kwargs: Optional[dict] = None,
        **kwargs
    ):

        self.__MAX_HORIZON = H

        if rnn_kwargs is None:
            rnn_kwargs = {}

        super(Decoder, self).__init__(name='Decoder', **kwargs)

        self.repeat_vector = RepeatVector(self.__MAX_HORIZON)
        self.lstm_decoder = LSTM(units=units, return_sequences=True, 
                                 **rnn_kwargs)
        self.time_dist = TimeDistributed(Dense(1))

    def call(self, x, initial_state):
        x = self.repeat_vector(x)
        x = self.lstm_decoder(x, initial_state=initial_state)

        return self.time_dist(x)
