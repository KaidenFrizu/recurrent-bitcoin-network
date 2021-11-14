from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed


class Encoder(Layer):
    def __init__(self, k, units, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.input_layer = Input(shape=(None,k), dtype='float64')
        self.lstm_encoder = LSTM(units=units, return_state=True, unroll=True)

    def call(self, x):
        x = self.input_layer(x)
        x, state_h, state_c = self.lstm_encoder(x)

        return x, state_h, state_c


class Decoder(Layer):
    def __init__(self, H, units, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)
        self.lstm_decoder = LSTM(units=units, return_sequences=True)
        self.time_dist = TimeDistributed(Dense(H))

    def call(self, x, initial_state):
        x = self.lstm_decoder(x, initial_state=initial_state)
        x = self.time_dist(x)

        return x
