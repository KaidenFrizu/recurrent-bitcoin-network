from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed


class Encoder(Layer):
    def __init__(self, t, k, units, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.input_layer = Input(shape=(t, k), dtype='float64')
        self.lstm_encoder = LSTM(units=units, return_state=True)

    def call(self, x):
        x = self.input_layer(x)
        x = self.lstm_encoder(x)

        return x


class Decoder(Layer):
    def __init__(self, k, H, units, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)
        self.repeat_vector = RepeatVector(H)
        self.lstm_decoder = LSTM(units=units, return_sequences=True)
        self.resolve = Dense(k)
        self.time_dist = TimeDistributed(self.resolve)

    def call(self, x):
        x = self.repeat_vector(x)
        x = self.lstm_decoder(x)
        x = self.time_dist(x)

        return x
