import numpy as np
import pandas as pd
from typing import Optional

from utils import create_ts_batch
from sklearn.decomposition import TruncatedSVD

from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

class RBN(Model):

    def __init__(
        self,
        model_name: str,
        t: int,
        k: Optional[int] = None,
        H: int = 7,
        encoder_kwargs: Optional[dict] = None,
        decoder_kwargs: Optional[dict] = None,
        **kwargs
    ):

        if encoder_kwargs is None:
            encoder_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        self.model_name = model_name
        self.t = t
        self.k = k
        self.H = H

        super(RBN, self).__init__(name=self.model_name, **kwargs)

        # Encoder
        self.input_layer = InputLayer(input_shape=(self.t,self.k),
                                      name='Past_Inputs')
        self.normalize = Normalization(axis=-1,
                                       name='Normalizer')
        self.lstm_encoder = LSTM(units=125,
                                 name='Encoder',
                                 activation=LeakyReLU(alpha=0.01),
                                 return_state=True,
                                 **encoder_kwargs)

        # Decoder
        self.future_layer = InputLayer(input_shape=(self.t, 5), # date features
                                       name='Future_Inputs')
        self.lstm_decoder = LSTM(units=125,
                                 name='Decoder',
                                 activation=LeakyReLU(alpha=0.01),
                                 return_sequences=True,
                                 **decoder_kwargs)
        self.resolve = TimeDistributed(Dense(1))

    def _encode(self, x):
        x = self.input_layer(x)
        x = self.normalize(x)
        return self.lstm_encoder(x)

    def _decode(self, x, state_h, state_c):
        x = self.future_layer(x)
        x = self.lstm_decoder(x, initial_state=[state_h, state_c])
        return self.resolve(x)

    def call(self, x):
        past_x, future_x = x
        # Encoding Phase
        past_x, state_h, state_c = self._encode(past_x)

        # Decoding Phase
        return self._decode(future_x, state_h, state_c)

class RNNmodel:

    def __init__(
        self,
        t: int,
        k: Optional[int] = None,
        H: Optional[int] = 7,
        model_name: Optional[str] = None
    ):
        self.t = t
        self.k = k
        self.H = H
        self.model_name = model_name
        self.RNNmodel = RBN(
            model_name=self.model_name,
            t=self.t,
            k=self.k,
            H=self.H
        )
        self.SVDmodel = TruncatedSVD(n_components=self.k)

        self.metric = rmse(name='rmse')
        self.optimizer = Adam(learning_rate = 0.001, name='Adam')

        self.logdirectory = '../logs/fit/' + self.model_name
        self.callback = TensorBoard(
            log_dir=self.logdirectory, histogram_freq=1
        )

    def initialize(self, train, test):
        train_x = train.pivot_table(
            index='timestamp',
            columns=['metric', 'submetric'],
            values='value'
        )
        train_y = train_x.pop('price')
        train_y = train_y['close']

        test_x = test.pivot_table(
            index='timestamp',
            columns=['metric', 'submetric'],
            values='value'
        )
        test_y = test_x.pop('price')
        test_y = test_y['close']

        selected_features = ['act.addr.cnt','blk.size.byte',
                             'blk.size.bytes.avg','daily.shp',
                             'daily.vol','exch.flow.in.usd.incl','exch.sply',
                             'hashrate','mcap.dom','mcap.realized',
                             'nvt.adj.90d.ma','real.vol',
                             'txn.cnt','txn.tsfr.cnt','txn.vol']

        self.train_x = train_x[selected_features].values
        self.test_x = test_x[selected_features].values

        self.train_y = train_y
        self.test_y = test_y

        # Train Dates
        dates_past = self.train_y.index
        date_quarter_past = dates_past.quarter

        dow_cos_past = np.cos(dates_past.weekday * 2 * np.pi / 6)
        dow_sin_past = np.sin(dates_past.weekday * 2 * np.pi / 6)
        doy_cos_past = np.cos(dates_past.dayofyear * 2 * np.pi / 6)
        doy_sin_past = np.sin(dates_past.dayofyear * 2 * np.pi / 6)

        self.future_traindata = pd.DataFrame(
            {
                'Quarter':date_quarter_past,'DayOfWeek_cos':dow_cos_past,
                'DayOfWeek_sin':dow_sin_past,'DayOfYear_cos':doy_cos_past,
                'DayOfYear_sin':doy_sin_past
            }
        )

        # Test Dates
        dates = self.test_y.index
        date_quarter = dates.quarter

        dow_cos = np.cos(dates.weekday * 2 * np.pi / 6)
        dow_sin = np.sin(dates.weekday * 2 * np.pi / 6)
        doy_cos = np.cos(dates.dayofyear * 2 * np.pi / 365)
        doy_sin = np.sin(dates.dayofyear * 2 * np.pi / 365)

        self.future_testdata = pd.DataFrame(
            {
                'Quarter':date_quarter, 'DayOfWeek_cos':dow_cos,
                'DayOfWeek_sin':dow_sin, 'DayOfYear_cos':doy_cos,
                'DayOfYear_sin':doy_sin
            }
        )

        if self.train_x.shape[1] < self.k and self.test_x.shape[1] < self.k:
            self.train_x = self.SVDmodel.fit_transform(train_x)
            self.test_x = self.SVDmodel.fit_transform(test_x)

        self.xtrain, self.ytrain, self.trainplotdata = create_ts_batch(
            self.train_x, self.train_y.values,
            self.future_traindata.values, self.t, self.H
        )

        self.xtest, self.ytest, self.testplotdata = create_ts_batch(
            self.test_x, self.test_y.values,
            self.future_testdata.values, self.t, self.H
        )

        self.RNNmodel.normalize.adapt(self.xtrain[0])
        self.RNNmodel.compile(
            optimizer=self.optimizer,
            loss='mse',
            metrics=[self.metric]
        )

        return None

    def train(self, batch_size, epochs, **kwargs):
        self.RNNmodel.fit(
            self.xtrain,
            self.ytrain, batch_size, epochs,
            callbacks=[self.callback], validation_split=0.2,
            verbose=0,
            **kwargs
        )

        return None

    def evaluate(self, batch_size):
        self.RNNmodel.evaluate(
            self.xtest,
            self.ytest, batch_size,
            callbacks=[self.callback]
        )

        return None

    def predict(self, data):
        return self.RNNmodel.predict(data)
