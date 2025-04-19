import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, GRU, Dense, Concatenate
from tensorflow.keras import backend as K

class DIALSTM_GRU(Layer):
    def __init__(self, lstm_units=128, gru_units=128, **kwargs):
        super(DIALSTM_GRU, self).__init__(**kwargs)
        self.lstm = LSTM(lstm_units, return_sequences=False, name="lstm_layer")
        self.gru = GRU(gru_units, return_sequences=False, name="gru_layer")
        self.concat = Concatenate()

    def build(self, input_shape):
        # Define attention weights if needed
        super(DIALSTM_GRU, self).build(input_shape)

    def call(self, inputs):
        # Optional: Attention mechanism can be added here if needed
        lstm_out = self.lstm(inputs)
        gru_out = self.gru(inputs)
        combined = self.concat([lstm_out, gru_out])
        return combined

    def get_config(self):
        config = super(DIALSTM_GRU, self).get_config()
        config.update({
            'lstm_units': self.lstm.units,
            'gru_units': self.gru.units
        })
        return config
