import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

class DIALSTM_GRU(keras.Model):
    def __init__(self, n_features, timesteps, batch_size, n_commodities, embedding_dim=5, **kwargs):
        super(DIALSTM_GRU, self).__init__(**kwargs)
        self.n_features = n_features
        self.timesteps = timesteps
        self.batch_size = batch_size

        # Commodity Embedding Layer
        self.commodity_embedding = layers.Embedding(input_dim=n_commodities, output_dim=embedding_dim)

        # Feature Attention Layer
        self.feature_attention = layers.Attention()
        self.feature_dense_q = layers.Dense(n_features)
        self.feature_dense_k = layers.Dense(n_features)
        self.feature_dense_v = layers.Dense(n_features)

        # Temporal Attention Layer
        self.temporal_attention = layers.Attention()
        self.temporal_dense_q = layers.Dense(timesteps)
        self.temporal_dense_k = layers.Dense(timesteps)
        self.temporal_dense_v = layers.Dense(timesteps)

        # LSTM Layer
        self.lstm = layers.LSTM(6, activation='tanh', stateful=True, dropout=0.2, return_sequences=True, kernel_regularizer=l2(0.1))

        # GRU Layer
        self.gru = layers.GRU(6, activation='tanh', stateful=True, dropout=0.2, return_sequences=True, kernel_regularizer=l2(0.1))

        # Fully Connected Layers
        self.concat = layers.Concatenate(axis=-1)
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.2)
        self.fc1 = layers.Dense(10, activation="relu")
        self.fc2 = layers.Dense(1, activation='relu')

    def build(self, input_shape):
        super(DIALSTM_GRU, self).build(input_shape)

    def call(self, inputs):
        # Split the input: Features and Commodity Code
        features, commodity_code = inputs[:, :, :-1], inputs[:, 0, -1]

        # Embed Commodity Code
        commodity_embedding = self.commodity_embedding(tf.cast(commodity_code, tf.int32))

        # Feature Attention
        q_feature = self.feature_dense_q(inputs)
        k_feature = self.feature_dense_k(inputs)
        v_feature = self.feature_dense_v(inputs)
        feature_attention_output = self.feature_attention([q_feature, k_feature, v_feature])

        # Temporal Attention
        transposed_inputs = tf.transpose(inputs, perm=[0, 2, 1])
        q_temporal = self.temporal_dense_q(transposed_inputs)
        k_temporal = self.temporal_dense_k(transposed_inputs)
        v_temporal = self.temporal_dense_v(transposed_inputs)
        temporal_attention_output = self.temporal_attention([q_temporal, k_temporal, v_temporal])
        temporal_attention_output = tf.transpose(temporal_attention_output, perm=[0, 2, 1])

        # Combined Attention
        combined_attention = feature_attention_output * temporal_attention_output

        # Parallel LSTM and GRU
        lstm_output = self.lstm(combined_attention)
        gru_output = self.gru(combined_attention)

        # Concatenate LSTM and GRU outputs
        concatenated = self.concat([lstm_output, gru_output])

        # Flatten
        flattened = self.flatten(concatenated)

        # Concatenate with commodity embedding
        combined = self.concat([flattened, commodity_embedding])

        # Dense layers
        dropout = self.dropout(combined)
        fc1_output = self.fc1(dropout)
        output = self.fc2(fc1_output)
        return output

    def get_config(self):
        config = super(DIALSTM_GRU, self).get_config()
        config.update({
            'n_features': self.n_features,
            'timesteps': self.timesteps,
            'batch_size': self.batch_size
        })
        return config
