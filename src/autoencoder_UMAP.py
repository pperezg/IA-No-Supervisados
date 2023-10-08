import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from importlib.machinery import SourceFileLoader
mp = SourceFileLoader("umap_", "venvIA/lib/python3.9/site-packages/umap/umap_.py").load_module()

class AutoEncoders(Model):

  def __init__(self, output_units):

    super().__init__()
    self.encoder = Sequential(
        [
          Dense(8, activation="relu"),
          Dense(10, activation="relu"),
          Dense(12, activation="relu")
        ]
    )

    self.decoder = Sequential(
        [
          Dense(10, activation="relu"),
          Dense(8, activation="relu"),
          Dense(output_units, activation="sigmoid")
        ]
    )

  def call(self, inputs):

    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded

def get_MLP_encoding(data):
    
  data = pd.DataFrame(data)

  auto_encoder = AutoEncoders(len(data.columns))

  auto_encoder.compile(loss='mae', metrics=['mae'], optimizer='adam')

  history = auto_encoder.fit(data, data, epochs=15, batch_size=32, verbose=0)
    
  encoder_layer = auto_encoder.get_layer('sequential')
  reduced_df = pd.DataFrame(encoder_layer.predict(data))
  reduced_df = reduced_df.add_prefix('feature_')

  return reduced_df

def umap2d3d(data, dims):
  reducer = mp.UMAP(n_components=dims)
  embedding = reducer.fit_transform(data)

  return embedding