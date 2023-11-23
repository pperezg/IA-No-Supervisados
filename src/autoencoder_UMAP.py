# Importación de las librerías necesarias para este file
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from importlib.machinery import SourceFileLoader
mp = SourceFileLoader("umap_", "venvIA/lib/python3.9/site-packages/umap/umap_.py").load_module()

#Clase autoencoder
class AutoEncoders(Model):

  def __init__(self, output_units, dims):

    super().__init__()
    self.encoder = Sequential( #Codificar
        [
          Dense(dims, activation="relu"),
          Dense(dims, activation="relu"),
          Dense(dims+1, activation="relu")
        ]
    )

    self.decoder = Sequential( #Decodificar
        [
          Dense(dims, activation="relu"),
          Dense(dims, activation="relu"),
          Dense(output_units, activation="sigmoid")
        ]
    )

  def call(self, inputs): #Llama al autoencoder

    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded
    
'''
Función: get_MPL_encoding

Descripción: Crea el autoencoder y codifica los datos

Parametros: data (array de numpy), dims (int)

Retorna: reduced_df (dataframe de pandas)
'''
def get_MLP_encoding(data, dims):
    
  data = pd.DataFrame(data)

  auto_encoder = AutoEncoders(len(data.columns), dims)

  auto_encoder.compile(loss='mae', metrics=['mae'], optimizer='adam')

  history = auto_encoder.fit(data, data, epochs=15, batch_size=32, verbose=0)
    
  encoder_layer = auto_encoder.get_layer('sequential')
  reduced_df = pd.DataFrame(encoder_layer.predict(data))
  reduced_df = reduced_df.add_prefix('feature_')

  return reduced_df

'''
Función: umap2d3d

Descripción: Aplica el algoritmo UMAP a los datos para dejarlos en 2d

Parametros: data (array de numpy), dims (int)

Retorna: embedding (dataframe de pandas)
'''
def umap2d3d(data, dims):
  reducer = mp.UMAP(n_components=dims)
  embedding = reducer.fit_transform(data)

  return embedding
