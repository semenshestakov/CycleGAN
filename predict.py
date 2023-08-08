import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image

path_save = "data/result/"
path_load = "data/photo/"

model = tf.keras.models.load_model("models/generator_real_to_monet ep100")

df = pd.read_csv("data/data.csv")
df = df[df.path == "photo"]
k = 1
for name in df.name:
    ph = np.array(Image.open(f"{path_load}{name}")).astype(float).reshape(1, 256, 256, 3) / 255.0
    res = (model.predict(ph) * 255.0).reshape(256,256,3)
    image = Image.fromarray(res.astype(np.uint8), 'RGB')
    image.save(f"{path_save}{name}")
    print(f"....{k}" )
    k += 1