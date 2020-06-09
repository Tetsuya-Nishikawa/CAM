import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf


image = Image.open("./cat.jpg")
data  = np.asarray(image).astype(np.float32)


#構築済みモデルをロードする。
checkpoint_path = "./chckpnt"
latest = tf.train.latest_checkpoint(checkpoint_path)
model = Model()
model.load_weights(latest)

model(np.reshape(data, [1, 200, 220, 3]))#おまじない

#途中マップを取得する
image = model.get_map(np.reshape(data, [1, 200, 220, 3])).numpy()

#全結合層の重みを入手する
weights = np.array(model.layers[-1].get_weights()[0][:,1])
h = 200/(image.shape[1])
w = 220/(image.shape[2])

#マップをズームする
zoomed_data = sp.ndimage.zoom(np.reshape(image, [image.shape[1], image.shape[2],256]), (h, w, 1), order=3)

#ヒートマップを表示する
heatmap = weights[0]*zoomed_data[:,:,0]
for idx in range(1, 256):
    heatmap = heatmap + weights[idx]*zoomed_data[:,:,idx]

plt.imshow(np.reshape(data/255.0, [200, 220, 3]), alpha=0.5)
plt.imshow(np.reshape(heatmap, [200, 220]), cmap="jet", alpha=0.5)
plt.show()