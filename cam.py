import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import seaborn as sns

%matplotlib inline

tf.random.set_seed(
    1234
)

def tensor_cast(inputs, labels):
    inputs = tf.cast(inputs, tf.float32)
    return inputs, tf.cast(labels, tf.int64)

class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = tf.keras.layers.BatchNormalization(input_shape=(200, 220))
        self.layer2 = tf.keras.layers.Conv2D(16, 3, padding='same', kernel_initializer="he_normal",activation='relu')
        self.layer3 = tf.keras.layers.AveragePooling2D()
        self.layer4 = tf.keras.layers.BatchNormalization()
        self.layer5 = tf.keras.layers.Conv2D(32, 3, padding='same', kernel_initializer="he_normal",activation='relu')
        self.layer6 = tf.keras.layers.AveragePooling2D()
        self.layer7 = tf.keras.layers.BatchNormalization()
        self.layer8 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer="he_normal",activation='relu')
        self.layer9 = tf.keras.layers.AveragePooling2D()
        self.layer10 = tf.keras.layers.BatchNormalization()
        self.layer11 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer="he_normal",activation='relu')
        self.layer12 = tf.keras.layers.AveragePooling2D()
        self.layer13 = tf.keras.layers.BatchNormalization()
        self.layer14 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer="he_normal",activation='relu')
        self.layer15 = tf.keras.layers.GlobalAveragePooling2D()

        self.layer16 = tf.keras.layers.Flatten()
        self.layer17 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        inputs   = self.layer1(inputs)
        inputs   = self.layer2(inputs)
        inputs   = self.layer3(inputs)
        inputs   = self.layer4(inputs)
        inputs   = self.layer5(inputs)
        inputs   = self.layer6(inputs)
        inputs   = self.layer7(inputs)
        inputs   = self.layer8(inputs)
        inputs   = self.layer9(inputs)
        inputs   = self.layer10(inputs)
        inputs   = self.layer11(inputs)
        inputs   = self.layer12(inputs)
        inputs   = self.layer13(inputs)
        inputs   = self.layer14(inputs)
        inputs   = self.layer15(inputs)
        inputs   = self.layer16(inputs)
        outputs = self.layer17(inputs)
        return outputs
    
    def get_map(self, inputs):
        inputs = self.layer1(inputs)
        inputs = self.layer2(inputs)
        inputs = self.layer3(inputs)
        inputs = self.layer4(inputs)
        inputs = self.layer5(inputs)
        inputs = self.layer6(inputs)
        inputs = self.layer7(inputs)
        inputs = self.layer8(inputs)
        inputs = self.layer9(inputs)
        inputs = self.layer10(inputs)
        inputs = self.layer11(inputs)
        inputs = self.layer12(inputs)
        inputs = self.layer13(inputs)
        inputs = self.layer14(inputs)
        return inputs


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
