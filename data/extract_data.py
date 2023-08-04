from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Data(tf.keras.utils.Sequence):
    def __init__(self, batch_size: int = 32, path: str = ""):

        self.batch_size = batch_size
        self.path = path

        self.df = pd.read_csv(path + "data.csv")

        self._gen_monet = self.__gen("monet")
        self._gen_photo = self.__gen("photo")

        self.__len = self.__calc() // batch_size

    def __calc(self):
        k1 = self.df[self.df["path"] == "photo"].__len__()
        k2 = self.df[self.df["path"] == "monet"].__len__()

        return max(k2, k1)

    def __gen(self, path: str):

        X = np.zeros((self.batch_size, 256, 256, 3), dtype=float)
        k = 0

        for name in self.df.name[self.df["path"] == path].sample(frac=1).values:

            if k >= self.batch_size:
                yield X
                X = np.zeros((self.batch_size, 256, 256, 3), dtype=float)
                k = 0

            X[k] = np.array(Image.open(f"{self.path}{path}/{name}"), dtype=float) / 255.0
            k += 1

        yield X[:k]

    def __len__(self):
        return self.__len

    def __getitem__(self, idx):
        try:
            x1, x2 = next(self._gen_monet), next(self._gen_photo)
            if len(x1) != len(x2):
                raise StopIteration
            return x1, x2
        except StopIteration:
            self._gen_monet = self.__gen("monet")
            self._gen_photo = self.__gen("photo")

        x1, x2 = next(self._gen_monet), next(self._gen_photo)
        return x1, x2


if __name__ == '__main__':
    d = Data()
    x1, x2 = d[0]
    print(x2.shape,x1.shape)
    plt.imshow(x1[1])
    plt.show()
