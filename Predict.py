import numpy as np
import torch
from PIL import Image


class Predict:
    def __init__(self, model, image_path=None, image=None):
        self.net = model
        if image_path == None:
            self.image = image
        else:
            self.image_path = image_path

    def load_image(self):  # Возвращает матрицу пикселей картинки
        if self.image_path == None:
            img = Image.open(self.image)
        else:
            img = Image.open(self.image_path)
        img.thumbnail((64, 64))
        return np.asarray(img)

    def prediction(self, pic_size=64*64*4):
        dictionary = {0: 'hyperbola', 1: 'sigmoid', 2: 'abs', 3: 'linear', 4: 'parabola'}
        test = self.load_image()
        test = (torch.Tensor(test - np.mean(test)) / np.abs(np.std(test)))
        model = self.net
        mass = np.array(model(test.view(-1, pic_size)).tolist())
        probability = np.max(np.round(np.power(np.e, mass) / np.sum(np.power(np.e, mass)), 3))
        return dictionary[model(test.view(-1, pic_size)).argmax().tolist()], probability
