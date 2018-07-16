from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.core.window import Window
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('num_model.h5')

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        with self.canvas:
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=30)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

class MyPaintApp(App):

    def build(self):
        parent = Widget()
        self.parent = parent
        self.painter = MyPaintWidget()
        parent.add_widget(self.painter)
        Window.bind(on_key_down=self.key_action)
        return parent

    def key_action(self, *args):
        if args[1] != 32:
            return
        tmp_file = 'tmp.png'
        self.parent.export_to_png(tmp_file)
        img = Image.open(open(tmp_file, 'rb'))
        img = img.resize(size=(28,28), resample=Image.LANCZOS)
        img.save('tmp2.png')
        pixels = np.add.reduce(np.array(img), 2) / (3 * 256)
        prediction = model.predict([[pixels]])[0]
        percentage = (prediction * 100).round()
        sorted_index = np.flip(np.argsort(prediction), axis=0)
        sorted_values = [(sorted_index[i], percentage[sorted_index[i]]) for i in range(10)]
        print(sorted_index[0], sorted_values)

        self.painter.canvas.clear()

if __name__ == '__main__':
    MyPaintApp().run()
