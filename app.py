import pygame as pg
import numpy as np
import math
import tensorflow as tf

pg.init()
window = pg.display.set_mode((700, 700))
clock = pg.time.Clock()
pg.display.set_caption('Digit Classifier')

model = tf.keras.models.load_model('digit_classifier.h5')
pixels = np.zeros((28,28))

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            raise SystemExit
    
    keys = pg.key.get_pressed()
    if keys[pg.K_c]:
        pixels = np.zeros((28,28))
    if keys[pg.K_p]:
        prediction = model.predict(np.array([pixels.T]))
        num = np.argmax(prediction[0])
        pg.display.set_caption(f'Predicted = {num}')
    buttons = pg.mouse.get_pressed()
    mousepos = pg.mouse.get_pos()
    if buttons[0]:
        x = math.floor(mousepos[0]/25)
        y = math.floor(mousepos[1]/25)
        pixels[x,y]=1

    window.fill('black')
    for i in range(0,700,25):
        pg.draw.line(window, (255,255,255), (i, 0), (i, 700))
    for j in range(0,700,25):
        pg.draw.line(window, (255,255,255), (0, j), (700, j))
    for y in range(28):
        for x in range(28):
            if pixels[x,y] == 1:
                pg.draw.rect(window, 'white', (x*25, y*25, 25, 25))
    pg.display.flip()
    clock.tick(60)