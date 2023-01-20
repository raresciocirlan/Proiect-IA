import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split


data = pd.read_csv('./data-set/colors.csv')
X = data[['R', 'G', 'B']].values
y = data['color_name'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(np.unique(y)), activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10)


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)


img_path = "./data-set/image_2.jpg"
img = cv2.imread(img_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb = rgb


model.save('color_detection_model.h5')


model.save_weights('color_detection_weights.h5')

loaded_model = keras.models.load_model('color_detection_model.h5')
loaded_model.load_weights('color_detection_weights.h5')

clicked = False
r = g = b = xpos = ypos = 0


def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global r, g, b, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)
        if (clicked):
            rgb_values = np.array([[r, g, b]])
            predictions = loaded_model.predict(rgb_values)
            index = np.argmax(predictions)
            color_names = list(data["color_name"])
            color_name = color_names[index]
            cv2.rectangle(img, (0, 0), (600, 60), (rgb_values[0][0], rgb_values[0][1], rgb_values[0][2]), -1)
            cv2.putText(img, color_name, (40, 40), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if(r + g + b >= 600):
                cv2.putText(img, color_name,(50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            clicked = False


cv2.namedWindow('color detection')
cv2.setMouseCallback('color detection', draw_function)

while(1):
    cv2.imshow("color detection", img)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
