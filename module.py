import cv2
import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    original_img = cv2.imread(image_path)
    img = original_img
    new_img = np.zeros(img.shape, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 20, 20)
    return original_img, img


def extract_and_filter_boxes(img, original_img):
    contours = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.intp(box)  # округление координат
        if len(cv2.approxPolyDP(cnt, 10, True)) == 4:
            boxes.append(box)

    counts_y1 = defaultdict(int)
    counts_y2 = defaultdict(int)
    for box in boxes:
        for point in box[:2]:
            counts_y1[point[1]] += 1
        for point in box[2:]:
            counts_y2[point[1]] += 1

    i = 0
    while i < len(boxes) - 1:
        coords_curr = [boxes[i][x][y] for x in range(4) for y in range(2)]
        coords_next = [boxes[i + 1][x][y] for x in range(4) for y in range(2)]
        if set(coords_curr) == set(coords_next):
            boxes.pop(i)
        i += 1

    counts_y1 = {k: v for k, v in sorted(counts_y1.items(), key=lambda item: item[1], reverse=True)}
    counts_y2 = {k: v for k, v in sorted(counts_y2.items(), key=lambda item: item[1], reverse=True)}
    lst_y1 = list(counts_y1)
    lst_y2 = list(counts_y2)
    y1, y2 = max(lst_y1[:2]), min(lst_y2[:2])  # y1- верхняя граница, y2 - нижняя

    i = 0
    while i < len(boxes):
        box = boxes[i]
        coords = [box[i][1] for i in range(4)]
        if not (y1 in set(coords) and y2 in set(coords)):
            boxes.pop(i)
        i += 1

    i = 0
    alpha = 1  # погрешность в пикселях

    all_y1 = [y1 + i for i in range(-alpha, alpha + 1)]
    all_y2 = [y2 + i for i in range(-alpha, alpha + 1)]

    while i < len(boxes):
        box = boxes[i]
        coords = [box[i][1] for i in range(4)]
        flag1 = flag2 = False
        for y in all_y1:
            if y in set(coords):
                flag1 = True
        for y in all_y2:
            if y in set(coords):
                flag2 = True
        if not (flag1 and flag2):
            boxes.pop(i)
        i += 1



    return boxes, y1, y2


def recognize_characters(original_img, boxes, y1, y2, model_path, class_names):
    model = keras.models.load_model(model_path)
    result = ''
    alpha = 1

    for i in range(len(boxes)):
        box = boxes[len(boxes) - i - 1]
        # Получение координат выделенной области
        x, y, w, h = cv2.boundingRect(box)
        # Обрезка изображения по контуру
        roi = original_img[y:y + h, x:x + w]
        # Проверка на пустоту
        roi_check = original_img[y + alpha:y + h - alpha, x + alpha:x + w - alpha]
        roi_check = cv2.cvtColor(roi_check, cv2.COLOR_BGR2GRAY)
        roi_check = cv2.GaussianBlur(roi_check, (5, 5), 0)
        roi_check = cv2.Canny(roi_check, 20, 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(roi_check, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts == ((), None):
            result += ' '
            continue
        # Приведение изображения к нужному виду
        roi_resized = cv2.resize(roi, (32, 32))
        roi_array = tf.keras.utils.img_to_array(roi_resized)
        roi_array = tf.expand_dims(roi_array, 0)
        # Составление прогнозов нейросети
        predictions = model.predict(roi_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(predictions)
        result += class_names[np.argmax(score)]

    return result


def draw_boxes_on_image(original_img, boxes):
    img_with_contours = original_img.copy()
    for box in boxes:
        cv2.drawContours(img_with_contours, [box], 0, (255, 0, 0), 1)
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    model_path = 'model.h5'
    class_names = ['!', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'Ё', 'А', 'Б', 'В', 'Г',
                   'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч',
                   'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']
