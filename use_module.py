from module import *
def start_module(image_path):
    model_path = 'model.h5'
    class_names = ['!', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'Ё', 'А', 'Б', 'В', 'Г',
                   'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч',
                   'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

    original_img, img = preprocess_image(image_path)
    boxes, y1, y2 = extract_and_filter_boxes(img, original_img)
    result = recognize_characters(original_img, boxes, y1, y2, model_path, class_names)
    print(result)

    return result