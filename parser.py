import os
import cv2
import torch
import transliterate
from PIL import Image
from pathlib import Path
from yip_utils import save_images
from yip_utils import remove_duplicates
from yandex_images_parser import Parser, ImageType

# Функция создания и проверки папки по пути 'path'
def create_and_check_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Папка '{path}' успешно создана.")
    else:
        print(f"Папка '{path}' уже существует.")

# Функция для конвертации или переименования файла в формат .jpg
def process_image(image_path, counter, failed_files):
    try:
        img = Image.open(image_path)
        # Конвертируем в RGB, если изображение имеет альфа-канал
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # Формируем новое имя файла face_{i}.jpg
        new_image_name = f"face_{counter}.jpg"
        new_image_path = os.path.join(os.path.dirname(image_path), new_image_name)
        
        # Сохраняем изображение в формате .jpg с новым именем
        img.save(new_image_path, "JPEG")
        
        # Удаляем исходный файл, если его расширение было не .jpg
        if image_path != new_image_path:
            os.remove(image_path)
        print(f"Файл {image_path} успешно конвертирован и переименован в {new_image_path}")
        return True  # Успех
    except Exception as e:
        print(f"Ошибка при обработке файла {image_path}: {e}")
        failed_files.append(image_path)  # Добавляем файл в список с ошибками
        return False  # Неудача

# Функция для обработки файлов в директории
def process_files_in_directory(directory):
    failed_files = []  # Локальный список для хранения файлов, которые не удалось обработать
    success_counter = 1  # Счётчик успешных фотографий

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Проверяем, что это файл, а не папка
        if os.path.isfile(file_path):
            # Получаем расширение файла
            file_extension = os.path.splitext(file_path)[1].lower()

            # Если расширения нет, пробуем обработать как изображение
            if file_extension == '':
                print(f"Файл {file_path} не имеет расширения. Пытаемся конвертировать в .jpg...")
                if process_image(file_path, success_counter, failed_files):
                    success_counter += 1  # Увеличиваем счётчик при успешной обработке
            # Если файл не в формате .jpg, конвертируем
            elif file_extension != ".jpg":
                if process_image(file_path, success_counter, failed_files):
                    success_counter += 1  # Увеличиваем счётчик при успешной обработке
            # Если файл уже .jpg, просто переименовываем
            else:
                print(f"Файл {file_path} уже в формате .jpg. Переименовываем...")
                if process_image(file_path, success_counter, failed_files):
                    success_counter += 1  # Увеличиваем счётчик при успешной обработке

    return failed_files  # Возвращаем список файлов с ошибками

# Функция для детекции и сохранения лиц
def detect_and_crop_faces(image_path, output_folder, model, FACE_CLASS_ID):
    img = cv2.imread(image_path)
    results = model(img)  # Применение модели YOLO

    # Обрабатываем детекции
    for i, (xmin, ymin, xmax, ymax, conf, cls) in enumerate(results.xyxy[0]):
        if int(cls) == FACE_CLASS_ID:  # Только лица (класс person)
            # Обрезаем изображение по координатам детекции
            face_crop = img[int(ymin):int(ymax), int(xmin):int(xmax)]

            # Генерация имени файла
            base_name = os.path.basename(image_path).split('.')[0]
            cropped_image_path = os.path.join(output_folder, f'{base_name}_face_{i}.jpg')

            # Сохранение обрезанного изображения
            cv2.imwrite(cropped_image_path, face_crop)
            print(f'Сохранено лицо: {cropped_image_path}')


def parse_img_by_text(query: str, name: str = None, limit: int = 10):
    '''Функция, которая парсит картинки по запросу query, в количестве limit
    и сохраняет результат в путь images/name'''
    create_and_check_dir('images')  # Создадим папку images, если она ещё не создана.

    # Создадим папку с фотографиями запроса.
    if name is None:
        name = transliterate.translit(query, 'ru', reversed=True).replace(' ', '_').lower()
        print(f'Папка названа автоматически {name}')
    create_and_check_dir('images/' + name)

    # Спарсим картинки с помощью "yandex_images_parser"
    request = Parser().query_search(query=query, limit=limit)
    save_images(urls=request, dir_path="images/" + name)

    # Чистим от сломанных и не от картинок
    directory = "images/" + name  # Путь к директории с изображениями
    failed_files = process_files_in_directory(directory)  # Вызов функции для обработки файлов в директории

    # Вывод списка файлов, которые не удалось обработать
    if failed_files:
        print("Не удалось обработать следующие файлы:")
        for failed_file in failed_files:
            print(failed_file)
    else:
        print("Все файлы успешно обработаны.")

    # Загрузка предобученной модели YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolo v5 small

    # Класс лиц для модели YOLOv5 (это 0 — в COCO используется класс "person")
    FACE_CLASS_ID = 0

    # Пути к папкам
    input_folder = f'images/{name}/'
    output_folder = f'images/{name}/norm/'

    # Создание выходной папки, если она не существует
    os.makedirs(output_folder, exist_ok=True)

    # Обрабатываем все изображения в папке
    for image_file in Path(input_folder).glob("*.jpg"):
        detect_and_crop_faces(str(image_file), output_folder, model, FACE_CLASS_ID)