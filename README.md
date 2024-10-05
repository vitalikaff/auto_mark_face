# Проект: Парсер и Обработка Изображений с Яндекс.Картинок

## Описание проекта

Этот проект позволяет автоматизировать процесс поиска, загрузки и обработки изображений по текстовому запросу через Яндекс.Картинки. После загрузки изображений проект выполняет следующие задачи:
1. Конвертация всех файлов в формат `.jpg`.
2. Детекция и обрезка лиц на изображениях с использованием предобученной модели YOLOv5.
3. Сохранение нормализированных изображений лиц в отдельную папку.

## Основные возможности
- **Парсинг изображений по текстовому запросу**: Поиск и загрузка изображений с помощью модуля `yandex_images_parser`.
- **Обработка изображений**:
  - Конвертация файлов в формат `.jpg`.
  - Переименование изображений в формате `face_{i}.jpg`.
  - Удаление некорректных файлов.
- **Детекция и обрезка лиц**: Использование модели YOLOv5 для обнаружения лиц на изображениях и сохранение обрезанных изображений в отдельную папку.

## Требования

Для корректной работы проекта необходимы следующие компоненты:

1. **Python версии 3.x**.
2. **Зависимости**:
   - `torch`
   - `opencv-python`
   - `Pillow`
   - `transliterate`
   - Модули `yandex_images_parser` и `yip_utils` (скачаны с чужого репозитория: [Ulbwaa/YandexImagesParser](https://github.com/Ulbwaa/YandexImagesParser)).
3. **Geckodriver**:
   - Скачайте и установите `geckodriver` для работы с браузером Firefox.
   - Путь к `geckodriver` должен быть указан корректно в окружении или в проекте (`geckodriver/geckodriver.exe`).
4. **Mozilla Firefox**:
   - Необходим для работы парсера изображений.
5. **YOLOv5**:
   - Модель `yolov5s.pt` будет загружена автоматически при запуске.

## Установка

1. Клонируйте проект:
   ```bash
   git clone https://github.com/vitalikaff/auto_mark_face
   ```

2. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Убедитесь, что `geckodriver` и Firefox установлены и настроены.

## Как использовать

### Парсинг и обработка изображений

1. Импортируйте функцию `parse_img_by_text` из модуля `parser.py`.

2. Вызовите функцию с необходимым текстовым запросом:

   ```python
   import parser

   parser.parse_img_by_text('лицо моргенштерн', limit=5)
   ```

   **Аргументы**:
   - `query`: Текстовый запрос для поиска изображений.
   - `name` (опционально): Название для папки, в которую будут сохранены изображения. Если не указано, будет сгенерировано автоматически.
   - `limit`: Количество загружаемых изображений (по умолчанию 10).

3. Все загруженные изображения будут сохранены в папке `images/{name}`. Далее изображения будут обработаны:
   - Конвертированы в формат `.jpg`.
   - Лица будут обрезаны и сохранены в папку `images/{name}/norm/`.

### Пример использования

В файле `example.py` приведён пример вызова функции:

```python
import parser

parser.parse_img_by_text('лицо моргенштерн', limit=5)
```

Этот код выполнит поиск пяти изображений по запросу "лицо моргенштерн", загрузит их, обработает и сохранит обрезанные изображения лиц.

## Структура проекта

- `parser.py` — основной скрипт для парсинга и обработки изображений.
- `yandex_images_parser.py` — скрипт для парсинга изображений через Яндекс.Картинки (взятый из другого репозитория).
- `yip_utils.py` — утилиты для работы с изображениями (взяты из другого репозитория).
- `example.py` — пример использования парсера.

## Лицензия

Проект использует код из [Ulbwaa/YandexImagesParser](https://github.com/Ulbwaa/YandexImagesParser)

## Контакты

Если у вас есть вопросы или предложения, свяжитесь с автором проекта через GitHub.
