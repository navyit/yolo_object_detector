import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import time
import os


class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection")

        # Настройки видео
        self.video_source = 0  # 0 - камера по умолчанию
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.out = None

        # Настройки YOLO
        self.confidence_threshold = 0.5
        self.net = None
        self.classes = []
        self.output_layers = []
        self.allowed_classes = ['person', 'car', 'dog', 'cat', 'horse', 'sheep', 'cow']

        self.class_colors = {
            'person': (0, 0, 255),    # Красный в BGR (RGB (255,0,0))
            'car': (0, 255, 0),       # Зелёный в BGR и RGB
            'dog': (0, 255, 255),     # Жёлтый в BGR (RGB (255,255,0))
            'cat': (255, 0, 255),     # Пурпурный в BGR и RGB
            'horse': (255, 255, 0),   # Голубой в BGR (RGB (0,255,255))
            'sheep': (128, 0, 128),   # Фиолетовый в BGR и RGB
            'cow': (0, 128, 128)      # Оливковый в BGR и RGB
        }

        # Настройки сохранения
        self.should_save = False
        self.save_path = ""
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = 20

        # Настройки отображения
        self.display_width = 640  # Фиксированная ширина окна
        self.frame_size = None  # Будет вычисляться автоматически
        self.aspect_ratio = None  # Соотношение сторон исходного видео

        # Инициализация
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
        self.setup_ui()
        self.load_yolo()

    def load_yolo(self):
        """Загрузка модели YOLO"""
        try:
            self.net = cv2.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
            with open("yolo/coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель YOLO: {str(e)}")
            self.root.destroy()

    def setup_ui(self):
        """Создание интерфейса"""
        # Фрейм для управления видео
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # Кнопки выбора источника
        self.btn_file = tk.Button(control_frame, text="Выбрать видеофайл", command=self.select_file)
        self.btn_file.pack(side=tk.LEFT, padx=5)

        self.btn_camera = tk.Button(control_frame, text="Использовать камеру", command=self.select_camera)
        self.btn_camera.pack(side=tk.LEFT, padx=5)

        # Кнопка настроек сохранения
        self.btn_save = tk.Button(control_frame, text="Настроить сохранение", command=self.setup_saving)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Слайдер для порога уверенности
        self.confidence_label = tk.Label(self.root, text="Порог уверенности (0.1-1.0):")
        self.confidence_label.pack()

        self.confidence_slider = ttk.Scale(self.root, from_=0.1, to=1.0, value=0.5,
                                           command=lambda v: setattr(self, 'confidence_threshold', float(v)))
        self.confidence_slider.pack(pady=5)

        # Кнопки управления
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.btn_start = tk.Button(btn_frame, text="Старт", command=self.toggle_detection)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_pause = tk.Button(btn_frame, text="Пауза", state=tk.DISABLED, command=self.toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        # Canvas для видео (начальный размер)
        self.canvas = tk.Canvas(self.root, width=self.display_width, height=480)
        self.canvas.pack()

    def select_file(self):
        """Выбор видеофайла с сохранением пропорций"""
        if self.is_running:
            messagebox.showwarning("Внимание", "Остановите детекцию перед сменой источника!")
            return

        file_path = filedialog.askopenfilename(filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ])

        if file_path:
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    raise ValueError("Не удалось открыть видеофайл")

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.aspect_ratio = width / height
                cap.release()

                # Вычисляем высоту для нашей ширины отображения
                display_height = int(self.display_width / self.aspect_ratio)
                self.frame_size = (self.display_width, display_height)

                # Меняем размер canvas
                self.canvas.config(width=self.display_width, height=display_height)

                self.video_source = file_path
                messagebox.showinfo("Успех",
                                    f"Видео загружено\nРазрешение: {width}x{height}\n"
                                    f"Размер отображения: {self.display_width}x{display_height}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def select_camera(self):
        """Выбор камеры с сохранением пропорций"""
        if self.is_running:
            messagebox.showwarning("Внимание", "Остановите детекцию перед сменой источника!")
            return

        # Проверяем камеру и получаем ее разрешение
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру!")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.aspect_ratio = width / height
        cap.release()

        # Вычисляем высоту для нашей ширины отображения
        display_height = int(self.display_width / self.aspect_ratio)
        self.frame_size = (self.display_width, display_height)

        # Меняем размер canvas
        self.canvas.config(width=self.display_width, height=display_height)

        self.video_source = 0
        messagebox.showinfo("Выбрано",
                            f"Используется веб-камера\nРазмер отображения: {self.display_width}x{display_height}")

    def setup_saving(self):
        """Настройка сохранения результатов"""
        save_window = tk.Toplevel(self.root)
        save_window.title("Настройки сохранения")

        # Делаем окно модальным и поверх основного
        save_window.grab_set()  # Захватываем все события
        save_window.transient(self.root)  # Связываем с родительским окном
        save_window.focus_set()  # Устанавливаем фокус

        # Обработка закрытия окна через крестик
        save_window.protocol("WM_DELETE_WINDOW", lambda: self.on_save_window_close(save_window))

        # Флажок сохранения
        self.save_var = tk.BooleanVar(value=self.should_save)
        tk.Checkbutton(save_window, text="Сохранять результат",
                       variable=self.save_var).pack(pady=5)

        # Выбор формата
        format_frame = tk.Frame(save_window)
        format_frame.pack(pady=5)
        tk.Label(format_frame, text="Формат:").pack(side=tk.LEFT)

        self.format_var = tk.StringVar(value="avi")
        formats = [("AVI", "avi"), ("MP4", "mp4"), ("MOV", "mov")]
        for text, fmt in formats:
            tk.Radiobutton(format_frame, text=text, variable=self.format_var,
                           value=fmt).pack(side=tk.LEFT, padx=5)

        # Кнопка выбора папки
        tk.Button(save_window, text="Выбрать место сохранения",
                  command=self.select_save_location).pack(pady=5)

        # Текущий путь
        self.save_location_label = tk.Label(save_window,
                                            text=f"Будет сохранено в: {self.save_path}" if self.save_path else "Место сохранения не выбрано")
        self.save_location_label.pack(pady=5)

        # Кнопка применения
        tk.Button(save_window, text="Применить",
                  command=lambda: self.apply_save_settings(save_window)).pack(pady=10)

    def select_save_location(self):
        """Выбор места сохранения с автоматическим подставлением расширения"""
        default_ext = f".{self.format_var.get()}"
        file_path = filedialog.asksaveasfilename(
            defaultextension=default_ext,
            filetypes=[(f"{self.format_var.get().upper()} files", f"*{default_ext}")],
            title="Укажите файл для сохранения"
        )

        if file_path:
            # Автоматически исправляем расширение, если пользователь его изменил
            if not file_path.endswith(default_ext):
                file_path = os.path.splitext(file_path)[0] + default_ext
            self.save_path = file_path
            self.save_location_label.config(text=f"Файл для сохранения: {self.save_path}")
    def on_save_window_close(self, window):
        """Обработчик закрытия окна настроек через крестик"""
        window.grab_release()  # Освобождаем захват событий
        window.destroy()  # Закрываем окно
    def apply_save_settings(self, window):
        """Применение настроек сохранения с проверками"""
        self.should_save = self.save_var.get()

        # Если сохранение включено, но путь не выбран
        if self.should_save and not self.save_path:
            messagebox.showwarning("Внимание",
                                   "Выберите место сохранения файла!")
            return  # Не закрываем окно, даём исправить

        # Установка кодека в зависимости от формата
        fmt = self.format_var.get()
        if fmt == "avi":
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif fmt == "mp4":
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif fmt == "mov":
            self.fourcc = cv2.VideoWriter_fourcc(*'avc1')

        window.destroy()

        # Сообщаем о результате настройки
        if self.should_save:
            messagebox.showinfo("Сохранение",
                                f"Результаты будут сохранены в:\n{self.save_path}\n"
                                f"Формат: {fmt.upper()}")
        else:
            messagebox.showinfo("Сохранение",
                                "Сохранение результатов отключено.")

    def init_video_writer(self):
        """Инициализация видеозаписи с текущим размером кадра"""
        if self.out:
            self.out.release()
            self.out = None

        if self.should_save and self.save_path and self.frame_size:
            self.out = cv2.VideoWriter(self.save_path, self.fourcc,
                                       self.fps, self.frame_size)
            return True
        return False

    def toggle_detection(self):
        """Запуск/остановка детекции"""
        if not self.is_running:
            try:
                # Проверка источника
                if isinstance(self.video_source, str) and not os.path.exists(self.video_source):
                    raise FileNotFoundError("Видеофайл не найден!")

                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    raise ValueError("Не удалось открыть источник видео!")

                # Если сохранение включено, но не настроено
                if self.should_save and not self.save_path:
                    choice = messagebox.askyesno("Настройки сохранения",
                                                 "Сохранение включено, но файл не выбран.\n"
                                                 "Хотите настроить сейчас?",
                                                 parent=self.root)
                    if choice:
                        self.setup_saving()
                        return  # Позволяем пользователю настроить сохранение
                    else:
                        self.should_save = False  # Отключаем сохранение

                if self.init_video_writer() and self.save_path:
                    messagebox.showinfo("Сохранение",
                                        f"Результат будет сохранён в:\n{self.save_path}")

                self.is_running = True
                self.is_paused = False
                self.btn_start.config(text="Стоп")
                self.btn_pause.config(state=tk.NORMAL)
                self.update_frame()

            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                if self.cap:
                    self.cap.release()
        else:
            self.is_running = False
            self.is_paused = False
            self.btn_start.config(text="Старт")
            self.btn_pause.config(state=tk.DISABLED, text="Пауза")

            if self.cap:
                self.cap.release()
            if self.out:
                self.out.release()
                self.out = None

    def toggle_pause(self):
        """Пауза/продолжение"""
        self.is_paused = not self.is_paused
        self.btn_pause.config(text="Продолжить" if self.is_paused else "Пауза")
        if not self.is_paused:
            self.update_frame()

    def update_frame(self):
        """Обновление кадра с детекцией"""
        if not self.is_running or self.is_paused:
            return

        try:
            # Оптимизация. Пропуск кадров
            skip_frames = 1
            file_size_mb = 0
            if isinstance(self.video_source, str):
                file_size_mb = os.path.getsize(self.video_source) / (1024 * 1024)
                if file_size_mb > 100:
                    skip_frames = 2

            for _ in range(skip_frames):
                ret, frame = self.cap.read()
                if not ret:
                    self.toggle_detection()
                    return

            # Изменение размера с сохранением пропорций
            if self.frame_size:
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]

            # Детекция объектов с динамическим разрешением
            blob_size = 320 if file_size_mb > 100 else 416
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (blob_size, blob_size), swapRB=True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Обработка результатов
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    label = str(self.classes[class_id])

                    if confidence > self.confidence_threshold and label in self.allowed_classes:
                        # Координаты bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Пропускаем слишком маленькие объекты
                        if w * h < (width * height) / 2000:  # 0.2% площади кадра
                            continue

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Применяем Non-Max Suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.3)

            # Рисуем результаты
            font = cv2.FONT_HERSHEY_PLAIN

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    color = self.class_colors.get(label, (0, 0, 255))  # По умолчанию синий
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}",
                                (x, y - 5), font, 1, color, 1)

            # Замер FPS
            current_time = time.time()
            if hasattr(self, 'prev_time'):
                fps = 1 / (current_time - self.prev_time)
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.prev_time = current_time

            # Сохранение кадра (если нужно)
            if self.out and self.should_save:
                self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Отображение в GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk

            self.root.after(10, self.update_frame)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки кадра: {str(e)}")
            self.toggle_detection()

    def _create_blob(self, frame):
        return cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()