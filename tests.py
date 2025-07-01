import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tkinter as tk
from object_detection import YOLOApp


class TestYOLOApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Быстрая инициализация без реальной загрузки модели"""
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = YOLOApp(cls.root)

        # Мокаем модель YOLO
        cls.app.net = MagicMock()
        cls.app.output_layers = ['mock_layer']
        cls.app.classes = ['person', 'car']

    def test_1_model_loading(self):
        """Тест загрузки модели (мок)"""
        with patch('cv2.dnn.readNet', return_value=MagicMock()) as mock_read:
            self.app.load_yolo()
            mock_read.assert_called_once()

    def test_2_blob_creation(self):
        """Тест создания blob с подменой OpenCV"""
        test_img = np.zeros((416, 416, 3), dtype=np.uint8)

        with patch('cv2.dnn.blobFromImage') as mock_blob:
            mock_blob.return_value = np.zeros((1, 3, 416, 416))
            result = self.app._create_blob(test_img)
            self.assertEqual(result.shape, (1, 3, 416, 416))

    def test_3_video_processing(self):
        """Тест обработки видео с полной подменой VideoCapture"""

        class MockVideoCapture:
            def __init__(self, *args, **kwargs):
                self.test_frame = np.zeros((416, 416, 3), dtype=np.uint8)

            def read(self):
                return (True, self.test_frame)

            def isOpened(self):
                return True

            def release(self):
                pass

            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    return 416
                elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 416
                return 0

        with patch('cv2.VideoCapture', MockVideoCapture):
            # Тестируем с камеры
            self.app.video_source = 0
            self.app.toggle_detection()

            # Проверяем что обработка запущена
            self.assertTrue(self.app.is_running)

            # Имитируем обработку кадра
            self.app.update_frame()

            # Останавливаем
            self.app.toggle_detection()

    def test_4_error_handling(self):
        """Тест обработки ошибок"""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.read.return_value = (False, None)
            self.app.video_source = 0
            self.app.toggle_detection()
            self.app.update_frame()
            self.assertFalse(self.app.is_running)


if __name__ == "__main__":
    unittest.main(failfast=True)
