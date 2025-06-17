import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QListWidget, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
from PyQt5.QtWidgets import QSizePolicy

class TrafficSignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận diện biển báo giao thông")
        self.setGeometry(100, 50, 1200, 800)

        self.model_dir = "models"
        self.model = None
        self.model_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.running_mode = None
        self.detected_classes = set()

        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI';
                background-color: #fdfdfd;
            }
            QLabel#TitleLabel {
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
            QLabel {
                color: #444;
            }
        """)

        title = QLabel("Hệ thống nhận diện biển báo - YOLOv8")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)

        # Danh sách chọn model từ thư mục models/
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                padding: 10px;
                font-size: 14px;
                border-radius: 8px;
                background-color: #5eaaa8;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                selection-background-color: #d6f5f2;
                font-size: 13px;
            }
        """)
        self.model_combo.setCursor(Qt.PointingHandCursor)
        self.model_combo.addItem("Chọn model")
        for file in os.listdir(self.model_dir):
            if file.endswith(".pt"):
                self.model_combo.addItem(file)
        self.model_combo.currentIndexChanged.connect(self.select_model)

        self.model_label = QLabel("Chưa chọn model")
        self.model_label.setFont(QFont("Segoe UI", 11))
        self.model_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Vui lòng chọn model")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #2f80ed; font-size: 14px; font-weight: bold")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 3px dashed #ccc;
                border-radius: 10px;
                background-color: #fafafa;
                min-height: 400px;
            }
        """)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        class_label = QLabel("Biển báo đã phát hiện:")
        class_label.setFont(QFont("Segoe UI", 13))
        class_label.setStyleSheet("margin-top: 10px;")

        self.class_list_widget = QListWidget()
        self.class_list_widget.setFixedHeight(150)
        self.class_list_widget.setStyleSheet("""
            QListWidget {
                font-size: 15px;
                padding: 8px;
                background-color: #ffffff;
                border: 2px solid #5eaaa8;
                border-radius: 10px;
                color: #333;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #d6f5f2;
                color: #000;
                font-weight: bold;
                border-radius: 6px;
            }
            QListWidget::item:hover {
                background-color: #eafaf8;
                color: #000;
            }
        """)

        # Nút chức năng
        btn_img = QPushButton("Ảnh")
        btn_vid = QPushButton("Video")
        btn_cam = QPushButton("Camera")

        for btn in (btn_img, btn_vid, btn_cam):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 10px 30px;
                    font-size: 15px;
                    border-radius: 10px;
                    background-color: #ff914d;
                    color: white;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #d96b1f;
                }
            """)

        btn_img.clicked.connect(self.detect_image)
        btn_vid.clicked.connect(self.detect_video)
        btn_cam.clicked.connect(self.detect_camera)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(btn_img)
        button_layout.addWidget(btn_vid)
        button_layout.addWidget(btn_cam)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.model_label)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.image_label)
        layout.addWidget(class_label)
        layout.addWidget(self.class_list_widget)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def select_model(self, index):
        if index == 0:
            return
        model_name = self.model_combo.currentText()
        model_path = os.path.join(self.model_dir, model_name)
        self.load_model(model_path)

    def load_model(self, path):
        try:
            self.model = YOLO(path)
            self.model_path = path
            self.model_label.setText(f"Model: {os.path.basename(path)}")
            self.status_label.setText("Model đã tải thành công")
            self.status_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px")
            self.detected_classes.clear()
            self.class_list_widget.clear()
        except Exception as e:
            self.status_label.setText("Lỗi tải model")
            QMessageBox.critical(self, "Lỗi", str(e))

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qt_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio
        )
        self.image_label.setPixmap(pixmap)

    def update_class_list(self, class_name):
        if class_name not in self.detected_classes:
            self.detected_classes.add(class_name)
            self.class_list_widget.addItem(class_name)

    def stop_running(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.timer.isActive():
            self.timer.stop()
        self.running_mode = None

    def detect_image(self):
        self.stop_running()
        if not self.model:
            QMessageBox.warning(self, "Chưa chọn model", "Hãy chọn model trước.")
            return
        file, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if not file:
            return
        result = self.model(file)[0]
        names = self.model.names
        img = result.plot()
        self.display_image(img)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            self.update_class_list(names[cls_id])

    def detect_video(self):
        self.stop_running()
        if not self.model:
            QMessageBox.warning(self, "Chưa chọn model", "Hãy chọn model trước.")
            return
        file, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Videos (*.mp4 *.avi)")
        if not file:
            return
        self.cap = cv2.VideoCapture(file)
        self.running_mode = "video"
        self.timer.start(30)

    def detect_camera(self):
        self.stop_running()
        if not self.model:
            QMessageBox.warning(self, "Chưa chọn model", "Hãy chọn model trước.")
            return
        self.cap = cv2.VideoCapture(0)
        self.running_mode = "camera"
        self.timer.start(30)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_running()
                return
            result = self.model(frame)[0]
            img = result.plot()
            self.display_image(img)
            names = self.model.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                self.update_class_list(names[cls_id])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficSignApp()
    window.show()
    sys.exit(app.exec_())
