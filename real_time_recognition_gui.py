import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from mss import mss


class ScreenCaptureThread(QtCore.QThread):
    """Thread that captures the screen and runs face detection."""

    frame_captured = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.running = True
        # Load OpenCV's pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def run(self):
        with mss() as sct:
            while self.running:
                img = np.array(sct.grab(self.monitor))
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.frame_captured.emit(img)
                self.msleep(30)

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QtWidgets.QLabel):
    def __init__(self, monitor):
        super().__init__()
        self.setWindowTitle("Real-time Screen Recognition")
        self.monitor = monitor

        self.thread = ScreenCaptureThread(self.monitor)
        self.thread.frame_captured.connect(self.update_image)
        self.thread.start()

    def update_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pixmap)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    # Screen region to capture
    monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
    window = MainWindow(monitor)
    window.resize(monitor["width"], monitor["height"])
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
