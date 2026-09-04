import os
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sync2act.gui import MainWindow, main


def capture(path: str) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()

    def save_and_close():
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(target)):
            raise RuntimeError("Qt could not save the screenshot")
        window.close()
        app.quit()

    QTimer.singleShot(1500, save_and_close)
    return app.exec()


if __name__ == "__main__":
    capture_path = os.environ.get("SYNC2ACT_CAPTURE_PATH")
    raise SystemExit(capture(capture_path) if capture_path else main())
