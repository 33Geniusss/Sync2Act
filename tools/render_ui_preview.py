"""Render the main window offscreen for visual QA."""

from pathlib import Path

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from sync2act.gui import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication([])
    print(f"font families: {QFontDatabase.families()[:12]}")
    window = MainWindow()
    window.show()
    app.processEvents()
    output = Path("build").resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        app.processEvents()
        target = output / f"ui_preview_{index}.png"
        if not window.grab().save(str(target)):
            raise RuntimeError(f"Could not save UI preview to {target}")
        print(target)
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
