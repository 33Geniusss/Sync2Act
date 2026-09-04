"""Launch Sync2Act offscreen, process events, and save a README screenshot."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from sync2act.gui import MainWindow


def main() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    target = Path("assets/gui-overview.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(target)):
        raise RuntimeError("Qt could not save the screenshot")
    window.close()
    print(target)


if __name__ == "__main__":
    main()
