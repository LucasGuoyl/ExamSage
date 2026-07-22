from __future__ import annotations

import json
import sys
import threading
import tkinter
from tkinter import filedialog


def main() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("folder picker helper requires the main thread")

    root = tkinter.Tk()
    try:
        root.withdraw()
        selected = filedialog.askdirectory(mustexist=True) or None
        payload = json.dumps(selected, ensure_ascii=False).encode("utf-8")
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
