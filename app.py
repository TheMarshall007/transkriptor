import sys

from cli import run_cli
from gui import TranskriptorApp

import tkinter as tk


def main():
    # Se passar argumentos → CLI
    if len(sys.argv) > 1:
        run_cli()
    else:
        # Sem argumentos → GUI
        root = tk.Tk()
        app = TranskriptorApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()