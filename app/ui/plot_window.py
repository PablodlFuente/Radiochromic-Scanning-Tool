"""Tk-owned Matplotlib windows used by interactive analysis views."""

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt


def show_figure(parent, figure, title):
    """Display *figure* in a Tk window whose lifetime is owned by the application."""
    owner = parent.winfo_toplevel()
    window = tk.Toplevel(owner)
    window.title(title)
    window.geometry("900x700")
    canvas = FigureCanvasTkAgg(figure, master=window)
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()

    def close_window():
        plt.close(figure)
        window.destroy()

    window.close_figure = close_window
    window.protocol("WM_DELETE_WINDOW", close_window)
    window.lift()
    window.focus_force()
    return window
