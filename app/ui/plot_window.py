"""Tk-owned Matplotlib windows used by interactive analysis views."""

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt


def show_figure(parent, figure, title, window_key=None):
    """Display a figure in one reusable Tk window for each logical view."""
    owner = parent.winfo_toplevel()
    key = window_key or title
    windows = getattr(owner, "_plot_windows", None)
    if windows is None:
        windows = {}
        owner._plot_windows = windows

    existing = windows.get(key)
    if existing is not None:
        try:
            if existing.winfo_exists():
                # The newly created figure is not displayed; release it and
                # bring the already open view to the foreground instead.
                plt.close(figure)
                existing.deiconify()
                existing.lift()
                existing.focus_force()
                return existing
        except tk.TclError:
            pass
        windows.pop(key, None)

    window = tk.Toplevel(owner)
    windows[key] = window
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
        if windows.get(key) is window:
            windows.pop(key, None)
        window.destroy()

    window.close_figure = close_window
    window.protocol("WM_DELETE_WINDOW", close_window)
    window.lift()
    window.focus_force()
    return window
