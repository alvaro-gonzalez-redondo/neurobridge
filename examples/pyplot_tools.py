import os
import sys
import matplotlib
import matplotlib.pyplot as plt


def can_display_graphics():
    # Lista de backends que podrían ser interactivos
    interactive_backends = [backend.lower() for backend in [
        'GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg',
        'Qt5Agg', 'QtAgg', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg'
    ]]
    backend = matplotlib.get_backend()

    # En Unix, requiere DISPLAY; en Windows/Mac suele funcionar siempre
    has_display = (
        sys.platform.startswith('win') or
        sys.platform == 'darwin' or
        os.environ.get("DISPLAY") is not None
    )

    return backend.lower() in interactive_backends and has_display


def show_or_save_plot(filename="output.png", log=None):
    if can_display_graphics():
        plt.show()
    else:
        plt.savefig(filename)
        if log:
            log(f"Gráfico guardado como '{filename}'")
        else:
            print(f"Gráfico guardado como '{filename}'")