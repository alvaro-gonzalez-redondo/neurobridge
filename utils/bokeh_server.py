import json
from functools import partial
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.server.server import Server
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.layouts import column
from bokeh.palettes import Category10_10
from bokeh.transform import transform
from tornado.web import RequestHandler
from tornado.websocket import WebSocketHandler

DEFAULT_ROLLOVER = 2000 

server_context = {
    "doc": None,
    "layout": None,
    "elements": {} 
}

class BaseHandler(RequestHandler):
    def set_default_headers(self): self.set_header("Content-Type", "application/json")
    def send_ok(self, msg="ok"): self.write(json.dumps({"status": "ok", "msg": msg}))
    def send_error(self, msg): self.set_status(400); self.write(json.dumps({"status": "error", "msg": msg}))
    def get_json(self): return json.loads(self.request.body.decode("utf-8"))

class ResetHandler(BaseHandler):
    def post(self):
        doc = server_context["doc"]
        if doc: doc.add_next_tick_callback(lambda: self._do_reset())
        server_context["elements"] = {}
        self.send_ok()
    def _do_reset(self):
        if server_context["layout"]: server_context["layout"].children = []

# === CREATE RASTER (Simplificado: Solo Spikes) ===
class CreateRasterHandler(BaseHandler):
    def post(self):
        data = self.get_json()
        rollover = data.get("rollover", DEFAULT_ROLLOVER)
        # Ya no necesitamos group_names para líneas aquí
        self._create(data.get("id"), data.get("title"), rollover)
        self.send_ok()

    def _create(self, pid, title, rollover):
        doc = server_context["doc"]
        if doc: doc.add_next_tick_callback(partial(self._add_plot, pid, title, rollover))

    def _add_plot(self, pid, title, rollover):
        source = ColumnDataSource(data=dict(t=[], n=[], g=[]))
        
        # Altura ajustada (un poco menos alto ya que ahora tendremos dos gráficas)
        fig = figure(title=title, height=200, sizing_mode="stretch_width", 
                     x_axis_label="Time (s)", y_axis_label="Neuron ID")
                     # WebGL quitado para evitar problemas con marcadores 'dash'
        
        mapper = LinearColorMapper(palette=Category10_10, low=0, high=9)
        fig.scatter(x='t', y='n', source=source, 
                    #marker="dash", angle=1.5707, size=15, 
                    marker="dot", size=10, 
                    line_color=transform('g', mapper), fill_color=None,
                    legend_field='g')
        
        fig.legend.click_policy = "hide"
        
        server_context["elements"][pid] = {
            "source": source, 
            "fig": fig, 
            "type": "raster",
            "rollover": rollover 
        }
        server_context["layout"].children.append(fig)

# === CREATE LINEPLOT (Estándar) ===
class CreateLinePlotHandler(BaseHandler):
    def post(self):
        data = self.get_json()
        series_names = data.get("series_names", ["value"])
        rollover = data.get("rollover", DEFAULT_ROLLOVER)
        self._create(data.get("id"), data.get("title"), series_names, rollover)
        self.send_ok()

    def _create(self, pid, title, series_names, rollover):
        doc = server_context["doc"]
        if doc: doc.add_next_tick_callback(partial(self._add_plot, pid, title, series_names, rollover))

    def _add_plot(self, pid, title, series_names, rollover):
        data_dict = dict(t=[])
        for name in series_names: data_dict[name] = []
        source = ColumnDataSource(data=data_dict)
        
        # WebGL activado aquí porque son líneas continuas (mejora rendimiento)
        fig = figure(title=title, height=350, sizing_mode="stretch_width", 
                     x_axis_label="Time (s)", output_backend="webgl")

        colors = Category10_10 * (len(series_names) // 10 + 1)
        for i, name in enumerate(series_names):
            fig.line(x='t', y=name, source=source, color=colors[i], line_width=2, legend_label=name)
        
        fig.legend.click_policy = "hide"

        server_context["elements"][pid] = {
            "source": source, 
            "fig": fig, 
            "series_names": series_names,
            "type": "line",
            "rollover": rollover
        }
        server_context["layout"].children.append(fig)

# === WEBSOCKET HANDLER (Simplificado) ===
class WSDataHandler(WebSocketHandler):
    def open(self): pass
    def check_origin(self, origin): return True

    def on_message(self, message):
        try:
            msg = json.loads(message)
            cmd = msg.get("cmd")
            doc = server_context["doc"]
            if not doc: return

            pid = msg["id"]
            if pid not in server_context["elements"]: return
            elem = server_context["elements"][pid]

            if cmd == "push_spikes":
                new_data = dict(t=msg["times"], n=msg["ids"], g=msg["groups"])
                doc.add_next_tick_callback(partial(self._stream, elem, new_data))

            elif cmd == "push_values":
                t_val = msg["t"]
                y_raw = msg["y"]
                if not isinstance(y_raw, list): y_vals = [y_raw]
                else: y_vals = y_raw

                new_data = {"t": [t_val]}
                if "series_names" in elem:
                    for i, series_key in enumerate(elem["series_names"]):
                        val = y_vals[i] if i < len(y_vals) else 0 
                        new_data[series_key] = [val]
                
                doc.add_next_tick_callback(partial(self._stream, elem, new_data))

        except Exception as e:
            print(f"WS Error: {e}")

    def _stream(self, elem, new_data):
        source = elem["source"]
        rollover = elem.get("rollover", DEFAULT_ROLLOVER)
        source.stream(new_data, rollover=rollover)

# ... Setup del servidor igual ...
def modify_doc(doc):
    doc.theme = "dark_minimal"   # 👈 ACTIVAR TEMA OSCURO
    layout = column(sizing_mode="stretch_width")
    server_context["doc"] = doc
    server_context["layout"] = layout
    doc.add_root(layout)
    doc.title = "Neurobridge RT"

api_patterns = [
    (r"/api/reset", ResetHandler),
    (r"/api/create_raster", CreateRasterHandler),
    (r"/api/create_lineplot", CreateLinePlotHandler),
    (r"/ws", WSDataHandler),
]
app = Application(FunctionHandler(modify_doc))
server = Server({'/viz': app}, num_procs=1, port=5006, extra_patterns=api_patterns)
server.start()
server.io_loop.add_callback(server.show, "/viz")
server.io_loop.start()