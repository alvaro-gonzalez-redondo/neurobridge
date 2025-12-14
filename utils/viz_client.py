import requests
import json
import websocket
import time
import socket
import subprocess
import os
import sys

class VisualizerClient:
    def __init__(self, host="localhost", port=5006, target_fps=30):
        self.host = host
        self.port = port
        
        self._ensure_server_running()
        
        self.http_url = f"http://{host}:{port}/api"
        self.ws_url = f"ws://{host}:{port}/ws"
        
        self.target_interval = 1.0 / target_fps
        self.last_flush_time = time.time()
        
        # Buffers: Usaremos claves compuestas para evitar colisiones
        # Clave: (plot_id, "spikes") o (plot_id, "values")
        self.buffers = {}

        self.ws = websocket.WebSocket()
        self._connect_ws_with_retries()

    def _ensure_server_running(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((self.host, self.port))
        sock.close()
        
        if result != 0:
            print(f"⚠️ Puerto {self.port} cerrado. Lanzando servidor Bokeh...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            server_script = os.path.join(current_dir, "bokeh_server.py")
            
            if not os.path.exists(server_script):
                # Fallback simple
                possible_path = os.path.join(current_dir, "utils", "bokeh_server.py")
                if os.path.exists(possible_path):
                    server_script = possible_path
            
            if os.path.exists(server_script):
                subprocess.Popen([sys.executable, server_script], 
                                 cwd=os.path.dirname(server_script),
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL) 
                print("⏳ Esperando a que el servidor inicie...")
                time.sleep(3)
            else:
                print(f"❌ No encuentro bokeh_server.py en {current_dir}")

    def _connect_ws_with_retries(self, retries=5):
        for i in range(retries):
            try:
                self.ws.connect(self.ws_url)
                print(f"✅ Conectado a WS: {self.ws_url}")
                return
            except Exception:
                time.sleep(1)
        print("❌ No se pudo conectar al WebSocket.")

    def _post(self, endpoint, payload):
        try:
            requests.post(f"{self.http_url}/{endpoint}", json=payload)
        except Exception as e:
            print(f"HTTP Error: {e}")

    def reset(self):
        self._post("reset", {})
        self.buffers = {} 

    def create_raster(self, plot_id, title=None, rollover=None, group_names=None):
        payload = {
            "id": plot_id, 
            "title": title or plot_id,
            "group_names": group_names or []
        }
        if rollover: payload["rollover"] = rollover
        self._post("create_raster", payload)

    def create_lineplot(self, plot_id, series_names=["value"], title=None, rollover=None):
        payload = {
            "id": plot_id, 
            "title": title or plot_id,
            "series_names": series_names
        }
        if rollover: payload["rollover"] = rollover
        self._post("create_lineplot", payload)

    # ============================================================
    #  PUSH METHODS (CORREGIDOS PARA EVITAR COLISIÓN DE CLAVES)
    # ============================================================

    def push_spikes(self, plot_id, neural_ids, spike_times, group_ids):
        if not neural_ids: return
        
        # Usamos clave compuesta para distinguir el buffer de spikes del de values
        # aunque vayan al mismo plot_id
        key = (plot_id, "spikes")
        
        if key not in self.buffers:
            self.buffers[key] = {
                "type": "spikes", 
                "real_id": plot_id, # Guardamos el ID real para enviar después
                "ids": [], 
                "times": [], 
                "groups": []
            }
            
        self.buffers[key]["ids"].extend(neural_ids)
        self.buffers[key]["times"].extend(spike_times)
        self.buffers[key]["groups"].extend(group_ids)
        
        self._check_flush()

    def push_values(self, plot_id, t, y):
        key = (plot_id, "values")
        
        if key not in self.buffers:
            self.buffers[key] = {
                "type": "values", 
                "real_id": plot_id, # Guardamos el ID real
                "t": [], 
                "y": []
            }
        
        if isinstance(t, list):
            self.buffers[key]["t"].extend(t)
            self.buffers[key]["y"].extend(y)
        else:
            self.buffers[key]["t"].append(t)
            val = y if isinstance(y, list) else [y]
            self.buffers[key]["y"].append(val)
        
        self._check_flush()

    # ============================================================
    #  FLUSH (ADAPTADO A CLAVES COMPUESTAS)
    # ============================================================

    def _check_flush(self):
        now = time.time()
        if (now - self.last_flush_time) >= self.target_interval:
            self.flush()
            self.last_flush_time = now

    def flush(self):
        if not self.buffers: return

        try:
            # Iteramos sobre los valores del buffer, ignorando las claves compuestas
            # y usando 'real_id' para construir el mensaje.
            for data in self.buffers.values():
                pid = data["real_id"] # Recuperamos el ID original

                if data["type"] == "spikes":
                    if not data["ids"]: continue
                    payload = json.dumps({
                        "cmd": "push_spikes",
                        "id": pid,
                        "ids": data["ids"],
                        "times": data["times"],
                        "groups": data["groups"]
                    })
                    self.ws.send(payload)

                elif data["type"] == "values":
                    if not data["t"]: continue
                    count = len(data["t"])
                    for i in range(count):
                        payload = json.dumps({
                            "cmd": "push_values",
                            "id": pid,
                            "t": data["t"][i],
                            "y": data["y"][i]
                        })
                        self.ws.send(payload)

            self.buffers = {}

        except websocket.WebSocketConnectionClosedException:
            try: self.ws.connect(self.ws_url)
            except: pass
        except Exception as e:
            print(f"Error flushing: {e}")