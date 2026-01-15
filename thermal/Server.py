import socket
import threading
import time
from typing import Optional, Tuple
import VivotekThermalPoller
from VivotekThermalPoller import VivotekThermalPoller
import http.server
import urllib.parse
import json


class TCPServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port

        self.temperature = 0.0

        # 建立 TCP server socket
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(5)

        # 讓 accept 最多卡 1 秒，避免 Ctrl+C 時卡死
        self.server_sock.settimeout(1.0)

        # 控制 server 狀態
        self.is_running = False

        # 紀錄「目前」連線中的 client（用來主動 send）
        self._current_conn: Optional[socket.socket] = None
        self._current_addr: Optional[Tuple[str, int]] = None
        self._conn_lock = threading.Lock()

        print(f"[INIT] Server listening on {self.host}:{self.port}")

    # ========= 對外 API =========

    def start(self) -> None:
        """主迴圈：接受 client 連線並開 thread 處理。"""
        self.is_running = True
        print("[START] TCPServer is running...")

        try:
            while self.is_running:
                try:
                    conn, addr = self.server_sock.accept()
                except socket.timeout:
                    # 每秒醒來一次，讓 KeyboardInterrupt 有機會發生
                    continue
                except OSError:
                    # socket 已關閉
                    break

                print(f"[CONNECT] Client connected from {addr}")

                # 更新目前的連線（用來主動 send）
                with self._conn_lock:
                    self._current_conn = conn
                    self._current_addr = addr

                # 為這個 client 建立獨立 thread 處理收資料
                t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                t.start()

        except KeyboardInterrupt:
            print("\n[STOP] KeyboardInterrupt, shutting down...")
        finally:
            self.close()

    def close(self) -> None:
        """關閉 server 與目前連線。"""
        if self.is_running:
            self.is_running = False

        # 關閉目前 client 連線
        with self._conn_lock:
            if self._current_conn is not None:
                try:
                    self._current_conn.close()
                except OSError:
                    pass
                self._current_conn = None
                self._current_addr = None

        # 關閉 server socket
        try:
            self.server_sock.close()
        except OSError:
            pass

        print("[CLOSE] Server socket closed.")

    # ========= 內部處理 =========

    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        """在獨立 thread 裡處理單一 client 的收資料與解析。"""
        with conn:
            while True:
                try:
                    data = conn.recv(8)  # 一次收 8 bytes
                except ConnectionResetError:
                    print(f"[ERROR] Connection reset by {addr}")
                    break
                except OSError:
                    print(f"[ERROR] Socket error from {addr}")
                    break

                if not data:
                    print(f"[DISCONNECT] Client {addr} closed connection")
                    break

                # 印出 HEX 格式
                print("[RECV]", ' '.join(f"{b:02X}" for b in data))

                # 解析資料
                self.data_parse(data)

        # 若這個 client 剛好是「目前」的那一個，把記錄清掉
        with self._conn_lock:
            if self._current_conn is conn:
                self._current_conn = None
                self._current_addr = None
        print(f"[END] Handle thread for {addr} exited.")

    def data_parse(self, data: bytes) -> None:
        """解析收到的 8-byte 資料。"""
        if len(data) != 8:
            print(f"[PARSE] Invalid length: {len(data)}, expected 8 bytes.")
            return

        # Header 檢查：0x88 0x66
        if data[0] == 0x88 and data[1] == 0x66:
            if not self.sum_check(data):
                print("[PARSE] Checksum invalid.")
                return

            cmd = data[2]
            match cmd:
                case 0x4F:  # 'O'
                    print("[PARSE] Received 'O' command.")
                case 0x51:  # 'Q'
                    self.send_temperature(self.temperature)
                    print("[PARSE] Received 'Q' command.")
                case _:
                    print(f"[PARSE] Unknown command: 0x{cmd:02X}")
        else:
            print("[PARSE] Invalid header.")

    def sum_check(self, data: bytes) -> bool:
        """檢查 8-byte 資料的加總是否為 0 (sum & 0xFF == 0)。"""
        if len(data) != 8:
            return False

        total = 0
        for b in data:
            total += b

        ok = (total & 0xFF) == 0
        return ok

    def check_sum_and_send(self, data: bytes) -> None:
        """
        對前 7 bytes 做 checksum，計算出第 8 byte，然後送出去。
        data 必須是 7 bytes。
        """
        if len(data) != 7:
            print(f"[CHECKSUM] Input length must be 7, got {len(data)}")
            return

        total = 0
        for b in data:
            total += b

        check_sum = (-total) & 0xFF
        packet = data + bytes([check_sum])

        self.send_data(packet)

    def send_data(self, data: bytes) -> None:
        """送資料給目前的 client（如果有連線的話）。"""
        with self._conn_lock:
            conn = self._current_conn

        if conn is None:
            print("[SEND] No client connected, cannot send.")
            return

        try:
            conn.sendall(data)
            print("[SEND]", ' '.join(f"{b:02X}" for b in data))
        except OSError as e:
            print(f"[SEND ERROR] {e}")

    # ========= 封包範例 =========

    def send_temperature(self, temperature: float) -> None:
        """
        範例封包：
        Header 0x88 0x66
        Command 0x61
        後面 4 bytes 先填 0
        最後一 byte 由 check_sum_and_send() 自動補上
        """
        print(f"temperature: {temperature}")
        temp_hex = hex(int(temperature))
        base = b'\x88\x66\x71' + bytes.fromhex(temp_hex[2:]) + b'\x00\x00\x00'  # 7 bytes
        self.check_sum_and_send(base)
        print("")


class StatusHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP handler that serves a minimal UI and an API endpoint
    to report the current max temperature from the TCPServer instance.
    """
    tcp_server = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = (
                                """<!doctype html>
                                <html>
                                <head>
                                    <meta charset="utf-8">
                                    <meta name="viewport" content="width=device-width, initial-scale=1">
                                    <title>Thermal Status</title>
                                    <style>
                                        :root{--bg:#f4f7fb;--card:#ffffff;--muted:#6b7280;--good:#16a34a;--warn:#f59e0b;--bad:#ef4444}
                                        html,body{height:100%;margin:0;font-family:Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial}
                                        body{background:var(--bg);display:flex;align-items:center;justify-content:center}
                                        .card{background:var(--card);padding:28px;border-radius:14px;box-shadow:0 10px 30px rgba(20,40,80,0.08);width:340px;text-align:center}
                                        .title{font-size:14px;color:var(--muted);letter-spacing:0.6px}
                                        .temp{font-size:64px;font-weight:700;margin:10px 0;color:#111;display:inline-flex;align-items:baseline}
                                        .unit{font-size:20px;margin-left:8px;color:var(--muted)}
                                        .meta{font-size:12px;color:var(--muted);margin-top:8px}
                                        .bar{height:10px;border-radius:99px;margin-top:14px;background:linear-gradient(90deg,var(--good),var(--warn),var(--bad));opacity:0.12}
                                        .indicator{height:10px;border-radius:99px;margin-top:10px}
                                        .small{font-size:13px;color:var(--muted)}
                                        @media (max-width:420px){.card{width:90vw}}
                                    </style>
                                </head>
                                <body>
                                    <div class="card">
                                        <div class="title">Current Max Temperature</div>
                                        <div id="temp" class="temp">--<span class="unit">°C</span></div>
                                        <div id="meta" class="meta">last updated: --</div>
                                        <div class="bar"></div>
                                        <div id="indicator" class="indicator"></div>
                                        <div style="margin-top:12px"><span class="small">Data refreshes every second</span></div>
                                    </div>

                                    <script>
                                        async function fetchTemp(){
                                            try{
                                                const r = await fetch('/api/max_temp');
                                                if(!r.ok) throw new Error('network');
                                                const j = await r.json();
                                                const t = Number(j.max_temperature) || 0;
                                                const formatted = t.toFixed(1);
                                                const tempEl = document.getElementById('temp');
                                                tempEl.innerHTML = formatted + '<span class="unit">°C</span>';
                                                const now = new Date();
                                                document.getElementById('meta').innerText = 'last updated: ' + now.toLocaleTimeString();

                                                // color indicator
                                                const ind = document.getElementById('indicator');
                                                ind.style.background = (t >= 38) ? 'linear-gradient(90deg,var(--bad),#ff7b7b)' : (t >= 37) ? 'linear-gradient(90deg,var(--warn),#ffd08a)' : 'linear-gradient(90deg,var(--good),#9ff3c9)';
                                            }catch(e){
                                                console.debug('fetch error', e);
                                            }
                                        }
                                        setInterval(fetchTemp,1000);
                                        fetchTemp();
                                    </script>
                                </body>
                                </html>"""
                        )
            self.wfile.write(html.encode('utf-8'))
            return

        if parsed.path == "/api/max_temp":
            temp = 0.0
            if self.tcp_server is not None:
                temp = getattr(self.tcp_server, 'temperature', 0.0)
            payload = json.dumps({"max_temperature": temp})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload.encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def start_ui(http_host: str = "0.0.0.0", http_port: int = 8080, tcp_server_instance: TCPServer = None):
    """Start a threaded HTTP server that exposes the UI and API."""
    handler = StatusHTTPRequestHandler
    handler.tcp_server = tcp_server_instance
    try:
        httpd = http.server.ThreadingHTTPServer((http_host, http_port), handler)
    except AttributeError:
        # Fallback for older Python versions
        httpd = http.server.HTTPServer((http_host, http_port), handler)

    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[UI] HTTP UI available at http://{http_host}:{http_port}/")
    return httpd


if __name__ == "__main__":
    server = TCPServer(host="192.168.1.133", port=9060)

    # 讓 server.start() 在背景 thread 跑，不要卡住主程式
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    poller = VivotekThermalPoller(
        ip="169.254.183.33",
        username="root",
        password="admin",
        poll_interval=2.0,
        timeout=10,
        print_full=False
    )
    # Start the simple HTTP UI that shows the current max temperature
    ui_server = start_ui(http_host="0.0.0.0", http_port=8080, tcp_server_instance=server)
    try:
        while True:
            # 每1秒主動發一筆封包給目前 client
            time.sleep(1.0)
            poller.poll_once()
            server.temperature = poller.maxTemperature
            server.send_temperature(server.temperature)

    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt, closing server...")
        server.close()
        try:
            if ui_server is not None:
                ui_server.shutdown()
                print("[UI] HTTP UI server shut down.")
        except Exception:
            pass
