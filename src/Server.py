import socket
import threading
import time
from typing import Optional, Tuple


class TCPServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port

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

    def send_empty(self) -> None:
        """
        範例封包：
        Header 0x88 0x66
        Command 0x61
        後面 4 bytes 先填 0
        最後一 byte 由 check_sum_and_send() 自動補上
        """
        base = b'\x88\x66\x61\x01\x00\x00\x00'  # 7 bytes
        self.check_sum_and_send(base)
        print("CUP IS EMPTY")
    
    def send_not_empty(self) -> None:
        """
        範例封包：
        Header 0x88 0x66
        Command 0x61
        後面 4 bytes 先填 0
        最後一 byte 由 check_sum_and_send() 自動補上
        """
        base = b'\x88\x66\x61\x00\x00\x00\x00'  # 7 bytes
        self.check_sum_and_send(base)
        print("CUP IS NOT EMPTY")


if __name__ == "__main__":
    server = TCPServer(host="0.0.0.0", port=9000)

    # 讓 server.start() 在背景 thread 跑，不要卡住主程式
    t = threading.Thread(target=server.start, daemon=True)
    t.start()

    try:
        while True:
            # 每5秒主動發一筆封包給目前 client
            time.sleep(5.0)
            server.send_alert()
    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt, closing server...")
        server.close()
