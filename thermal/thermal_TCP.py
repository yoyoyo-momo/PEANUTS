import socket
import threading
import time
import random


class ThermalClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.recv_thread = None
        self.min = 0
        self.sec = 0
        self.cur_temp = 0

    def connect(self):
        """連線到 Server"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.running = True
        print(f"[INFO] Connected to {self.host}:{self.port}")

        # 啟動接收執行緒
        self.recv_thread = threading.Thread(
            target=self._receive_loop, daemon=True)
        self.recv_thread.start()

    def _receive_loop(self):
        """持續接收 8-byte 資料"""
        while self.running:
            try:
                data = self.sock.recv(8)  # 固定接收 8 bytes
                print("[RECV]", ' '.join(f"{b:02X}" for b in data))
                if not data:
                    print("[INFO] Connection closed by server.")
                    break

                # 解析資料
                self.data_parse(data)

            except ConnectionResetError:
                print("[ERROR] Connection reset by server.")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                break

        self.running = False
        self.sock.close()

    def data_parse(self, data):
        # print("Received data:", data)
        if data[0] == 136 and data[1] == 102:  # 0x88 & 0x66
            if not self.sum_check(data):
                print("Checksum invalid.")
            else:
                match data[2]:
                    case 79:  # 'O'
                        print("Received 'O' command.")
                        left_min = self.min.to_bytes(1, 'little')
                        left_sec = self.sec.to_bytes(1, 'little')
                        send_data = b'\x88\x66\x6F' + left_min + left_sec + b'\x00\x00'
                        self.check_sum(send_data)
                    case 113:  # 'q'
                        print("Received 'q' command.")
                        cur_temp = int(data[3])
                        self.cur_temp = cur_temp
                        print(f"Current Max Temperature: {self.cur_temp}")
                        # send_data = b'\x88\x66\x71' + left_min + left_sec + b'\x00\x00'
                        # self.check_sum(send_data)
        else:
            print("Invalid header.")

    def sum_check(self, data):
        data_array = list(data)
        sum = 0
        for i in range(8):
            sum = sum+data_array[i]
        if (sum & 0xff) == 0:
            return True
        else:
            return False

    def check_sum(self, data):
        data_array = list(data)
        sum = 0
        for i in range(7):
            sum = sum+data_array[i]
        check_sum = (0-sum) & 0xff
        data += bytes([check_sum])
        self.send_data(data)

    def send_data(self, data: bytes):
        self.sock.sendall(data)

    def send_end(self):
        send_data = b'\x88\x66\x65\x00\x00\x00\x00'
        self.check_sum(send_data)

    def left_time(self, min, sec):
        self.min = min
        self.sec = sec

    def send_temp_req(self):
        send_data = b'\x88\x66\x51\x00\x00\x00\x00'
        self.check_sum(send_data)
        print("sending request")

    def close(self):
        """關閉連線"""
        self.running = False
        if self.sock:
            self.sock.close()
        print("[INFO] Disconnected.")
    
    def get_cur_temp(self):
        return self.cur_temp


if __name__ == "__main__":
    # === 使用範例 ===
    #client = ThermalClient("192.168.10.101", 9000)
    client = ThermalClient("192.168.1.133", 9000)
    client.connect()
    try:
        while client.running:
            # min = random.randint(0, 255)  # 產生 0~255 隨機整數
            # sec = random.randint(0, 59)  # 產生 0~59 隨機整數
            # client.left_time(min, sec)
            # client.send_temp_req()
            time.sleep(1)
    except KeyboardInterrupt:
        client.send_end()
        time.sleep(0.1)
        client.close()
