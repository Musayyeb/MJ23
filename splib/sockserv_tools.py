import splib.sound_tools as st
import sounddevice as sd
import time
import socket

port = 8998

class SocketServer:
    def __init__(self, msg_handler):
        self.msg_handler = msg_handler

    def run(self):
        print(f'\n\nSound server started - listening at port {port}\n')

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', port)
        s.bind(server_address)

        s.listen()

        while True:  # to be cancelled by terminal request

            conn, client_addr = s.accept()

            try:
                # Receive the data in small chunks and retransmit it
                while True:
                    msg = receive(conn)   # blocking, waiting for request
                    tup = msg.split(None,1)
                    cmd = tup[0]

                    if cmd == 'term':
                        break

                    data = tup[1] if len(tup) > 1 else ''

                    resp = self.msg_handler(cmd, data)

                    reply(conn, resp)

            finally:
                # Clean up the connection
                conn.close()


def soundserv_request(req_string):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', port)
        s.connect(server_address)
    except ConnectionRefusedError:
        if req_string.startswith("start"):
            print("no soundserver")
        return

    print("request", req_string)
    data = req_string.encode()
    dlen = f"{len(data):04d}".encode()
    s.send(dlen)
    s.send(data)

    resp = receive(s)

    s.send('0004term'.encode())
    s.close()
    print('response', resp)
    return resp

def receive(conn):
    # data = length {04d} + bytes
    byt1 = conn.recv(4)
    if not byt1:
        return ''
    dlen = int(byt1.decode())
    byt2 = conn.recv(dlen)
    data = byt2.decode()
    print("received", data)
    return data

def reply(conn, msg):
    print('reply', msg)
    data = msg.encode()
    dlen = f"{len(data):04d}".encode()
    conn.send(dlen)
    conn.send(data)
    return


class SoundDevice:
    # a wrapper for the sounddevice module
    def __init__(self):
        self.status = False
        sd.default.samplerate = 24000  # default for sounddevice
        self.wav = []

    def load(self, data):
        # expect a full filename
        print(f"load wav file: {data}")
        fr, wav = st.read_wav_file(data)
        self.wav = wav
        assert fr == 24000, f"fr={fr}"
        size = 2*len(wav) / (1024*1024)  # a frame is 2 bytes
        print(f'SD loaded {size:6.2f} MB')
        return ''

    def play(self, data):
        self.load(data)  # data is a file name
        wav = self.wav
        print(f'loaded wav: {type(wav)}')
        time.sleep(0.1)
        sd.play(wav)
        time.sleep(0.1)
        self.satus = True
        return "running"

    def start(self, data):
        if not len(self.wav):
            return 'SD no wav loaded'

        ms = int(data)  # expect a number = milliseconds offset
        frm = ms * 24
        resp = f'SD started at {ms} ms'
        if self.status:
            st.stop()
            time.sleep(0.3)
            resp = 're' + resp
        if frm+1000 > len(self.wav):
            return 'position beyond end of audio'
        sd.play(self.wav[frm:])
        self.satus = True
        return resp

    def stop(self):
        sd.stop()
        if self.status:
            resp = 'SD stopped'
            status = False
        else:
            resp = "SD didn't run"
        return resp

if __name__ == '__main__':
    print('this module is for importing')
