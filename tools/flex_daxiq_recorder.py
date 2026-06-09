#!/usr/bin/env python3
import socket
import struct
import sys
import time
import wave
import threading

# ===== USER CONFIG =====
CHANNEL_CONFIG = {
    1: 14.070e6,  # Channel 1 -> 14.070 MHz
    2: 7.040e6,   # Channel 2 -> 7.040 MHz
}
RECORD_SECONDS = 10
# ========================

# DAXIQ TCP port mapping
DAXIQ_PORTS = {1: 4991, 2: 4992, 3: 4993, 4: 4994}

HEADER_FORMAT = ">4sIHHII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

DISCOVERY_PORT = 4992
DISCOVERY_BROADCAST = ("255.255.255.255", 4992)

def discover_flexradio(timeout=3):
    """Discover FlexRadio via UDP broadcast."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    sock.sendto(b"discovery\n", DISCOVERY_BROADCAST)

    try:
        data, addr = sock.recvfrom(1024)
        ip = addr[0]
        print(f"Discovered FlexRadio at {ip}")
        return ip
    except socket.timeout:
        print("No FlexRadio found on network.")
        sys.exit(1)

def connect_control(ip):
    """Connect to FlexRadio control API."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, DISCOVERY_PORT))
    sock.settimeout(2.0)
    return sock

def send_command(sock, cmd):
    """Send a command to the radio."""
    msg = f"C1|{cmd}\n"
    sock.sendall(msg.encode("utf-8"))

def enable_daxiq(sock, channel, freq):
    """Enable DAXIQ stream."""
    send_command(sock, f"display pan create daxiq={channel}")
    time.sleep(0.3)
    send_command(sock, f"slice tune {channel-1} {freq}")
    time.sleep(0.3)
    send_command(sock, f"daxiq set {channel} state=1")
    print(f"[CH{channel}] Enabled at {freq/1e6:.6f} MHz")

def disable_daxiq(sock, channel):
    """Disable DAXIQ stream."""
    send_command(sock, f"daxiq set {channel} state=0")

    print(f"[CH{channel}] Disabled.")

def connect_daxiq(ip, channel):
    """Connect to DAXIQ TCP stream."""
    port = DAXIQ_PORTS[channel]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    sock.settimeout(5.0)
    return sock

def record_channel(ip, channel, freq, duration):
    """Record one DAXIQ channel to WAV."""
    try:
        iq_sock = connect_daxiq(ip, channel)
        filename = f"iq_ch{channel}_{int(freq/1e3)}kHz.wav"
        end_time = time.time() + duration
        sample_rate = None

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(2)  # I and Q
            wf.setsampwidth(4)  # 32-bit float
            wf.setcomptype("NONE", "not compressed")

            while time.time() < end_time:
                header_data = iq_sock.recv(HEADER_SIZE)
                if len(header_data) < HEADER_SIZE:
                    break
                magic, stream_id, seq, reserved, payload_size, sr = struct.unpack(HEADER_FORMAT, header_data)
                if magic != b"iqst":
                    continue
                if sample_rate is None:
                    sample_rate = sr
                    wf.setframerate(sample_rate)
                    print(f"[CH{channel}] Sample rate: {sample_rate} Hz")

                payload = b""
                while len(payload) < payload_size:
                    chunk = iq_sock.recv(payload_size - len(payload))
                    if not chunk:
                        break
                    payload += chunk

                if len(payload) == payload_size:
                    wf.writeframes(payload)

        print(f"[CH{channel}] Recording saved to {filename}")
        iq_sock.close()

    except Exception as e:
        print(f"[CH{channel}] Error: {e}")

def main():
    try:
        # Auto-discover radio
        radio_ip = discover_flexradio()

        # Connect to control API
        ctrl_sock = connect_control(radio_ip)
        print("Connected to FlexRadio control API.")

        # Enable all configured channels
        for ch, freq in CHANNEL_CONFIG.items():
            enable_daxiq(ctrl_sock, ch, freq)

        # Start recording threads
        threads = []
        for ch, freq in CHANNEL_CONFIG.items():
            t = threading.Thread(target=record_channel, args=(radio_ip, ch, freq, RECORD_SECONDS))
            t.start()
            threads.append(t)

        # Wait for all recordings to finish
        for t in threads:
            t.join()

        # Disable all channels
        for ch in CHANNEL_CONFIG.keys():
            disable_daxiq(ctrl_sock, ch)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            ctrl_sock.close()
        except:
            pass
        
if __name__ == "__main__":
    main()