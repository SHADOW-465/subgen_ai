"""Quick port checker with a timeout approach."""
import serial.tools.list_ports

print("Scanning all COM ports...")
ports = serial.tools.list_ports.comports()
for p in ports:
    vid = hex(p.vid) if p.vid else "None"
    pid = hex(p.pid) if p.pid else "None"
    print(f"  {p.device} | {p.description} | VID:{vid} PID:{pid} | {p.hwid}")

print("Done.")
