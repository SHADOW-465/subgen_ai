"""
Test ONLY the winreg fast path — no pyserial import at all.
This verifies the registry scan returns COM9 in milliseconds.
"""
import winreg, re, time

_ESP32_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A}

t0 = time.time()
results = []
usb_key_path = r"SYSTEM\CurrentControlSet\Enum\USB"

with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, usb_key_path) as usb_key:
    n_vid = winreg.QueryInfoKey(usb_key)[0]
    for i in range(n_vid):
        try:
            vid_pid_name = winreg.EnumKey(usb_key, i)
            m = re.match(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})", vid_pid_name)
            if not m:
                continue
            vid = int(m.group(1), 16)
            pid = int(m.group(2), 16)
            if vid not in _ESP32_VIDS:
                continue
            with winreg.OpenKey(usb_key, vid_pid_name) as vp_key:
                n_serial = winreg.QueryInfoKey(vp_key)[0]
                for j in range(n_serial):
                    try:
                        serial_name = winreg.EnumKey(vp_key, j)
                        dp_path = f"{serial_name}\\Device Parameters"
                        try:
                            with winreg.OpenKey(vp_key, dp_path) as dp_key:
                                port_name, _ = winreg.QueryValueEx(dp_key, "PortName")
                                results.append({
                                    "device": port_name,
                                    "vid": hex(vid),
                                    "pid": hex(pid),
                                    "serial": serial_name,
                                })
                        except FileNotFoundError:
                            pass
                    except OSError:
                        pass
        except OSError:
            pass

elapsed = time.time() - t0
print(f"Registry scan completed in {elapsed*1000:.1f} ms")
print(f"Found {len(results)} ESP32-compatible USB port(s):")
for r in results:
    print(f"  {r['device']}  VID:{r['vid']}  PID:{r['pid']}  Serial:{r['serial']}")
