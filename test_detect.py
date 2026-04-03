"""Test ESP32 detection with the updated fast-scan logic."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from subgen_ai.core.esp32_validator import find_all_esp32_ports, list_all_ports

print("Running fast port scan...")
import time
t0 = time.time()
esp_ports = find_all_esp32_ports()
elapsed = time.time() - t0
print(f"Scan completed in {elapsed:.2f}s")
print(f"ESP32 ports detected: {esp_ports}")

print("\nAll ports (raw):")
for p in list_all_ports():
    print(f"  {p['device']} | {p['description']} | VID:{p['vid']} | {p['hwid'][:40]}")
