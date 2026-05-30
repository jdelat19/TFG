#!/usr/bin/env python3

import subprocess
import sys

import sounddevice as sd

# print("\n===== DISPOSITIVOS DE AUDIO =====\n")

# for idx, dev in enumerate(sd.query_devices()):
#     if dev["max_input_channels"] > 0:
#         print(
#             f"{idx}: {dev['name']} "
#             f"(inputs={dev['max_input_channels']})"
#         )

print("\n===============================\n")
modes = {
    "1": "Modo 1: Cara + imágenes",
    "2": "Modo 2: Voz + cara + imágenes",
    "3": "Modo 3: Voz + cara + vídeos",
    "4": "Modo 4: Cara + vídeos gestos",
}

print("\nSelecciona modo:\n")

for k, v in modes.items():
    print(f"{k}. {v}")

mode = input("\nModo [1-4]: ").strip()

if mode not in modes:
    print("Modo inválido")
    sys.exit(1)

use_voice = "true" if mode in ["2", "3"] else "false"

subprocess.call([
    "roslaunch",
    "human_detection",
    "human_detection.launch",
    f"mode:={mode}",
    f"use_voice:={use_voice}",
])