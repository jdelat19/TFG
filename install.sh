#!/bin/bash
# Instala todo lo necesario para el proyecto en Ubuntu 20.04, con las mismas
# versiones que el PC donde se desarrolló:
#   1. ROS Noetic (desktop-full) y librerías del sistema (apt)
#   2. Un entorno virtual de Python en .venv con las versiones exactas de
#      requirements-lock.txt
#   3. Compila el workspace de catkin para que los nodos usen ese .venv
#
# Uso (desde la carpeta del repositorio):  ./install.sh
# Se puede volver a ejecutar: lo que ya está instalado se salta.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

step() { echo -e "\n\033[1;36m==> $*\033[0m"; }

if [ "$(lsb_release -rs 2>/dev/null)" != "20.04" ]; then
    echo "Este script es para Ubuntu 20.04 (ROS Noetic no se instala en otras versiones)."
    exit 1
fi

# --- 1. apt ------------------------------------------------------------------
if [ ! -f /etc/apt/sources.list.d/ros-latest.list ]; then
    step "Añadiendo el repositorio de ROS"
    sudo apt-get update
    sudo apt-get install -y curl gnupg lsb-release
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
fi

step "Instalando ROS Noetic y paquetes del sistema (apt)"
sudo apt-get update
sudo apt-get install -y \
    ros-noetic-desktop-full \
    python3-venv python3-pip \
    python3-opengl python3-yaml python3-oauthlib \
    libportaudio2 ffmpeg v4l-utils

# --- 2. Entorno virtual de Python ---------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    step "Creando el entorno virtual en .venv"
    # --system-site-packages: el venv ve rospy, catkin, etc. que instala apt
    python3 -m venv --system-site-packages "$VENV"
fi

USER_SITE="$HOME/.local/lib/python3.8/site-packages"
if [ -d "$USER_SITE" ] && [ -n "$(ls -A "$USER_SITE" 2>/dev/null)" ]; then
    echo -e "\033[1;33mAviso:\033[0m hay paquetes instalados con 'pip install --user' en $USER_SITE."
    echo "Python los carga antes que los del .venv y pueden cambiar las versiones."
    echo "Si algo falla, muévelos a otra carpeta o desinstálalos con 'pip3 uninstall'."
fi

step "Instalando los paquetes de Python (versiones exactas)"
# El pip de Python 3.8 es muy antiguo para algunas ruedas (mediapipe, onnxruntime)
"$VENV/bin/pip" install --upgrade "pip<25"
# --no-deps: la lista ya tiene todas las dependencias con su versión. Sin él pip
# se niega, porque tensorflow 2.13 declara typing_extensions<4.6 y torch pide
# >=4.8; en la práctica funciona con la 4.13.2, como en el PC original
"$VENV/bin/pip" install --no-deps -r "$ROOT/requirements-lock.txt"
# El cv2 que se usa es el de opencv-contrib-python (lo pide mediapipe); se
# instala el último porque comparte carpeta con opencv-python
"$VENV/bin/pip" install --no-deps --force-reinstall opencv-contrib-python==4.13.0.92

# --- 3. Workspace de catkin -----------------------------------------------------
step "Compilando el workspace"
source /opt/ros/noetic/setup.bash
cd "$ROOT"
rm -rf build devel
# Los nodos se generan con el Python del venv: no hace falta activarlo
catkin_make -DPYTHON_EXECUTABLE="$VENV/bin/python"

LINE="source $ROOT/devel/setup.bash"
if ! grep -qxF "$LINE" ~/.bashrc; then
    step "Añadiendo ROS y el workspace a ~/.bashrc"
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    echo "$LINE" >> ~/.bashrc
fi

step "Comprobando que todo carga"
"$VENV/bin/python" - <<'EOF'
import cv2, mediapipe, numpy, onnxruntime, tensorflow, torch, transformers, whisper, sounddevice, glfw, rospy
print("cv2", cv2.__version__, "| numpy", numpy.__version__, "| torch", torch.__version__,
      "| tensorflow", tensorflow.__version__)
EOF
(cd "$ROOT/src/human_detection" && "$VENV/bin/python" -m pytest -q test/)

echo -e "\n\033[1;32mListo.\033[0m Abre una terminal nueva y lanza:  rosrun human_detection launch_menu.py"
