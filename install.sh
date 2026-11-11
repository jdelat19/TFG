#!/bin/bash
# =====================================================================================================
# install.sh - Instala todo lo necesario en un PC con Ubuntu 20.04, desde cero
#
#   1. ROS Noetic (desktop-full, con Gazebo) y paquetes del sistema (apt)
#   2. Entorno virtual de Python en .venv con las versiones exactas de requirements-lock.txt
#   3. Descarga de los modelos de IA (emoción facial y voz), para no esperar en la primera ejecución
#   4. Compilación del workspace de catkin (los nodos usan el Python de .venv sin activarlo)
#   5. ~/.bashrc: carga ROS y el workspace y, si se quiere, la red con el TurtleBot2 real
#   6. Comprobación final: importa todas las librerías y pasa los tests
#
# Uso (desde la carpeta del repositorio):
#   ./install.sh                 instalación completa
#   ./install.sh --sin-modelos   sin descargar los modelos (se bajan solos al usarlos)
#
# Se puede volver a ejecutar sin problema: lo que ya está instalado se salta.
# Pide la contraseña de sudo para apt. Descarga unos 4 GB en total.
# =====================================================================================================
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
DOWNLOAD_MODELS=true
[ "${1:-}" = "--sin-modelos" ] && DOWNLOAD_MODELS=false

step() { echo -e "\n\033[1;36m==> $*\033[0m"; }
warn() { echo -e "\033[1;33mAviso:\033[0m $*"; }
fail() { echo -e "\033[1;31mError:\033[0m $*" >&2; exit 1; }
trap 'echo -e "\n\033[1;31mLa instalación se ha detenido en el último paso.\033[0m Revisa el mensaje de arriba y vuelve a lanzar ./install.sh"' ERR

# --- 0. Requisitos ----------------------------------------------------------------------------------
[ "$(id -u)" -ne 0 ] || fail "No lo ejecutes con sudo: ./install.sh (ya pedirá la contraseña cuando haga falta)"
[ "$(lsb_release -rs 2>/dev/null)" = "20.04" ] \
    || fail "Hace falta Ubuntu 20.04: ROS Noetic no se puede instalar en otras versiones"
case "$ROOT" in *" "*) fail "La ruta del proyecto tiene espacios ($ROOT); catkin no lo admite. Muévelo a otra carpeta" ;; esac

# --- 1. apt -----------------------------------------------------------------------------------------
if [ ! -f /etc/apt/sources.list.d/ros-latest.list ]; then
    step "Añadiendo el repositorio de ROS"
    sudo apt-get update
    sudo apt-get install -y curl gnupg lsb-release
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
fi

step "Instalando ROS Noetic y paquetes del sistema (apt, puede tardar un rato)"
sudo apt-get update
# - ros-noetic-desktop-full: ROS, Gazebo, xacro, tf2...
# - ros-noetic-kobuki-msgs: mensajes del bumper del TurtleBot2
# - libportaudio2: micrófono (sounddevice); ffmpeg: audio de Whisper
# - terminator, sshpass, openssh-client: los usa run.sh para arrancar el robot real
sudo apt-get install -y \
    ros-noetic-desktop-full ros-noetic-kobuki-msgs \
    python3-venv python3-pip \
    python3-opengl python3-yaml python3-oauthlib \
    libportaudio2 ffmpeg v4l-utils \
    terminator sshpass openssh-client

# --- 2. Entorno virtual de Python -------------------------------------------------------------------
if [ ! -x "$PY" ]; then
    step "Creando el entorno virtual en .venv"
    # --system-site-packages: el venv ve rospy, catkin, etc. que instala apt
    python3 -m venv --system-site-packages "$VENV"
fi

USER_SITE="$HOME/.local/lib/python3.8/site-packages"
if [ -d "$USER_SITE" ] && [ -n "$(ls -A "$USER_SITE" 2>/dev/null)" ]; then
    warn "hay paquetes instalados con 'pip install --user' en $USER_SITE."
    echo "Python los carga antes que los del .venv y pueden cambiar las versiones."
    echo "Si algo falla, muévelos a otra carpeta o desinstálalos con 'pip3 uninstall'."
fi

step "Instalando los paquetes de Python con las versiones exactas (unos 3 GB)"
# El pip de Python 3.8 es muy antiguo para algunas ruedas (mediapipe, onnxruntime)
"$PY" -m pip install --upgrade "pip<25"
# --no-deps: la lista ya tiene todas las dependencias con su versión. Sin él pip
# se niega, porque tensorflow 2.13 declara typing_extensions<4.6 y torch pide
# >=4.8; en la práctica funciona con la 4.13.2, como en el PC original
"$PY" -m pip install --no-deps -r "$ROOT/requirements-lock.txt"
# El cv2 que se usa es el de opencv-contrib-python (lo pide mediapipe); se
# instala el último porque comparte carpeta con opencv-python
"$PY" -m pip install --no-deps --force-reinstall opencv-contrib-python==4.13.0.92

# --- 3. Modelos de IA -------------------------------------------------------------------------------
if $DOWNLOAD_MODELS; then
    step "Descargando los modelos de IA (HSEmotion, Whisper, RoBERTa, wav2vec 2.0; ~1 GB)"
    # Si falla (sin internet, por ejemplo) no pasa nada: se descargan al usarlos
    "$PY" - "$ROOT" <<'EOF' || warn "no se pudieron descargar todos los modelos; se descargarán al usarlos"
import os, sys, urllib.request
sys.path.insert(0, os.path.join(sys.argv[1], "src", "human_detection", "src"))
from facial_expression import HSEMOTION_PATH, HSEMOTION_URL
if not os.path.isfile(HSEMOTION_PATH):
    os.makedirs(os.path.dirname(HSEMOTION_PATH), exist_ok=True)
    urllib.request.urlretrieve(HSEMOTION_URL, HSEMOTION_PATH)
print("HSEmotion:", HSEMOTION_PATH)
import whisper
whisper.load_model("base"); print("Whisper base")
from transformers import pipeline, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion"); print("RoBERTa")
Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er"); print("wav2vec 2.0")
EOF
fi

# --- 4. Workspace de catkin -------------------------------------------------------------------------
step "Compilando el workspace"
set +e; source /opt/ros/noetic/setup.bash; set -e
cd "$ROOT"
rm -rf build devel
# Los nodos se generan con el Python del venv: no hace falta activarlo
catkin_make -DPYTHON_EXECUTABLE="$PY"

# --- 5. ~/.bashrc -----------------------------------------------------------------------------------
LINE="source $ROOT/devel/setup.bash"
if ! grep -qxF "$LINE" ~/.bashrc; then
    step "Añadiendo ROS y el workspace a ~/.bashrc"
    grep -qxF "source /opt/ros/noetic/setup.bash" ~/.bashrc || echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    echo "$LINE" >> ~/.bashrc
fi

# Red con el TurtleBot2 real (opcional). El robot es el master de ROS (run.sh
# arranca minimal.launch en él), así que en este PC:
#   ROS_MASTER_URI=http://IP_DEL_ROBOT:11311   ROS_IP=ROS_HOSTNAME=IP_DE_ESTE_PC
if [ -t 0 ] && ! grep -q "^export ROS_MASTER_URI=" ~/.bashrc; then
    step "Red con el TurtleBot2 real (opcional)"
    echo "Si vas a usar el robot real, escribe la IP de su portátil (hostname -I en el robot)."
    read -r -p "IP del TurtleBot2 [Enter para saltar; solo simulación]: " ROBOT_IP
    if [ -n "$ROBOT_IP" ]; then
        PC_IP=$(ip route get "$ROBOT_IP" 2>/dev/null | grep -oP 'src \K[0-9.]+' || true)
        read -r -p "IP de este PC en la misma red [${PC_IP:-?}]: " ANSWER
        PC_IP=${ANSWER:-$PC_IP}
        if [ -n "$PC_IP" ]; then
            {
                echo "# TurtleBot2 real (install.sh): el robot es el master de ROS"
                echo "export ROS_MASTER_URI=http://$ROBOT_IP:11311"
                echo "export ROS_HOSTNAME=$PC_IP"
                echo "export ROS_IP=$PC_IP"
            } >> ~/.bashrc
            echo "Añadido a ~/.bashrc. Para cambiarlo, edita esas tres líneas."
        else
            warn "sin la IP del PC no se configura la red; mira el README (Robot real)"
        fi
    fi
fi

# --- 6. Comprobación ----------------------------------------------------------------------------------
step "Comprobando que todo carga"
"$PY" - <<'EOF'
import cv2, mediapipe, numpy, onnxruntime, torch, transformers, whisper, sounddevice, glfw, rospy
print("cv2", cv2.__version__, "| mediapipe", mediapipe.__version__, "| numpy", numpy.__version__,
      "| torch", torch.__version__)
EOF
(cd "$ROOT/src/human_detection" && "$PY" -m pytest -q test/)

trap - ERR
echo -e "\n\033[1;32mInstalación completa.\033[0m Abre una terminal NUEVA y lanza:"
echo "    rosrun human_detection launch_menu.py     (avatar y simulación)"
echo "    bash run.sh                               (TurtleBot2 real)"
