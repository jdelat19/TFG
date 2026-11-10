#!/usr/bin/env bash
# =====================================================================================================
# run.sh - Arranca todo el sistema en una ventana de Terminator con 5 terminales
#
#   +-----------------------------+--------------------------------------------+
#   | 1. rosrun human_detection   | 4. [SSH] roslaunch turtlebot_bringup      |
#   |    launch_menu.py           |          3dsensor.launch                   |
#   +-----------------------------+--------------------------------------------+
#   | 2. rostopic echo velocity   | 5. [SSH] roslaunch turtlebot_bringup      |
#   +-----------------------------+          minimal.launch                    |
#   | 3. rostopic hz /odom        |                                            |
#   +-----------------------------+--------------------------------------------+
#
# Uso:  bash run.sh
#
# Pide el usuario, la IP y la contraseña del portátil del TurtleBot2 y luego:
#   1. Comprueba la conexión SSH                             (falla si no puede entrar)
#   2. Abre Terminator, la terminal 5 lanza minimal.launch   (arranca también el master de ROS en el robot)
#   3. Cuando responde el master, la terminal 4 lanza 3dsensor.launch
#   4. Comprueba que el robot está inicializado: /odom, mux de velocidad, cámara y /scan
#                                                            (falla si alguno no aparece a tiempo)
#   5. Arranca el menú del proyecto y los dos monitores (terminales 1, 2 y 3)
# Si algo falla se cierran la ventana de Terminator y los roslaunch del robot.
#
# Las variables de red de ROS (ROS_MASTER_URI, ROS_HOSTNAME / ROS_IP) salen de ~/.bashrc, ver README.
# =====================================================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$PROJECT_DIR/run.sh"
WORKSPACE_SETUP="$PROJECT_DIR/devel/setup.bash"

# Tiempos máximos en segundos
MASTER_TIMEOUT=40           # minimal.launch arranca el master de ROS
ODOM_TIMEOUT=30             # La base Kobuki publica /odom en cuanto se conecta
CAMERA_TIMEOUT=60           # 3dsensor.launch es lento (driver de la cámara + profundidad a láser)
SCAN_TIMEOUT=20             # /scan sale de la cámara de profundidad (depthimage_to_laserscan)

# Topics que demuestran que el robot está inicializado
ODOM_TOPIC="/odom"
VELOCITY_TOPIC="/mobile_base/commands/velocity"
MUX_TOPIC="/cmd_vel_mux/input/teleop"
CAMERA_TOPIC="/camera/rgb/image_raw"
SCAN_TOPIC="/scan"
ODOM_MIN_RATE=40            # Hz, por debajo solo es un aviso (lo normal: unos 50 Hz)

SSH_OPTIONS=(-o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=5 -o ServerAliveCountMax=3)

# ---------------------------------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------------------------------

info()  { echo -e "\e[1;34m[run]\e[0m $*"; }
ok()    { echo -e "\e[1;32m[ OK]\e[0m $*"; }
warn()  { echo -e "\e[1;33m[WARN]\e[0m $*"; }
error() { echo -e "\e[1;31m[FAIL]\e[0m $*" >&2; }

# Carga ROS si la shell actual no lo tiene (por ejemplo al lanzarlo desde el explorador de archivos)
load_ros() {
    if [ -z "${ROS_DISTRO:-}" ] && [ -f /opt/ros/noetic/setup.bash ]; then
        source /opt/ros/noetic/setup.bash
    fi
    if [ -f "$HOME/turtlebot2_ws/devel/setup.bash" ] && [[ ":${ROS_PACKAGE_PATH:-}:" != *turtlebot2_ws* ]]; then
        source "$HOME/turtlebot2_ws/devel/setup.bash"
    fi
}

# Carga el workspace de este proyecto (paquete human_detection)
load_workspace() {
    source "$WORKSPACE_SETUP" --extend
}

# Ejecuta un comando en el robot por SSH, con terminal para que Ctrl+C llegue a roslaunch
# La shell de login remota carga el entorno ROS del robot (su ~/.profile y ~/.bashrc)
ssh_robot() {
    sshpass -e ssh -t "${SSH_OPTIONS[@]}" "$TFG_USER@$TFG_IP" "bash -lic '$1; exec bash -i'"
}

# Espera a que exista el fichero $TFG_STATE/$1. Devuelve 1 si el arranque ha fallado mientras
wait_flag() {
    until [ -e "$TFG_STATE/$1" ]; do
        if [ -e "$TFG_STATE/failed" ]; then
            error "Arranque cancelado, mira la terminal desde la que se lanzó run.sh"
            return 1
        fi
        sleep 0.5
    done
}

# Deja la terminal abierta con una shell interactiva cuando termina su comando
keep_open() {
    unset SSHPASS
    exec bash
}

master_online() { rostopic list >/dev/null 2>&1; }
topic_listed()  { rostopic list 2>/dev/null | grep -qx "$1"; }

# Espera hasta $1 segundos a que la función $2 (con argumentos) se cumpla
wait_for() {
    local timeout=$1; shift
    local end=$((SECONDS + timeout))
    until "$@"; do
        [ $SECONDS -ge $end ] && return 1
        sleep 1
    done
}

odom_received() { timeout 5 rostopic echo -n 1 "$ODOM_TOPIC" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------------------------------
# Comandos de cada terminal, run.sh se llama a sí mismo con --pane <nombre> dentro de cada una
# ---------------------------------------------------------------------------------------------------

run_pane() {
    cd "$PROJECT_DIR"
    load_ros
    case "$1" in
        minimal)
            info "Base del TurtleBot2: roslaunch turtlebot_bringup minimal.launch ($TFG_USER@$TFG_IP)"
            ssh_robot "roslaunch turtlebot_bringup minimal.launch"
            ;;
        sensor)
            info "Esperando al master de ROS antes de arrancar la cámara"
            wait_flag master && ssh_robot "roslaunch turtlebot_bringup 3dsensor.launch"
            ;;
        main)
            unset SSHPASS
            info "Esperando a que el TurtleBot2 esté inicializado"
            if wait_flag ready; then
                # Los nodos usan el Python de .venv (catkin_make con PYTHON_EXECUTABLE), no hace falta activarlo
                load_workspace
                rosrun human_detection launch_menu.py
                # Shell interactiva con el workspace cargado, para volver a lanzar el menú
                exec bash --rcfile <(echo "source ~/.bashrc; source '$WORKSPACE_SETUP' --extend")
            fi
            ;;
        echo)
            unset SSHPASS
            info "Esperando a que el TurtleBot2 esté inicializado"
            wait_flag ready && rostopic echo "$VELOCITY_TOPIC"
            ;;
        hz)
            unset SSHPASS
            info "Esperando a que el TurtleBot2 esté inicializado"
            wait_flag ready && rostopic hz "$ODOM_TOPIC"
            ;;
    esac
    keep_open
}

if [ "${1:-}" = "--pane" ]; then
    run_pane "${2:-}"
fi

# ---------------------------------------------------------------------------------------------------
# 1. Requisitos
# ---------------------------------------------------------------------------------------------------

load_ros
for command in terminator sshpass ssh rostopic; do
    if ! command -v "$command" >/dev/null; then
        error "No se encuentra '$command'. Instálalo (sudo apt install $command) o ejecuta: ./install.sh"
        exit 1
    fi
done
if [ ! -x "$PROJECT_DIR/.venv/bin/python" ]; then
    error "No se encuentra el entorno de Python ($PROJECT_DIR/.venv). Ejecuta: ./install.sh"
    exit 1
fi
if [ ! -f "$WORKSPACE_SETUP" ]; then
    error "El workspace no está compilado ($WORKSPACE_SETUP). Ejecuta: ./install.sh"
    exit 1
fi

# ---------------------------------------------------------------------------------------------------
# 2. Credenciales del TurtleBot2
# ---------------------------------------------------------------------------------------------------

# Por defecto: el usuario habitual del robot y la IP del master de ROS de ~/.bashrc
default_ip=""
if [ -n "${ROS_MASTER_URI:-}" ]; then
    default_ip=$(echo "$ROS_MASTER_URI" | sed -E 's#^[a-z]+://([^:/]+).*#\1#')
fi

read -r -p "Usuario del TurtleBot2 [turtlebot]: " TFG_USER
TFG_USER=${TFG_USER:-turtlebot}
read -r -p "IP del TurtleBot2${default_ip:+ [$default_ip]}: " TFG_IP
TFG_IP=${TFG_IP:-$default_ip}
read -r -s -p "Contraseña de $TFG_USER@$TFG_IP: " SSHPASS
echo

if [ -z "$TFG_IP" ] || [ -z "$SSHPASS" ]; then
    error "La IP y la contraseña son obligatorias"
    exit 1
fi

# El robot es el master de ROS, ~/.bashrc tiene que apuntar a él
if [ -z "${ROS_MASTER_URI:-}" ]; then
    warn "ROS_MASTER_URI no está definido (ver README, ~/.bashrc), se usa http://$TFG_IP:11311"
    export ROS_MASTER_URI="http://$TFG_IP:11311"
elif [ "$default_ip" != "$TFG_IP" ]; then
    error "ROS_MASTER_URI es $ROS_MASTER_URI pero la IP del robot es $TFG_IP. Corrige ~/.bashrc (ver README)"
    exit 1
fi
if [ -z "${ROS_IP:-}${ROS_HOSTNAME:-}" ]; then
    warn "ROS_IP / ROS_HOSTNAME sin definir, puede que el robot no llegue a este PC (ver README)"
fi

export TFG_USER TFG_IP SSHPASS

# ---------------------------------------------------------------------------------------------------
# 3. Conexión SSH
# ---------------------------------------------------------------------------------------------------

info "Comprobando la conexión SSH con $TFG_USER@$TFG_IP"
ssh_error=$(sshpass -e ssh "${SSH_OPTIONS[@]}" "$TFG_USER@$TFG_IP" true 2>&1 >/dev/null)
status=$?
if [ $status -ne 0 ]; then
    case $status in
        5) error "Contraseña incorrecta para $TFG_USER@$TFG_IP" ;;
        6) error "Clave de host desconocida de $TFG_IP (conéctate una vez a mano: ssh $TFG_USER@$TFG_IP)" ;;
        *) error "No se pudo conectar con $TFG_USER@$TFG_IP: $(echo "$ssh_error" | tail -n 1)" ;;
    esac
    exit 1
fi
ok "Conexión SSH"

if master_online; then
    error "Ya hay un master de ROS en $TFG_IP (¿sesión anterior?). Para antes su roslaunch:"
    error "  ssh $TFG_USER@$TFG_IP \"pkill -INT -f roslaunch\""
    exit 1
fi

# ---------------------------------------------------------------------------------------------------
# 4. Terminator
# ---------------------------------------------------------------------------------------------------

TFG_STATE=$(mktemp -d /tmp/tfg_run.XXXXXX)
export TFG_STATE
chmod 700 "$TFG_STATE"

cat > "$TFG_STATE/terminator.conf" <<EOF
[global_config]
  suppress_multiple_term_dialog = True
[keybindings]
[profiles]
  [[default]]
    scrollback_lines = 5000
[layouts]
  [[tfg]]
    [[[window]]]
      type = Window
      parent = ""
    [[[columns]]]
      type = HPaned
      parent = window
      order = 0
      ratio = 0.5
    [[[left]]]
      type = VPaned
      parent = columns
      order = 0
      ratio = 0.5
    [[[main]]]
      type = Terminal
      parent = left
      order = 0
      command = bash '$SCRIPT' --pane main
    [[[monitors]]]
      type = VPaned
      parent = left
      order = 1
      ratio = 0.5
    [[[echo]]]
      type = Terminal
      parent = monitors
      order = 0
      command = bash '$SCRIPT' --pane echo
    [[[hz]]]
      type = Terminal
      parent = monitors
      order = 1
      command = bash '$SCRIPT' --pane hz
    [[[right]]]
      type = VPaned
      parent = columns
      order = 1
      ratio = 0.5
    [[[sensor]]]
      type = Terminal
      parent = right
      order = 0
      command = bash '$SCRIPT' --pane sensor
    [[[minimal]]]
      type = Terminal
      parent = right
      order = 1
      command = bash '$SCRIPT' --pane minimal
[plugins]
EOF

# -u: proceso propio (sin D-Bus), así las terminales heredan las variables de este script
setsid terminator -u -m -g "$TFG_STATE/terminator.conf" -l tfg >/dev/null 2>&1 &
TERMINATOR_PID=$!
info "Terminator abierto, arrancando el TurtleBot2"

# Cierra todo y sale con error
fail() {
    error "$*"
    touch "$TFG_STATE/failed"
    read -r -p "Pulsa Enter para cerrar las terminales (la salida de roslaunch se ve hasta entonces)... " _
    sshpass -e ssh "${SSH_OPTIONS[@]}" "$TFG_USER@$TFG_IP" "pkill -INT -f 'roslaunch turtlebot_bringup'" >/dev/null 2>&1
    kill "$TERMINATOR_PID" 2>/dev/null
    rm -rf "$TFG_STATE"
    exit 1
}
trap 'fail "Interrumpido con Ctrl+C"' INT

# ---------------------------------------------------------------------------------------------------
# 5. Inicialización del TurtleBot2
# ---------------------------------------------------------------------------------------------------

info "Esperando al master de ROS en $TFG_IP (minimal.launch)"
wait_for "$MASTER_TIMEOUT" master_online || fail "El master de ROS no arrancó en ${MASTER_TIMEOUT} s (mira la terminal 5)"
touch "$TFG_STATE/master"
ok "Master de ROS"

info "Esperando a la base Kobuki ($ODOM_TOPIC)"
wait_for "$ODOM_TIMEOUT" odom_received \
    || fail "Sin odometría en ${ODOM_TIMEOUT} s: base apagada, cable USB o ROS_IP / ROS_HOSTNAME mal (mira la terminal 5)"
topic_listed "$MUX_TOPIC" || fail "El mux de velocidad ($MUX_TOPIC) no está en marcha (mira la terminal 5)"
ok "Base Kobuki"

info "Esperando a la cámara del robot ($CAMERA_TOPIC, 3dsensor.launch)"
wait_for "$CAMERA_TIMEOUT" topic_listed "$CAMERA_TOPIC" \
    || fail "La cámara no arrancó en ${CAMERA_TIMEOUT} s (mira la terminal 4)"
ok "Cámara del robot"

# El comportamiento del robot (mesas, persona, obstáculos) sale del /scan de la cámara de profundidad
info "Esperando al láser virtual ($SCAN_TOPIC, 3dsensor.launch)"
wait_for "$SCAN_TIMEOUT" topic_listed "$SCAN_TOPIC" \
    || fail "No se publica $SCAN_TOPIC en ${SCAN_TIMEOUT} s (mira la terminal 4)"
ok "Láser virtual"

touch "$TFG_STATE/ready"
ok "TurtleBot2 inicializado, arrancando el menú"

# Frecuencia de la odometría, el robot funciona con menos pero indica problemas de WiFi o de carga
rate=$(PYTHONUNBUFFERED=1 timeout -s INT 4 rostopic hz "$ODOM_TOPIC" 2>/dev/null \
       | grep "average rate" | tail -n 1 | awk '{print $3}')
if [ -n "$rate" ] && awk "BEGIN {exit !($rate >= $ODOM_MIN_RATE)}"; then
    ok "Odometría a $rate Hz"
else
    warn "Odometría a ${rate:-?} Hz, lo normal son unos 50 Hz (WiFi o portátil del robot sobrecargado)"
fi

# Las terminales ya han leído sus marcas, la contraseña queda solo en el proceso de Terminator
trap - INT
sleep 2
rm -rf "$TFG_STATE"
ok "En marcha. En el menú (terminal 1) elige TurtleBot → Robot real; al terminar, cierra Terminator"
