#!/usr/bin/env python3
"""Menú para lanzar las distintas partes del TFG.

Se maneja con las flechas ↑/↓ y Enter (o pulsando el número de la opción).
"""
import os
import select
import shutil
import signal
import subprocess
import sys
import termios
import time
import tty

DEFAULT_MICROPHONE = 14  # mismo valor por defecto que en los launch
SIM_LOG = os.path.expanduser("~/.ros/log/tfg_menu_turtlebot.log")
REAL_LOG = os.path.expanduser("~/.ros/log/tfg_menu_robot_real.log")
LOCAL_MASTER = "http://localhost:11311"

INTERACTIVE = sys.stdin.isatty() and sys.stdout.isatty()
USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def style(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t): return style(t, "1")
def dim(t): return style(t, "2")
def cyan(t): return style(t, "36")
def green(t): return style(t, "32")
def yellow(t): return style(t, "33")
def red(t): return style(t, "31")
def highlight(t): return style(t, "1;30;46")


def clear():
    if USE_COLOR:
        print("\033[2J\033[H", end="")


def header(title, subtitle=""):
    width = 50
    print(cyan("╭" + "─" * width + "╮"))
    print(cyan("│") + bold(title.center(width)) + cyan("│"))
    if subtitle:
        print(cyan("│") + dim(subtitle.center(width)) + cyan("│"))
    print(cyan("╰" + "─" * width + "╯"))


# ---------------------------------------------------------------------------
# Selección con flechas
# ---------------------------------------------------------------------------
def read_key():
    """Lee una tecla sin esperar a Enter: 'up', 'down', 'enter', 'quit' o el carácter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd, termios.TCSANOW)  # TCSANOW: no descartar teclas pendientes
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            # Las flechas llegan como ESC [ A/B; un ESC solo es salir
            if not select.select([fd], [], [], 0.05)[0]:
                return "quit"
            return {b"[A": "up", b"[B": "down"}.get(os.read(fd, 2))
        if ch == b"\x03":
            raise KeyboardInterrupt
        if ch in (b"\r", b"\n"):
            return "enter"
        return ch.decode(errors="ignore").lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def section(title):
    return ("section", title, "", None)


def option(label, hint="", value=None):
    return ("option", label, hint, value)


def choose(title, subtitle, entries, selected=None, footer="", numbers=True):
    """Muestra una lista y devuelve el `value` de la opción elegida (None si se sale).

    `selected` es el value que aparece marcado al principio. Con `numbers`,
    las opciones se pueden elegir también pulsando su número.
    """
    opts = [i for i, e in enumerate(entries) if e[0] == "option"]
    values = [entries[i][3] for i in opts]
    pos = values.index(selected) if selected is not None and selected in values else 0

    if not INTERACTIVE:
        return _choose_typed(entries, opts)

    print("\033[?25l", end="")  # ocultar cursor
    try:
        while True:
            clear()
            header(title, subtitle)
            n = 0
            for i, (kind, label, hint, _) in enumerate(entries):
                if kind == "section":
                    print("\n  " + bold(label) if label else "")
                    continue
                n += 1
                key = f"{n}" if numbers and n <= 9 else " "
                if i == opts[pos]:
                    print(f"  {cyan('›')} {highlight(f' {key}  {label} ')}  {dim(hint)}")
                else:
                    print(f"    {cyan(key)}  {label}  {dim(hint)}")
            if footer:
                print("\n" + footer)
            print(dim("\n  ↑/↓ moverse · Enter elegir · q salir"))

            key = read_key()
            if key in ("up", "k"):
                pos = (pos - 1) % len(opts)
            elif key in ("down", "j"):
                pos = (pos + 1) % len(opts)
            elif key == "enter":
                return values[pos]
            elif key in ("quit", "q"):
                return None
            elif numbers and key and key.isdigit() and 1 <= int(key) <= min(len(opts), 9):
                return values[int(key) - 1]
    finally:
        print("\033[?25h", end="", flush=True)  # mostrar cursor


def _choose_typed(entries, opts):
    """Alternativa sin flechas cuando no hay terminal (p. ej. con una tubería)."""
    for n, i in enumerate(opts, 1):
        print(f"  {n}  {entries[i][1]}")
    try:
        choice = input("  › ").strip()
    except EOFError:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(opts):
        return entries[opts[int(choice) - 1]][3]
    return None


def pause(message):
    print(message)
    print(dim("\n  Pulsa una tecla para volver al menú"))
    if INTERACTIVE:
        read_key()


def local_env():
    """Entorno con el roscore de este PC.

    La simulación nunca debe ir al master del robot real (si ROS_MASTER_URI
    apunta a él, Gazebo publicaría /odom y /scan falsos y movería el robot).
    """
    env = dict(os.environ, ROS_MASTER_URI=LOCAL_MASTER, ROS_HOSTNAME="localhost")
    env.pop("ROS_IP", None)
    return env


def roslaunch(*args, env=None):
    """Lanza roslaunch en primer plano; Ctrl+C lo para y vuelve al menú."""
    cmd = ["roslaunch", "human_detection", *args]
    clear()
    print(dim("  $ " + " ".join(cmd)) + "\n")
    proc = subprocess.Popen(cmd, env=env)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.wait()


# ---------------------------------------------------------------------------
# Micrófono
# ---------------------------------------------------------------------------
CANCEL = "cancel"


def choose_microphone():
    """Devuelve el índice del micrófono elegido, o CANCEL."""
    try:
        import sounddevice as sd
        devices = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                   if d["max_input_channels"] > 0]
    except Exception as e:
        pause(yellow(f"\n  No se pudieron listar los micrófonos ({e}). Se usará el {DEFAULT_MICROPHONE}."))
        return DEFAULT_MICROPHONE

    entries = [option(f"{idx:>3}  {name}", "por defecto" if idx == DEFAULT_MICROPHONE else "", idx)
               for idx, name in devices]
    mic = choose("Micrófono", "¿Qué micrófono usar para la voz?", entries,
                 selected=DEFAULT_MICROPHONE, numbers=False)
    return CANCEL if mic is None else mic


# ---------------------------------------------------------------------------
# Acciones
# ---------------------------------------------------------------------------
def run_avatar(mode, voice):
    args = ["human_detection.launch", f"mode:={mode}", f"use_voice:={str(voice).lower()}"]
    if voice:
        mic = choose_microphone()
        if mic == CANCEL:
            return
        args.append(f"microphone:={mic}")
    roslaunch(*args)


def master_online(uri):
    import rosgraph
    return rosgraph.is_master_online(master_uri=uri)


def ros_already_running():
    """Si ya hay un roscore local, pregunta si seguir (evita abrir un segundo Gazebo)."""
    if not master_online(LOCAL_MASTER):
        return False
    go_on = choose("Ya hay un ROS en marcha", LOCAL_MASTER, [
        option("Volver", "ciérralo primero (Ctrl+C en su terminal)", False),
        option("Continuar igualmente", "puede abrir un segundo Gazebo", True),
    ], footer=yellow("  Si ya tienes Gazebo abierto, este se sumaría al que hay."))
    return not go_on


WORLDS = [
    ("emotion_room", "Habitación con mesa", "una mesa a la vista"),
    ("emotion_far_table", "Pasillo con mesa al fondo", "la encuentra en la tercera búsqueda"),
    ("emotion_furniture", "Mesas, sillas y mesa baja", "se atasca en la baja y busca otra"),
    ("emotion_empty", "Habitación vacía", "no hay mesa: huye 3 veces y se para"),
]


def choose_world():
    """Devuelve los argumentos de mundo y sensor de la simulación, o CANCEL."""
    entries = [option(label, hint, name) for name, label, hint in WORLDS]
    world = choose("Mundo de Gazebo", "¿Dónde se prueba el robot?", entries)
    if world is None:
        return CANCEL
    sensor = choose("Sensor del robot simulado", "¿Qué lleva el TurtleBot?", [
        option("Lidar 360º", "odometría perfecta", "lidar"),
        option("Cámara de profundidad", "como el robot real: 63º, nada a menos de 0.45 m "
               "y odometría con deriva", "camera"),
    ])
    if sensor is None:
        return CANCEL
    odom = "encoder" if sensor == "camera" else "world"
    return [f"world:={world}", f"sim_sensor:={sensor}", f"sim_odom:={odom}"]


def run_turtlebot_sim_camera():
    world = choose_world()
    if world == CANCEL or ros_already_running():
        return
    roslaunch("turtlebot_emotion.launch", "gazebo:=true", "use_camera:=true", *world,
              env=local_env())


def run_turtlebot_real():
    master = os.environ.get("ROS_MASTER_URI", "")
    lines = [f"  ROS_MASTER_URI = {cyan(master or '(sin definir)')}"]
    if not master or "localhost" in master:
        lines.append(yellow("  Apunta a este PC: lanza en el portátil del TurtleBot2"))
        lines.append(dim("    roslaunch turtlebot_bringup minimal.launch  (y el driver del lidar)"))
        lines.append(dim("    roslaunch turtlebot_bringup 3dsensor.launch  (cámara del robot)"))
    lines.append(dim("  Coloca el robot mirando hacia ti, a ~1 m."))

    confirm = choose("TurtleBot real", "¿Lanzar?", [
        option("Sí, lanzar", value=True),
        option("No, volver", value=False),
    ], selected=False, footer="\n".join(lines))
    if confirm:
        roslaunch("turtlebot_emotion.launch", "use_camera:=true",
                  "image_topic:=/camera/rgb/image_raw")


EMOTIONS = [
    ("parar", "se para del todo (hasta mandar otra emoción)"),
    ("feliz", "se acerca"),
    ("triste", "se acerca despacio"),
    ("enojo", "se esconde bajo la mesa"),
    ("miedo", "retrocede"),
    ("sorpresa", "da una vuelta"),
    ("neutral", "se queda quieto"),
]


def stop_group(proc, name):
    """Para con Ctrl+C un proceso lanzado con start_new_session y sus hijos."""
    if proc is None or proc.poll() is not None:
        return
    print(dim(f"\n  Cerrando {name}..."))
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


# Publicador de /emotion_override en un proceso aparte: así cada panel usa el
# master que toca (local o el del robot) y no hace falta rospy.init_node en el menú
SENDER = r"""
import sys, rospy
from std_msgs.msg import String
rospy.init_node("emotion_panel", anonymous=True, disable_signals=True)
pub = rospy.Publisher("/emotion_override", String, queue_size=1, latch=True)
for line in sys.stdin:
    pub.publish(String(data=line.strip()))
rospy.sleep(0.5)  # que llegue el último mensaje antes de salir
"""


class EmotionSender:
    def __init__(self, env, log):
        self.proc = subprocess.Popen([sys.executable, "-c", SENDER], stdin=subprocess.PIPE,
                                     stdout=log, stderr=subprocess.STDOUT, env=env,
                                     text=True, start_new_session=True)

    def send(self, label):
        if self.proc.poll() is None:
            self.proc.stdin.write(label + "\n")
            self.proc.stdin.flush()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def run_turtlebot_sim_manual():
    """Gazebo en segundo plano y un panel para mandar emociones.

    La cámara solo se abre si se elige "cámara" en el panel.
    """
    world = choose_world()
    if world == CANCEL or ros_already_running():
        return
    env = local_env()
    run_emotion_panel(
        ["gazebo:=true", "use_camera:=false", *world], env, SIM_LOG,
        camera_cmd=["rosrun", "human_detection", "ros_node.py"],
        title="Emociones para el TurtleBot", subtitle="Gazebo en marcha",
        starting="Arrancando Gazebo...", name="Gazebo")


def run_turtlebot_real_manual():
    """Robot real con un panel para mandar las emociones a mano (sin cámara)."""
    master = os.environ.get("ROS_MASTER_URI", "")
    clear()
    print(dim(f"\n  Comprobando el robot en {master or '(ROS_MASTER_URI sin definir)'}..."))
    problems = check_real_robot(master)
    lines = [f"  ROS_MASTER_URI = {cyan(master or '(sin definir)')}"]
    lines += [red("  ✗ " + p) for p in problems] or [green("  ✓ Llegan /scan y /odom del robot")]
    lines += [dim("  Coloca el robot mirando hacia ti, a ~1 m, y deja sitio alrededor."),
              dim("  Ten a mano el botón \"parar\" (o Ctrl+C) por si acaso.")]
    entries = [option("Sí, lanzar", value=True), option("No, volver", value=False)]
    confirm = choose("TurtleBot real · emociones a mano", "¿Lanzar?", entries,
                     selected=False if problems else True, footer="\n".join(lines))
    if not confirm:
        return
    run_emotion_panel(
        ["use_camera:=false", "debug_dir:=" + os.path.expanduser("~/.ros/tfg_vueltas")],
        dict(os.environ), REAL_LOG,
        camera_cmd=["rosrun", "human_detection", "ros_node.py",
                    "_image_topic:=/camera/rgb/image_raw"],
        title="Emociones para el TurtleBot real", subtitle=master,
        starting="Arrancando el nodo del robot...", name="el nodo del robot")


def check_real_robot(master):
    """Lista de problemas que impiden probar con el robot real (vacía si todo bien)."""
    if not master or "localhost" in master or "127.0.0.1" in master:
        return ["ROS_MASTER_URI apunta a este PC, no al robot"]
    if not master_online(master):
        return ["No hay roscore en " + master + " (¿minimal.launch en el robot?)"]
    import rosgraph
    try:
        topics = dict(rosgraph.Master("/launch_menu", master_uri=master).getPublishedTopics(""))
    except Exception as e:
        return [f"No se pueden leer los topics ({e})"]
    missing = [t for t in ("/scan", "/odom") if t not in topics]
    problems = [f"No se publica {t}" for t in missing]
    if "/scan" in missing:
        problems.append("Lanza en el robot: roslaunch turtlebot_bringup 3dsensor.launch")
    return problems


def run_emotion_panel(launch_args, env, log_path, camera_cmd, title, subtitle, starting, name):
    """Lanza turtlebot_emotion.launch en segundo plano y abre el panel de emociones."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(
        ["roslaunch", "human_detection", "turtlebot_emotion.launch", *launch_args],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    camera = {"proc": None}
    sender = None

    def start_camera():
        if camera["proc"] is None or camera["proc"].poll() is not None:
            camera["proc"] = subprocess.Popen(camera_cmd, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True, env=env)

    try:
        clear()
        print(dim("\n  " + starting), end="", flush=True)
        while not master_online(env["ROS_MASTER_URI"]):
            if proc.poll() is not None:
                pause(red(f"\n  roslaunch ha terminado. Mira el log: {log_path}"))
                return
            time.sleep(0.5)
        sender = EmotionSender(env, log)
        emotion_panel(sender, proc, start_camera, lambda: camera["proc"], title, subtitle,
                      log_path)
        if proc.poll() not in (None, 0):
            pause(red(f"\n  roslaunch ha terminado con error. Mira el log: {log_path}"))
    except KeyboardInterrupt:
        pass
    finally:
        clear()
        if sender is not None:
            sender.send("parar")
            sender.close()
        stop_group(camera["proc"], "la cámara")
        stop_group(proc, name)
        log.close()


def emotion_panel(sender, proc, start_camera, camera_proc, title, subtitle, log_path):
    entries = [option(name, action, name) for name, action in EMOTIONS]
    entries += [section(""),
                option("cámara", "abre la cámara y usa la emoción detectada", "auto"),
                option("Salir", "para el robot y cierra todo", None)]
    last = None
    time.sleep(1.0)  # dar tiempo a que roslaunch falle si algo va mal
    while proc.poll() is None:
        cam = camera_proc()
        if last == "auto":
            footer = green("  ● Usando la emoción de la cámara")
        elif last == "parar":
            footer = yellow("  ■ Parado")
        elif last:
            footer = green(f"  ● Enviado: {last}")
        else:
            footer = dim("  Todavía no se ha enviado ninguna emoción")
        if cam is not None and cam.poll() is not None:
            footer += red("\n  La cámara se ha cerrado (mira el log)")
        elif cam is not None:
            footer += dim("\n  Cámara abierta (se cierra al salir)")
        footer += dim(f"\n  Log de ROS: {log_path}")

        value = choose(title, subtitle, entries, selected=last or "feliz", footer=footer)
        if value is None:
            return
        if value == "auto":
            start_camera()
        sender.send(value)
        last = value


# ---------------------------------------------------------------------------
# Menú principal
# ---------------------------------------------------------------------------
MENU = [
    section("Avatar"),
    option("Cara + imágenes", value=lambda: run_avatar(1, voice=False)),
    option("Voz + cara + imágenes", value=lambda: run_avatar(2, voice=True)),
    option("Voz + cara + vídeos", value=lambda: run_avatar(3, voice=True)),
    option("Cara + vídeos de gestos", value=lambda: run_avatar(4, voice=False)),
    section("TurtleBot"),
    option("Simulación con cámara", value=run_turtlebot_sim_camera),
    option("Simulación con emociones a elegir", value=run_turtlebot_sim_manual),
    option("Robot real", value=run_turtlebot_real),
    option("Robot real con emociones a elegir", value=run_turtlebot_real_manual),
    section(""),
    option("Salir", value=None),
]


def main():
    if not shutil.which("roslaunch"):
        print(red("No se encuentra roslaunch. Carga el entorno primero:"))
        print("  source /opt/ros/noetic/setup.bash && source ~/Escritorio/TFG/devel/setup.bash")
        sys.exit(1)

    last = None
    while True:
        try:
            action = choose("TFG · Gestos y emociones",
                            "ROS " + os.environ.get("ROS_DISTRO", ""), MENU, selected=last)
        except KeyboardInterrupt:
            action = None
        if action is None:
            clear()
            return
        last = action
        action()


if __name__ == "__main__":
    main()
