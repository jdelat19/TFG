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


def roslaunch(*args):
    """Lanza roslaunch en primer plano; Ctrl+C lo para y vuelve al menú."""
    cmd = ["roslaunch", "human_detection", *args]
    clear()
    print(dim("  $ " + " ".join(cmd)) + "\n")
    proc = subprocess.Popen(cmd)
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


def ros_already_running():
    """Si ya hay un roscore, pregunta si seguir (evita abrir un segundo Gazebo)."""
    import rosgraph
    if not rosgraph.is_master_online():
        return False
    go_on = choose("Ya hay un ROS en marcha", os.environ.get("ROS_MASTER_URI", ""), [
        option("Volver", "ciérralo primero (Ctrl+C en su terminal)", False),
        option("Continuar igualmente", "puede abrir un segundo Gazebo", True),
    ], footer=yellow("  Si ya tienes Gazebo abierto, este se sumaría al que hay."))
    return not go_on


def run_turtlebot_sim_camera():
    if ros_already_running():
        return
    roslaunch("turtlebot_emotion.launch", "gazebo:=true", "use_camera:=true")


def run_turtlebot_real():
    master = os.environ.get("ROS_MASTER_URI", "")
    lines = [f"  ROS_MASTER_URI = {cyan(master or '(sin definir)')}"]
    if not master or "localhost" in master:
        lines.append(yellow("  Apunta a este PC: lanza en el robot"))
        lines.append(dim("    roslaunch turtlebot3_bringup turtlebot3_robot.launch"))
    lines.append(dim("  Coloca el robot mirando hacia ti, a ~1 m."))

    confirm = choose("TurtleBot real", "¿Lanzar?", [
        option("Sí, lanzar", value=True),
        option("No, volver", value=False),
    ], selected=False, footer="\n".join(lines))
    if confirm:
        roslaunch("turtlebot_emotion.launch", "use_camera:=true")


EMOTIONS = [
    ("feliz", "se acerca"),
    ("triste", "se acerca despacio"),
    ("enojo", "se esconde bajo la mesa"),
    ("miedo", "retrocede"),
    ("sorpresa", "da una vuelta"),
    ("neutral", "se queda quieto"),
]


def run_turtlebot_sim_manual():
    """Gazebo en segundo plano y un panel para mandar emociones."""
    if ros_already_running():
        return
    os.makedirs(os.path.dirname(SIM_LOG), exist_ok=True)
    log = open(SIM_LOG, "w")
    proc = subprocess.Popen(
        ["roslaunch", "human_detection", "turtlebot_emotion.launch",
         "gazebo:=true", "use_camera:=false"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    try:
        import rosgraph
        import rospy
        from std_msgs.msg import String

        clear()
        print(dim("\n  Arrancando Gazebo..."), end="", flush=True)
        while not rosgraph.is_master_online():
            if proc.poll() is not None:
                pause(red(f"\n  roslaunch ha terminado. Mira el log: {SIM_LOG}"))
                return
            time.sleep(0.5)

        rospy.init_node("launch_menu", anonymous=True, disable_signals=True)
        pub = rospy.Publisher("/emotion_override", String, queue_size=1, latch=True)
        emotion_panel(pub, proc)
        if proc.poll() not in (None, 0):
            pause(red(f"\n  roslaunch ha terminado con error. Mira el log: {SIM_LOG}"))
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            clear()
            print(dim("\n  Cerrando Gazebo..."))
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        log.close()


def emotion_panel(pub, proc):
    from std_msgs.msg import String

    entries = [option(name, action, name) for name, action in EMOTIONS]
    entries += [section(""),
                option("auto", "vuelve a usar cámara/voz", "auto"),
                option("Salir", "cierra Gazebo", None)]
    last = None
    time.sleep(1.0)  # dar tiempo a que roslaunch falle si algo va mal
    while proc.poll() is None:
        if last:
            footer = green(f"  ● Enviado: {last}")
        else:
            footer = dim("  Todavía no se ha enviado ninguna emoción")
        footer += dim(f"\n  Log de ROS: {SIM_LOG}")

        value = choose("Emociones para el TurtleBot", "Gazebo en marcha", entries,
                       selected=last or "feliz", footer=footer)
        if value is None:
            return
        pub.publish(String(data=value))
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
