#!/usr/bin/env python3
"""Menú para lanzar las distintas partes del TFG."""
import os
import shutil
import signal
import subprocess
import sys
import time

DEFAULT_MICROPHONE = 14  # mismo valor por defecto que en los launch
SIM_LOG = os.path.expanduser("~/.ros/log/tfg_menu_turtlebot.log")

USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def style(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t): return style(t, "1")
def dim(t): return style(t, "2")
def cyan(t): return style(t, "36")
def green(t): return style(t, "32")
def yellow(t): return style(t, "33")
def red(t): return style(t, "31")


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


def ask(prompt):
    try:
        return input(f"\n  {prompt} {cyan('›')} ").strip().lower()
    except EOFError:
        return "q"


def roslaunch(*args):
    """Lanza roslaunch en primer plano; Ctrl+C lo para y vuelve al menú."""
    cmd = ["roslaunch", "human_detection", *args]
    print(dim("\n  $ " + " ".join(cmd)) + "\n")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.wait()


# ---------------------------------------------------------------------------
# Micrófono
# ---------------------------------------------------------------------------
def choose_microphone():
    """Lista los micrófonos y devuelve el elegido (None = el de por defecto)."""
    try:
        import sounddevice as sd
        devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    except Exception as e:
        print(yellow(f"\n  No se pudieron listar los micrófonos ({e})."))
        return None

    print(bold("\n  Micrófonos disponibles\n"))
    for idx, dev in devices:
        mark = green("  ← por defecto") if idx == DEFAULT_MICROPHONE else ""
        print(f"   {cyan(f'{idx:>3}')}  {dev['name']}{mark}")

    valid = {str(idx) for idx, _ in devices}
    while True:
        choice = ask(f"Micrófono {dim('[Enter = ' + str(DEFAULT_MICROPHONE) + ']')}")
        if choice == "":
            return None
        if choice in valid:
            return int(choice)
        print(red("  Número no válido."))


# ---------------------------------------------------------------------------
# Acciones
# ---------------------------------------------------------------------------
def run_avatar(mode, voice):
    args = ["human_detection.launch", f"mode:={mode}", f"use_voice:={str(voice).lower()}"]
    if voice:
        mic = choose_microphone()
        if mic is not None:
            args.append(f"microphone:={mic}")
    roslaunch(*args)


def run_turtlebot_sim_camera():
    roslaunch("turtlebot_emotion.launch", "gazebo:=true", "use_camera:=true")


def run_turtlebot_real():
    master = os.environ.get("ROS_MASTER_URI", "")
    print(bold("\n  TurtleBot real\n"))
    print(f"   ROS_MASTER_URI = {cyan(master or '(sin definir)')}")
    if not master or "localhost" in master:
        print(yellow("\n   Parece que ROS_MASTER_URI apunta a este PC. Recuerda lanzar en el robot:"))
        print(dim("     roslaunch turtlebot3_bringup turtlebot3_robot.launch"))
    print(dim("\n   Coloca el robot mirando hacia ti, a ~1 m."))
    if ask(f"¿Lanzar? {dim('[s/N]')}") == "s":
        roslaunch("turtlebot_emotion.launch", "use_camera:=true")


EMOTIONS = [
    ("1", "feliz", "se acerca"),
    ("2", "triste", "se acerca despacio"),
    ("3", "enojo", "se esconde bajo la mesa"),
    ("4", "miedo", "retrocede"),
    ("5", "sorpresa", "da una vuelta"),
    ("6", "neutral", "se queda quieto"),
]


def run_turtlebot_sim_manual():
    """Gazebo en segundo plano y un panel para mandar emociones con el teclado."""
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

        print(dim("\n  Arrancando Gazebo..."), end="", flush=True)
        while not rosgraph.is_master_online():
            if proc.poll() is not None:
                print(red(f"\n  roslaunch ha terminado. Mira el log: {SIM_LOG}"))
                ask("Enter para volver")
                return
            time.sleep(0.5)
        print(green(" listo"))

        rospy.init_node("launch_menu", anonymous=True, disable_signals=True)
        pub = rospy.Publisher("/emotion_override", String, queue_size=1, latch=True)
        emotion_panel(pub, proc)
    finally:
        if proc.poll() is None:
            print(dim("\n  Cerrando Gazebo..."))
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        log.close()


def emotion_panel(pub, proc):
    from std_msgs.msg import String

    by_key = {key: name for key, name, _ in EMOTIONS}
    last = None
    while proc.poll() is None:
        clear()
        header("Emociones para el TurtleBot", "Gazebo en marcha")
        print()
        for key, name, action in EMOTIONS:
            marker = green("●") if name == last else " "
            print(f"  {marker} {cyan(key)}  {name:<10} {dim(action)}")
        print(f"\n    {cyan('a')}  {'auto':<10} {dim('vuelve a usar cámara/voz')}")
        print(f"    {cyan('q')}  {'salir':<10} {dim('cierra Gazebo')}")
        print(dim(f"\n  Log de ROS: {SIM_LOG}"))

        try:
            choice = ask("Emoción")
        except KeyboardInterrupt:
            return
        if choice == "q":
            return
        if choice == "a":
            pub.publish(String(data="auto"))
            last = None
        elif choice in by_key:
            last = by_key[choice]
            pub.publish(String(data=last))


# ---------------------------------------------------------------------------
# Menú principal
# ---------------------------------------------------------------------------
MENU = [
    ("Avatar", [
        ("1", "Cara + imágenes", lambda: run_avatar(1, voice=False)),
        ("2", "Voz + cara + imágenes", lambda: run_avatar(2, voice=True)),
        ("3", "Voz + cara + vídeos", lambda: run_avatar(3, voice=True)),
        ("4", "Cara + vídeos de gestos", lambda: run_avatar(4, voice=False)),
    ]),
    ("TurtleBot", [
        ("5", "Simulación con cámara", run_turtlebot_sim_camera),
        ("6", "Simulación con emociones por teclado", run_turtlebot_sim_manual),
        ("7", "Robot real", run_turtlebot_real),
    ]),
]


def main():
    if not shutil.which("roslaunch"):
        print(red("No se encuentra roslaunch. Carga el entorno primero:"))
        print("  source /opt/ros/noetic/setup.bash && source ~/Escritorio/TFG/devel/setup.bash")
        sys.exit(1)

    actions = {key: action for _, items in MENU for key, _, action in items}
    while True:
        clear()
        header("TFG · Gestos y emociones", "ROS " + os.environ.get("ROS_DISTRO", ""))
        for section, items in MENU:
            print("\n  " + bold(section))
            for key, label, _ in items:
                print(f"    {cyan(key)}  {label}")
        print(f"\n    {cyan('q')}  Salir")

        try:
            choice = ask("Elige una opción")
        except KeyboardInterrupt:
            choice = "q"
        if choice == "q":
            print()
            return
        if choice in actions:
            actions[choice]()
        else:
            print(red("  Opción no válida."))
            time.sleep(0.8)


if __name__ == "__main__":
    main()
