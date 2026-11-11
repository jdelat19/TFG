#!/usr/bin/env python3
"""Pruebas de integridad del paquete: que los launch, el CMakeLists, el menú y
los ficheros de datos apunten a cosas que existen."""
import os
import py_compile
import re

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCRIPTS = os.path.join(PKG, "scripts")


def read(*parts):
    with open(os.path.join(PKG, *parts), encoding="utf-8") as f:
        return f.read()


def node_scripts():
    return sorted(f for f in os.listdir(SCRIPTS) if f.endswith(".py"))


def test_launch_files_use_existing_nodes():
    for launch in os.listdir(os.path.join(PKG, "launch")):
        text = read("launch", launch)
        for node in re.findall(r'pkg="human_detection"\s+type="([^"]+)"', text):
            assert node in node_scripts(), f"{launch}: no existe scripts/{node}"


def test_every_script_is_installed_and_executable():
    cmake = read("CMakeLists.txt")
    for script in node_scripts():
        assert f"scripts/{script}" in cmake, f"{script} no está en CMakeLists.txt"
        path = os.path.join(SCRIPTS, script)
        assert os.access(path, os.X_OK), f"{script} no es ejecutable (chmod +x)"
        assert read("scripts", script).startswith("#!/usr/bin/env python3")


def test_all_python_files_compile():
    for folder in ("scripts", "src", "test"):
        for name in os.listdir(os.path.join(PKG, folder)):
            if name.endswith(".py"):
                py_compile.compile(os.path.join(PKG, folder, name), doraise=True)


def test_menu_worlds_exist():
    menu = read("scripts", "launch_menu.py")
    worlds = re.findall(r'\("(emotion_\w+)", "', menu)
    assert len(worlds) == 4
    for world in worlds:
        assert os.path.isfile(os.path.join(PKG, "worlds", world + ".world")), world


def test_sim_launch_defaults_exist():
    sim = read("launch", "turtlebot_sim.launch")
    world = re.search(r'name="world" default="(\w+)"', sim).group(1)
    assert os.path.isfile(os.path.join(PKG, "worlds", world + ".world"))
    assert os.path.isfile(os.path.join(PKG, "urdf", "turtlebot2.urdf.xacro"))


def test_no_hardcoded_home_paths():
    # El proyecto tiene que funcionar se clone donde se clone
    for folder in ("scripts", "src", "launch"):
        for name in os.listdir(os.path.join(PKG, folder)):
            if not name.endswith((".py", ".launch")):
                continue
            text = read(folder, name)
            assert "Escritorio/TFG" not in text and "/home/" not in text, f"{folder}/{name}"
