# Robot que reacciona a las emociones humanas

Sistema en **ROS Noetic** que detecta en tiempo real los **gestos** y la **emoción** de una persona
(por la cara con la cámara y, opcionalmente, por la voz con el micrófono) y la usa de dos formas:

- **Avatar**: una ventana muestra una imagen o un vídeo según la emoción o el gesto, con las manos
  de la persona dibujadas en 3D.
- **TurtleBot2** (real o simulado en Gazebo): el robot reacciona a la emoción.

| Emoción | El robot... |
|---|---|
| Feliz | se acerca a la persona (hasta 0,6 m) |
| Triste | se acerca despacio y se queda a 0,9 m, acompañando |
| Enojo / Disgusto | busca una mesa o silla donde quepa y **se esconde debajo** |
| Miedo | retrocede hasta 1,5 m |
| Sorpresa | da una vuelta sobre sí mismo |
| Neutral | se queda quieto mirando a la persona |

**Cómo lo hace:** MediaPipe Holistic saca los puntos del cuerpo, la cara y las manos; los gestos
salen de reglas geométricas sobre esos puntos y la emoción facial de HSEmotion (red entrenada con
AffectNet). La voz se transcribe con Whisper y su emoción combina el texto (RoBERTa), el tono
(wav2vec 2.0) y una lista de frases clave. El robot ve el entorno con un lidar o con la cámara de
profundidad (que imita un lidar de 63°) y reconoce las mesas por sus patas.

## 1. Requisitos

- PC con **Ubuntu 20.04** (obligatorio para ROS Noetic), webcam y, para la voz, micrófono.
- ~15 GB libres y conexión a internet durante la instalación.
- Solo para el robot real: un **TurtleBot2** (base Kobuki + cámara de profundidad) con su portátil
  y `turtlebot_bringup` instalado, en la misma red WiFi que el PC.

## 2. Instalación

```bash
sudo apt install -y git
git clone https://github.com/jdelat19/TFG.git ~/TFG
cd ~/TFG
./install.sh
```

`install.sh` lo hace todo (tarda entre 20 y 60 minutos y pide la contraseña de `sudo`): instala ROS
Noetic con Gazebo, crea un entorno de Python en `.venv` con las versiones exactas que se usaron en
el desarrollo, descarga los modelos de IA, compila el workspace, añade ROS a `~/.bashrc` y pasa los
tests. Al final pregunta la IP del TurtleBot2: **pulsa Enter si solo vas a usar la simulación**.

Si algo falla, vuelve a ejecutar `./install.sh`: retoma lo que falte. Al terminar, **abre una
terminal nueva**.

## 3. Uso

Todo se lanza desde un menú que se maneja con las flechas y Enter:

```bash
rosrun human_detection launch_menu.py
```

| Opción | Qué abre |
|---|---|
| **Avatar** → Cara + imágenes | Cámara y avatar con la imagen de la emoción facial |
| Avatar → Voz + cara + imágenes | Igual, y la voz manda si no es neutral (pide elegir micrófono) |
| Avatar → Voz + cara + vídeos | Igual, con vídeos en lugar de imágenes |
| Avatar → Cara + vídeos de gestos | Vídeo según el gesto (p. ej. rascarse el cuello) |
| **TurtleBot** → Simulación con cámara | Gazebo + tu webcam: el robot simulado reacciona a tu cara |
| TurtleBot → Simulación con emociones a elegir | Gazebo + un panel para mandar emociones a mano |
| TurtleBot → Robot real | El TurtleBot2 reacciona a la cara que ve su cámara |
| TurtleBot → Robot real con emociones a elegir | El TurtleBot2 obedece a un panel (incluye **parar**) |

En la simulación el menú pregunta el mundo (habitación con mesa, pasillo, varias mesas o habitación
vacía) y el sensor (lidar 360° o cámara de profundidad como el robot real). Con el TurtleBot se abre
además el **mapa mental**: lo que el robot cree que hay a su alrededor visto desde arriba.

Para cerrar: `Esc` en las ventanas de cámara y mapa, `Ctrl+C` en la terminal o **Salir** en el panel.

### Robot real

1. **Una sola vez**, en el portátil del robot (por SSH): añadir a su `~/.bashrc`
   ```bash
   export ROS_MASTER_URI=http://IP_DEL_ROBOT:11311
   export ROS_IP=IP_DEL_ROBOT
   ```
   y en el PC, si no lo hizo `install.sh`, añadir a `~/.bashrc`
   ```bash
   export ROS_MASTER_URI=http://IP_DEL_ROBOT:11311
   export ROS_HOSTNAME=IP_DEL_PC
   export ROS_IP=IP_DEL_PC
   ```
   (las IP salen con `hostname -I` en cada máquina). El master de ROS corre en el robot.
2. Encender la base Kobuki y el portátil del robot, y conectar los dos equipos a la misma red.
3. En el PC: `bash run.sh`. Pide usuario, IP y contraseña del portátil del robot, arranca por SSH la
   base y la cámara, comprueba que llegan `/odom`, `/scan` y la imagen, y abre Terminator con el
   menú (terminal 1) y los monitores de velocidad y odometría.
4. Colocar el robot **mirando a la persona, a ~1 m**, y elegir en el menú **TurtleBot → Robot real**
   (o **con emociones a elegir** para controlarlo a mano). Al terminar, cerrar Terminator.

Para esconderse hace falta una mesa o silla más alta que el robot (~45 cm), con patas separadas al
menos 45 cm y visibles a la altura de la cámara (~30 cm del suelo), con ~1,5 m libres delante.

### Problemas frecuentes

| Problema | Solución |
|---|---|
| `rosrun: command not found` o no encuentra `human_detection` | Abrir una terminal nueva (o `source ~/.bashrc`) |
| La ventana de la cámara sale negra | Otra aplicación usa la webcam; ciérrala. Comprobar con `v4l2-ctl --list-devices` |
| La voz no detecta nada | Elegir otro micrófono en el menú; ver la lista con `.venv/bin/python -m sounddevice` |
| `run.sh`: "Contraseña incorrecta" o "No se pudo conectar" | Comprobar IP y usuario con `ssh usuario@IP_DEL_ROBOT` |
| `run.sh`: "ROS_MASTER_URI es ... pero la IP del robot es ..." | Corregir la IP del robot en `~/.bashrc` del PC y abrir otra terminal |
| `run.sh`: "Sin odometría" | Base Kobuki apagada o cable USB suelto (mirar la terminal 5) |
| El robot no encuentra la mesa | Ver el mapa mental; con "emociones a elegir" cada vuelta se guarda en `~/.ros/tfg_vueltas/` |
| El robot se mueve sin querer | Panel → **parar** o `Ctrl+C` en la terminal 1 (si se corta la conexión, la base se para sola en ~0,6 s) |

## 4. Estructura

```
TFG/
├── install.sh                 Instalación completa
├── run.sh                     Arranque del TurtleBot2 real (SSH + Terminator + menú)
├── PRUEBAS_FUNCIONALES.md     Batería de 30 pruebas para verificar y grabar el sistema
├── requirements*.txt          Dependencias de Python (lock = versiones exactas)
├── docs/capturas/             Capturas de gestos y emociones detectados
└── src/human_detection/       Paquete ROS (ver su README para el detalle técnico)
    ├── launch/                Avatar, TurtleBot y simulación en Gazebo
    ├── scripts/               Nodos: detección, voz, avatar, TurtleBot, mapa mental y menú
    ├── src/                   Lógica: gestos, emociones, procesado del lidar y comportamientos
    ├── config/                Frases clave de cada emoción para la voz
    ├── media/                 Imágenes y vídeos del avatar
    ├── urdf/, worlds/         Robot y mundos de la simulación
    └── test/                  Tests automáticos
```

Detalle de nodos, topics, algoritmos y parámetros: [src/human_detection/README.md](src/human_detection/README.md).

## 5. Pruebas

- **Tests automáticos** (sin cámara ni robot, unos segundos):
  `cd src/human_detection && ../../.venv/bin/python -m pytest test/`
- **Pruebas funcionales** con cámara, simulación y robot: [PRUEBAS_FUNCIONALES.md](PRUEBAS_FUNCIONALES.md).

## 6. Tecnologías

ROS Noetic, Gazebo 11, OpenCV, MediaPipe Holistic, HSEmotion (`enet_b2_7`, onnxruntime), FER
(alternativa), Whisper, RoBERTa (`cardiffnlp/twitter-roberta-base-emotion`), wav2vec 2.0
(`superb/wav2vec2-base-superb-er`), PyOpenGL y GLFW.
