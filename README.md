# Detector de gestos y emociones TODO: Cambiar por el nombre del TFG

## Descripción
Este proyecto **detecta gestos y emociones de una persona en tiempo real** con la cámara del ordenador y, opcionalmente, con el micrófono. Está hecho como paquete de **ROS Noetic** (`human_detection`) y la información detectada se usa de dos formas:

- **Avatar**: una ventana muestra una imagen o un vídeo según la emoción o el gesto detectado, con las manos dibujadas en 3D.
- **TurtleBot3**: el robot (simulado en Gazebo o real) reacciona a la emoción de la persona: se acerca, retrocede, se esconde debajo de una mesa, etc.

La cámara obtiene los puntos clave del cuerpo, la cara y las manos (landmarks) con MediaPipe. A partir de sus posiciones se identifican gestos como cruzar los brazos, rascarse el cuello o llevarse las manos a la cara. La emoción facial se obtiene con HSEmotion y la de la voz con Whisper, wav2vec 2.0 y RoBERTa.

## Estructura
```
TFG/                                  # Workspace de catkin (ROS Noetic)
├── requirements.txt                  # Dependencias de Python
├── Img/                              # Capturas de gestos y emociones
└── src/human_detection/              # Paquete ROS
    ├── launch/
    │   ├── human_detection.launch    # Detección + avatar (+ voz)
    │   ├── turtlebot_emotion.launch  # Detección + comportamiento del TurtleBot
    │   └── turtlebot_sim.launch      # Gazebo con TurtleBot3, mesa y persona
    ├── scripts/                      # Nodos ROS
    │   ├── ros_node.py               # Cámara -> gestos y emoción facial (/human_state)
    │   ├── voice_node.py             # Micrófono -> emoción por voz (/voice_emotion)
    │   ├── avatar_node.py            # Avatar 2D (imágenes / vídeos), se lanza con rosrun
    │   ├── avatar_node_3d.py         # Avatar 3D con las manos de MediaPipe
    │   ├── turtlebot_emotion_node.py # TurtleBot que reacciona a la emoción
    │   ├── launch_menu.py            # Menú para elegir modo
    │   └── emociones.json            # Frases clave por emoción para la voz
    ├── src/                          # Módulos
    │   ├── gesture_detector.py       # Detección con MediaPipe Holistic
    │   ├── gestures.py               # Reglas de cada gesto
    │   ├── facial_expression.py      # Emoción facial (HSEmotion o FER)
    │   ├── scan_utils.py             # Procesado del lidar (patas de mesa, etc.)
    │   ├── avatar_media.py           # Imagen o vídeo del avatar según modo/emoción/gesto
    │   └── utils.py                  # Coordenadas y distancias
    ├── media/                        # Imágenes y vídeos del avatar
    ├── worlds/                       # Mundo de Gazebo (habitación con mesa y persona)
    └── test/                         # Tests (python3 -m pytest test/)
```

## Cómo funciona

```
 Cámara ──> ros_node.py ──/human_state──┬──> avatar_node_3d.py   (avatar)
                                        │
 Micrófono ──> voice_node.py ──/voice_emotion──┤
                                        │
                                        └──> turtlebot_emotion_node.py ──/cmd_vel──> TurtleBot3
                                                   ▲
                                          /scan, /odom (lidar y odometría)
```

1. **Captura de vídeo**: `ros_node.py` lee la webcam del ordenador.
2. **Detección de landmarks**: MediaPipe Holistic detecta los puntos clave del cuerpo, la cara y las manos.
3. **Gestos**: se calculan distancias y ángulos entre landmarks y se aplican las reglas de `gestures.py`.
4. **Emoción facial**: HSEmotion clasifica la cara en una de 7 emociones.
5. **Publicación**: el gesto, la emoción y los landmarks se publican en `/human_state` (JSON) y se muestra la imagen con el esqueleto dibujado.
6. **Voz** (opcional): `voice_node.py` graba fragmentos de 4 s, los transcribe con Whisper y combina la emoción del texto (RoBERTa y las frases de `emociones.json`) con la del tono (wav2vec 2.0). El resultado va a `/voice_emotion`.
7. **Reacción**: el avatar o el TurtleBot usan la emoción de la voz si es reciente y no es neutral; si no, la de la cara.

## Gestos
### Gestos Corporales
- Brazos cruzados
- Brazos abiertos
- Manos en las caderas
- Manos juntas

### Gestos de Manos y Cara
- Rascarse el cuello
- Morderse las uñas
- Manos en la cara
- Tocarse la cabeza

### Gestos Faciales
- Inclinar la cabeza

### Gestos Personalizados
- Pulgar hacia arriba
- Señalar
- Seña de paz

### Gestos por Forma de la Mano
Se calcula qué dedos están extendidos comparando distancias a la muñeca,
así que funcionan aunque la mano esté girada.
- Puño
- Mano abierta
- Saludar (mano abierta moviéndose de lado a lado)
- OK
- Cuernos
- Llámame
- Pulgar hacia abajo

## Emociones
Feliz, triste, enojo, disgusto, miedo, sorpresa y neutral.

## Avatar
El avatar muestra de fondo una imagen (`media/imagenes/<emoción>.png`) o un vídeo (`media/videos/<emoción>.mp4`) según el modo. Si no hay fichero para la emoción, se usa `no detectado`.

| Modo | Entrada | Muestra |
|---|---|---|
| 1 | Cara | Imagen de la emoción |
| 2 | Voz + cara | Imagen de la emoción |
| 3 | Voz + cara | Vídeo de la emoción |
| 4 | Cara + gesto | Vídeo del gesto (p. ej. `rascarse_cuello.mp4`), o de la emoción si no hay vídeo del gesto |

## TurtleBot
`turtlebot_emotion_node.py` lee la emoción, el lidar (`/scan`) y la odometría (`/odom`) y mueve el robot por `/cmd_vel`. Una emoción tiene que mantenerse 1,5 s para que el robot cambie de comportamiento.

| Emoción | Comportamiento |
|---|---|
| Feliz | Se acerca a la persona hasta 0,5 m |
| Triste | Se acerca despacio y se queda a 0,8 m (acompañar) |
| Enojo / Disgusto | Busca una mesa con el lidar (detecta sus patas) y se esconde debajo |
| Miedo | Retrocede hasta quedar a 1,5 m de la persona |
| Sorpresa | Da una vuelta sobre sí mismo |
| Neutral | Se queda quieto mirando a la persona |

Al arrancar se supone que **la persona está delante del robot, a 1 m**; después se la sigue con el lidar. El robot no avanza si hay un obstáculo a menos de 20 cm en su camino.

El estado activo se publica en `/turtlebot_emotion/state`. Para probar sin cámara se puede forzar una emoción:
```bash
rostopic pub -1 /emotion_override std_msgs/String "data: 'enojo'"   # "auto" vuelve a la cámara/voz
```

## Instalación
Probado en **Ubuntu 20.04 + ROS Noetic** (Python 3.8). Otras combinaciones de versiones de MediaPipe, TensorFlow y FER dan conflictos.

```bash
sudo apt install ros-noetic-desktop-full ros-noetic-turtlebot3 ros-noetic-turtlebot3-simulations \
                 libportaudio2 ffmpeg
pip3 install --user -r requirements.txt
```

Compilar el workspace:
```bash
cd ~/Escritorio/TFG
catkin_make
source /opt/ros/noetic/setup.bash && source devel/setup.bash
```

La primera vez que se ejecuta se descargan solos el modelo de HSEmotion (en `~/.hsemotion/`) y los de voz.

## Uso
La forma más sencilla es el menú:
```bash
rosrun human_detection launch_menu.py
```

| Opción | Qué hace |
|---|---|
| Avatar → Cara + imágenes | Modo 1 |
| Avatar → Voz + cara + imágenes | Modo 2 (pide elegir micrófono) |
| Avatar → Voz + cara + vídeos | Modo 3 (pide elegir micrófono) |
| Avatar → Cara + vídeos de gestos | Modo 4 |
| TurtleBot → Simulación con cámara | Gazebo + webcam |
| TurtleBot → Simulación con emociones a elegir | Gazebo + un panel para mandar emociones a mano |
| TurtleBot → Robot real | Webcam + TurtleBot real |

También se puede lanzar directamente:
```bash
roslaunch human_detection human_detection.launch mode:=3 use_voice:=true microphone:=14
roslaunch human_detection turtlebot_emotion.launch gazebo:=true            # simulación
roslaunch human_detection turtlebot_emotion.launch                         # robot real
```

Parámetros útiles:
- `use_camera:=false` / `use_voice:=true` en `turtlebot_emotion.launch`
- `~emotion_backend:=fer` en `ros_node.py` para usar FER en vez de HSEmotion

### Conexión con el TurtleBot3 real
1. Conectar el PC y el robot a la misma red y apuntar la IP de cada uno (`hostname -I`).
2. En el `~/.bashrc` del **PC**:
   ```bash
   export ROS_MASTER_URI=http://IP_DEL_PC:11311
   export ROS_HOSTNAME=IP_DEL_PC
   export TURTLEBOT3_MODEL=burger
   ```
3. En el `~/.bashrc` del **robot** (por SSH):
   ```bash
   export ROS_MASTER_URI=http://IP_DEL_PC:11311
   export ROS_HOSTNAME=IP_DEL_ROBOT
   export TURTLEBOT3_MODEL=burger
   ```
4. Arrancar `roscore` en el PC y, en el robot, `roslaunch turtlebot3_bringup turtlebot3_robot.launch`.
5. Comprobar en el PC que llegan `/scan` y `/odom` (`rostopic list`).
6. Colocar el robot mirando a la persona, a ~1 m, y lanzar `launch_menu.py` → **Robot real**.

Para esconderse hace falta una mesa con patas finas (menos de 9 cm) separadas entre 0,4 y 1,6 m.

## Tests
```bash
cd src/human_detection
python3 -m pytest test/
```

## Tecnologías y modelos
- **ROS Noetic** y **Gazebo** con **TurtleBot3**
- **OpenCV** para el vídeo y **NumPy** para los cálculos
- **MediaPipe Holistic** para los puntos del cuerpo, la cara y las manos
- **HSEmotion** (`enet_b2_7`, AffectNet) con onnxruntime para la emoción facial
- **FER** como alternativa para la emoción facial
- **Whisper** para transcribir la voz, **RoBERTa** (`cardiffnlp/twitter-roberta-base-emotion`) para la emoción del texto y **wav2vec 2.0** (`superb/wav2vec2-base-superb-er`) para la del tono
- **PyOpenGL** y **GLFW** para el avatar 3D

## Anexo: landmarks del cuerpo (MediaPipe Pose)
| Índice | Parte del cuerpo         |
| ------ | ------------------------ |
|      0 | Nariz                    |
|      1 | Ojo izquierdo interno    |
|      2 | Ojo izquierdo            |
|      3 | Ojo izquierdo externo    |
|      4 | Ojo derecho interno      |
|      5 | Ojo derecho              |
|      6 | Ojo derecho externo      |
|      7 | Oreja izquierda          |
|      8 | Oreja derecha            |
|      9 | Boca izquierda           |
|     10 | Boca derecha             |
|     11 | Hombro izquierdo         |
|     12 | Hombro derecho           |
|     13 | Codo izquierdo           |
|     14 | Codo derecho             |
|     15 | Muñeca izquierda         |
|     16 | Muñeca derecha           |
|     17 | Meñique izquierdo (base) |
|     18 | Meñique derecho (base)   |
|     19 | Índice izquierdo (base)  |
|     20 | Índice derecho (base)    |
|     21 | Pulgar izquierdo (base)  |
|     22 | Pulgar derecho (base)    |
|     23 | Cadera izquierda         |
|     24 | Cadera derecha           |
|     25 | Rodilla izquierda        |
|     26 | Rodilla derecha          |
|     27 | Tobillo izquierdo        |
|     28 | Tobillo derecho          |
|     29 | Talón izquierdo          |
|     30 | Talón derecho            |
|     31 | Punta del pie izquierdo  |
|     32 | Punta del pie derecho    |

Las puntas de los dedos de la mano (MediaPipe Hands) son los índices 4, 8, 12, 16 y 20.
