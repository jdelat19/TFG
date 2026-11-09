# Detector de gestos y emociones TODO: Cambiar por el nombre del TFG

## Descripción
Este proyecto **detecta gestos y emociones de una persona en tiempo real** con la cámara del ordenador y, opcionalmente, con el micrófono. Está hecho como paquete de **ROS Noetic** (`human_detection`) y la información detectada se usa de dos formas:

- **Avatar**: una ventana muestra una imagen o un vídeo según la emoción o el gesto detectado, con las manos dibujadas en 3D.
- **TurtleBot2**: el robot (simulado en Gazebo o real, con un lidar 360º) reacciona a la emoción de la persona: se acerca, retrocede, se esconde debajo de una mesa, etc.

La cámara obtiene los puntos clave del cuerpo, la cara y las manos (landmarks) con MediaPipe. A partir de sus posiciones se identifican gestos como cruzar los brazos, rascarse el cuello o llevarse las manos a la cara. La emoción facial se obtiene con HSEmotion y la de la voz con Whisper, wav2vec 2.0 y RoBERTa.

## Estructura
```
TFG/                                  # Workspace de catkin (ROS Noetic)
├── install.sh                        # Instalación completa en Ubuntu 20.04
├── requirements.txt                  # Dependencias de Python
├── requirements-lock.txt             # Versiones exactas (las usa install.sh)
├── Img/                              # Capturas de gestos y emociones
└── src/human_detection/              # Paquete ROS
    ├── launch/
    │   ├── human_detection.launch    # Detección + avatar (+ voz)
    │   ├── turtlebot_emotion.launch  # Detección + comportamiento del TurtleBot
    │   └── turtlebot_sim.launch      # Gazebo con TurtleBot2, mesa y persona
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
    ├── urdf/                         # Modelo del TurtleBot2 con lidar para Gazebo
    ├── worlds/                       # Mundos de Gazebo para probar el TurtleBot
    └── test/                         # Tests (python3 -m pytest test/)
```

## Cómo funciona

```
 Cámara ──> ros_node.py ──/human_state──┬──> avatar_node_3d.py   (avatar)
                                        │
 Micrófono ──> voice_node.py ──/voice_emotion──┤
                                        │
                                        └──> turtlebot_emotion_node.py ──/mobile_base/commands/velocity──> TurtleBot2
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
`turtlebot_emotion_node.py` lee la emoción, el lidar (`/scan`) y la odometría (`/odom`) y mueve la base Kobuki por `/mobile_base/commands/velocity`. Una emoción tiene que mantenerse 1,5 s para que el robot cambie de comportamiento.

| Emoción | Comportamiento |
|---|---|
| Feliz | Se acerca a la persona hasta 0,6 m |
| Triste | Se acerca despacio y se queda a 0,9 m (acompañar) |
| Enojo / Disgusto | Busca una mesa con el `/scan` (detecta sus patas) y se esconde debajo |
| Miedo | Retrocede hasta quedar a 1,5 m de la persona |
| Sorpresa | Da una vuelta sobre sí mismo |
| Neutral | Se queda quieto mirando a la persona |

Al arrancar se supone que **la persona está delante del robot, a 1 m**; después se la sigue con el lidar. Solo cuentan como persona los objetos con su forma a la altura del lidar: un cuerpo de 15 a 60 cm de ancho o dos piernas juntas, así que no se confunde con patas de sillas o mesas ni con paredes. El robot no avanza si hay un obstáculo a menos de 35 cm del lidar (unos 17 cm del borde de la base) en su camino. Las distancias se miden desde el centro del robot, donde va el lidar.

### Esconderse debajo de una mesa
El lidar solo ve un corte horizontal a 43 cm, así que la mesa se reconoce por sus patas:
- Tres patas en ángulo recto (una esquina) con lados de 0,6 a 1,6 m. Las sillas (lados de unos 40 cm) no cuentan.
- La cuarta pata tiene que verse o quedar tapada por otra pata de la misma mesa.
- Dentro del rectángulo de patas no puede haber nada.
- Los puntos sueltos de una pared vista de lado no cuentan como patas.

Si el `/scan` cubre menos de 300° (por ejemplo, el que saca `3dsensor.launch` de la cámara de profundidad, unos 63°), antes de buscar da una vuelta de 360° en el sitio acumulando los puntos en odometría y busca la mesa y la dirección de huida sobre ese barrido virtual. Con un lidar de 360° no gira.

Para que la odometría no descuadre la mesa:
- Los puntos del `/scan` se pasan al centro del robot con el TF del sensor (en el TurtleBot real la cámara va 8,7 cm por detrás y 5 cm a la izquierda del centro) y se colocan en odometría con la pose del instante del scan, no con la última.
- En la vuelta solo se usan los puntos vistos cerca del centro de la cámara, así cada pata sale de unos pocos scans seguidos y no se emborrona aunque la odometría derive al girar.
- Tras la vuelta se gira hacia la mesa y vuelve a localizar en directo las dos patas del lado por el que va a entrar (busca la pareja con la misma separación).
- Mientras ve esas dos patas se guía por ellas y no por la odometría. Con la cámara (63°) se coloca lo bastante lejos de la mesa para verlas las dos (unos 0,8–1,1 m), se pone perpendicular al lado y entra siguiendo la recta que pasa por su centro. Los últimos ~50 cm, cuando ya no las ve, van por odometría, que en un tramo recto tan corto apenas se desvía.

Elige el lado más cercano, se coloca delante (rodeando las patas que se le crucen), se alinea y entra. Si se atasca al entrar, choca con el bumper o no consigue llegar, descarta esa mesa y busca otra. Si no encuentra ninguna, se aleja de la persona por la zona más despejada y vuelve a buscar; tras 3 intentos se queda quieto.

La cámara de profundidad no ve nada a menos de 45 cm, así que el filtro de seguridad solo puede parar el robot ante obstáculos a partir de ~40 cm del centro; por debajo de eso actúa el bumper del Kobuki.

Con `debug_dir:=<carpeta>` en `turtlebot_emotion.launch` se guarda cada vuelta de búsqueda (`.npz` con los puntos en odometría y el barrido virtual) para revisar por qué no ha encontrado una mesa.

**Limitación**: el lidar no ve la altura. Asientos y tableros por debajo de unos 45 cm no se ven, y una mesa demasiado baja solo se detecta al chocar con ella (a 0,1 m/s).

### Mundos de simulación
| Mundo | Qué hay | Qué debería pasar con "enojo" |
|---|---|---|
| `emotion_room` | Habitación con una mesa | Se esconde debajo de la mesa |
| `emotion_far_table` | Pasillo con la mesa al fondo | Huye dos veces y encuentra la mesa en la tercera búsqueda |
| `emotion_furniture` | Dos mesas, sillas y una mesa auxiliar de 47 cm | Se atasca en la mesa baja, la descarta y busca otra |
| `emotion_empty` | Habitación vacía | Huye 3 veces y se queda quieto |

Además del mundo, el menú pregunta el sensor del robot simulado:
- **Lidar 360°** con odometría perfecta (`sim_sensor:=lidar sim_odom:=world`, por defecto).
- **Cámara de profundidad** como la del robot real: 63°, nada a menos de 0,45 m, en la misma posición que en el TurtleBot, y odometría integrada de las ruedas, que deriva (`sim_sensor:=camera sim_odom:=encoder`). La de Gazebo deriva bastante más que la del Kobuki real, que usa el giróscopo, así que es un caso peor que el real.

Sin menú: `roslaunch human_detection turtlebot_emotion.launch gazebo:=true world:=emotion_furniture sim_sensor:=camera sim_odom:=encoder`.

Las simulaciones del menú siempre usan un roscore local (`localhost:11311`) aunque `ROS_MASTER_URI` apunte al robot, para no mezclar la simulación con el robot real.

El estado activo se publica en `/turtlebot_emotion/state`. Para probar sin cámara se puede forzar una emoción:
```bash
rostopic pub -1 /emotion_override std_msgs/String "data: 'enojo'"   # "auto" vuelve a la cámara/voz
rostopic pub -1 /emotion_override std_msgs/String "data: 'parar'"   # se para hasta recibir otra emoción
```

## Instalación
Probado en **Ubuntu 20.04 + ROS Noetic** (Python 3.8). Otras combinaciones de versiones de MediaPipe, TensorFlow y FER dan conflictos.

En un ordenador nuevo con Ubuntu 20.04 basta con:
```bash
git clone https://github.com/jdelat19/TFG.git ~/Escritorio/TFG
cd ~/Escritorio/TFG
./install.sh
```

`install.sh` hace todo lo necesario (se puede volver a ejecutar sin problema):
1. Instala ROS Noetic (`desktop-full`) y las librerías del sistema con `apt` (pide la contraseña de `sudo`).
2. Crea un entorno virtual en `.venv` con las versiones exactas de `requirements-lock.txt`, las del PC donde se desarrolló (unos 3 GB de descarga).
3. Compila el workspace para que los nodos usen el Python de `.venv` (no hace falta activarlo) y añade ROS al `~/.bashrc`.
4. Comprueba que todo carga y pasa los tests.

`requirements.txt` es la lista corta de lo que usa el proyecto; `requirements-lock.txt` tiene todas las dependencias con su versión exacta.

La simulación usa un modelo propio del TurtleBot2 (`urdf/turtlebot2.urdf.xacro`) con los plugins de `gazebo_plugins`, así que no hace falta instalar `kobuki_gazebo` (no está en apt para Noetic).

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
| TurtleBot → Robot real | Cámara del robot + TurtleBot2 real |
| TurtleBot → Robot real con emociones a elegir | TurtleBot2 real + un panel para mandar emociones a mano (incluye "parar") |

También se puede lanzar directamente:
```bash
roslaunch human_detection human_detection.launch mode:=3 use_voice:=true microphone:=14
roslaunch human_detection turtlebot_emotion.launch gazebo:=true            # simulación
roslaunch human_detection turtlebot_emotion.launch                         # robot real
```

Parámetros útiles:
- `use_camera:=false` / `use_voice:=true` en `turtlebot_emotion.launch`
- `cmd_vel_topic:=/cmd_vel_mux/input/navi` si el robot usa `cmd_vel_mux` (por defecto `/mobile_base/commands/velocity`)
- `~emotion_backend:=fer` en `ros_node.py` para usar FER en vez de HSEmotion

### Conexión con el TurtleBot2 real
El TurtleBot2 lleva sus drivers (Kobuki y cámara de profundidad) en el portátil que va encima. Necesita publicar `/odom` y `/scan` (de un lidar 360° o de la cámara de profundidad con `3dsensor.launch`) y escuchar `/mobile_base/commands/velocity`.

1. Conectar el PC y el portátil del robot a la misma red y apuntar la IP de cada uno (`hostname -I`).
2. En el `~/.bashrc` del **PC**:
   ```bash
   export ROS_MASTER_URI=http://IP_DEL_PC:11311
   export ROS_HOSTNAME=IP_DEL_PC
   ```
3. En el `~/.bashrc` del **portátil del robot** (por SSH):
   ```bash
   export ROS_MASTER_URI=http://IP_DEL_PC:11311
   export ROS_HOSTNAME=IP_DEL_ROBOT
   ```
4. Arrancar `roscore` en el PC y, en el robot, la base y el sensor (`roslaunch turtlebot_bringup minimal.launch` y `roslaunch turtlebot_bringup 3dsensor.launch`, o el driver del lidar si lo tiene).
5. Comprobar en el PC que llegan `/scan` y `/odom` (`rostopic list`, `rostopic hz /scan`).
6. Colocar el robot mirando a la persona, a ~1 m, y lanzar `launch_menu.py` → **Robot real** (emoción de la cámara) o **Robot real con emociones a elegir** (emociones a mano).

Con **Robot real con emociones a elegir** el menú comprueba antes que llegan `/scan` y `/odom`, lanza el nodo del robot sin cámara y abre un panel para mandar emociones a mano. La opción **parar** detiene el robot al momento; al salir del panel se manda "parar" y se cierra todo. El log queda en `~/.ros/log/tfg_menu_robot_real.log` y las vueltas de búsqueda de mesa en `~/.ros/tfg_vueltas/`.

Para esconderse hace falta una mesa más alta que el robot (el TurtleBot2 mide unos 45 cm) con patas finas (menos de 9 cm) separadas entre 0,6 y 1,6 m, para que quepa la base de 35 cm. Con la cámara de profundidad las patas tienen que verse a la altura de la cámara (unos 30 cm del suelo), sin travesaños ni cajones a esa altura, y conviene dejar ~1,5 m libres delante del lado por el que va a entrar.

## Tests
```bash
cd src/human_detection
python3 -m pytest test/
```

## Tecnologías y modelos
- **ROS Noetic** y **Gazebo** con **TurtleBot2** (base Kobuki + lidar 360º)
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
