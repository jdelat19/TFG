# Detector de gestos TODO: Cambiar por el nombre del TFG

## Descripcion 
Este proyecto permite **detectar gestos corporales y faciales** en tiempo real usando la cámara del ordenador.  

La idea es capturar el video, detectar los puntos clave del cuerpo (llamados landmarks), y a partir de sus posiciones, identificar si la persona está realizando gestos como cruzar los brazos, rascarse el cuello, llevarse las manos a la cara, etc.
 
## Estructura
```
TFG/                                  # Workspace de catkin (ROS Noetic)
├── Deprecated/                       # Versión anterior sin ROS
├── Img/                              # Capturas de gestos y emociones
└── src/human_detection/              # Paquete ROS
    ├── launch/
    │   ├── human_detection.launch    # Detección + avatar (+ voz)
    │   ├── turtlebot_emotion.launch  # Detección + comportamiento del TurtleBot
    │   └── turtlebot_sim.launch      # Gazebo con TurtleBot3, mesa y persona
    ├── scripts/                      # Nodos ROS
    │   ├── ros_node.py               # Cámara -> gestos y emoción facial (/human_state)
    │   ├── voice_node.py             # Micrófono -> emoción por voz (/voice_emotion)
    │   ├── avatar_node.py            # Avatar 2D (imágenes / vídeos)
    │   ├── avatar_node_3d.py         # Avatar 3D con las manos de MediaPipe
    │   ├── turtlebot_emotion_node.py # TurtleBot que reacciona a la emoción
    │   ├── launch_menu.py            # Menú para elegir modo
    │   └── emociones.json            # Frases clave por emoción para la voz
    ├── src/                          # Módulos
    │   ├── gesture_detector.py       # Detección con MediaPipe Holistic
    │   ├── gestures.py               # Reglas de cada gesto
    │   ├── facial_expression.py      # Emoción facial con FER
    │   ├── scan_utils.py             # Procesado del lidar (patas de mesa, etc.)
    │   └── utils.py                  # Coordenadas y distancias
    ├── media/                        # Imágenes, vídeos y modelos 3D del avatar
    ├── worlds/                       # Mundo de Gazebo
    └── test/                         # Tests (python3 -m pytest test/)
```

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
```
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
```

FINGER_TIPS = [4, 8, 12, 16, 20]

## Tecnologías Utilizadas

- **Python 3** - Lenguaje de programación
- **OpenCV** - Procesamiento de video
- **MediaPipe Holistic** - Detección de landmarks corporales
- **NumPy** - Operaciones numéricas

## Cómo Funciona

1. **Captura de video**: Se obtiene el flujo de la cámara web
2. **Detección de landmarks**: MediaPipe Holistic detecta puntos clave del cuerpo
3. **Análisis de posición**: Se calculan distancias y ángulos entre landmarks
4. **Clasificación de gestos**: Se identifican los gestos basándose en las reglas definidas
5. **Visualización**: Se dibuja el esqueleto y se muestra el gesto detectado

## Modelos usados
- **FER**

