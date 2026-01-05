# Detector de gestos TODO: Cambiar por el nombre del TFG

## Descripcion 
Este proyecto permite **detectar gestos corporales y faciales** en tiempo real usando la cámara del ordenador.  

La idea es capturar el video, detectar los puntos clave del cuerpo (llamados landmarks), y a partir de sus posiciones, identificar si la persona está realizando gestos como cruzar los brazos, rascarse el cuello, llevarse las manos a la cara, etc.
 
## Estructura
```
TFG/
│
├── main.py               # Script principal: captura video y muestra detección
├── gestures.py           # Núcleo del reconocimiento.
├── gesture_detector.py   # Clase que gestiona la detección con MediaPipe
├── utils.py              # Funciones auxiliares (coordenadas, distancia, etc.)
```

## Gestos
### Gestos Corporales
- Brazos cruzados
- Brazos abiertos
- Manos juntas
- Sacudir la pierna

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
- Manos juntas (oración)
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

ros2 run gesture_emotion_ros gesture_emotion_node

ros2 topic echo /gestures_emotions


# Tras agregar ROS
## Funcionalidades

- Captura de video en tiempo real desde la cámara.
- Detección de gestos usando landmarks (MediaPipe).
- Detección de emociones usando FER.
- Publicación de resultados en ROS 2:
  - **Topic:** `/gestures_emotions`
  - **Formato del mensaje:** `Gesto: <nombre> | Emoción: <nombre> (<score>)`

---

## Instalación de dependencias

**ROS 2** y paquetes necesarios:

```bash
sudo apt update
sudo apt install ros-kaiju-cv-bridge python3-opencv
```

**Dependecias de Python**

```bash
pip install --user mediapipe fer tensorflow mtcnn
```

## Construcción e inicio del nodo

Desde la raíz del workspace ROS 2 (ros2_ws):

# Limpiar builds previos
rm -rf build install log

# Construir workspace
colcon build

# Fuente del setup del workspace
source install/setup.bash

# Ejecutar nodo de gestos y emociones
ros2 run gesture_emotion_ros gesture_emotion_node

# Visualización de los datos

* Listar topics de ROS 2:
```bash
ros2 topic list
```

* Ver lo que publica el nodo:
```bash
ros2 topic echo /gestures_emotions
```

## Estructura 
src
    ├── gesture_emotion_ros
    │   ├── gesture_emotion_ros
    │   │   ├── facial_expression.py
    │   │   ├── gesture_detector.py
    │   │   ├── gesture_emotion_node.py
    │   │   ├── gestures.py
    │   │   ├── __init__.py
    │   │   └── utils.py
    │   ├── package.xml
    │   ├── resource
    │   │   └── gesture_emotion_ros
    │   ├── setup.cfg
    │   ├── setup.py