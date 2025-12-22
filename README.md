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

