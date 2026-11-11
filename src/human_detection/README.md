# Paquete `human_detection`: referencia técnica

Guía de instalación y uso: [README principal](../../README.md).

## Arquitectura

```
 Webcam o cámara del robot                               Micrófono
          │                                                  │
 human_detector_node.py ◄──── /voice_emotion ──────── voice_node.py
          │  (MediaPipe + gestos + HSEmotion)                (Whisper + RoBERTa + wav2vec 2.0)
          │                                                  │
     /human_state ─────────────┬─────────────────────────────┘
          │                    │
   avatar_node.py      turtlebot_emotion_node.py ◄── /scan, /odom, bumper, /emotion_override
   (ventana 3D)                │
                               ├──► /mobile_base/commands/velocity ──► TurtleBot2 / Gazebo
                               └──► /turtlebot_emotion/map ──► mental_map_node.py (ventana)
```

| Nodo (`scripts/`) | Qué hace | Lógica en (`src/`) |
|---|---|---|
| `human_detector_node.py` | Cámara → gesto, emoción facial y manos en `/human_state` | `gesture_detector.py`, `gestures.py`, `facial_expression.py` |
| `voice_node.py` | Micrófono (fragmentos de 4 s) → emoción en `/voice_emotion` | `emotions.py` (reglas), `config/emociones.json` |
| `avatar_node.py` | Imagen o vídeo según la emoción, con las manos en 3D (OpenGL) | `emotions.py` (`media_path`) |
| `turtlebot_emotion_node.py` | Emoción → comportamiento → velocidad del robot | `turtlebot_behaviors.py`, `scan_utils.py`, `emotions.py` |
| `mental_map_node.py` | Dibuja lo que el robot cree que hay alrededor | — |
| `launch_menu.py` | Menú de terminal que lanza todo lo anterior | — |

| Launch | Uso |
|---|---|
| `human_detection.launch` | Avatar: `mode:=1..4`, `use_voice:=true`, `microphone:=N` |
| `turtlebot_emotion.launch` | TurtleBot: `gazebo:=true`, `world:=...`, `sim_sensor:=lidar\|camera`, `image_topic:=...`, `map:=true`, `use_camera:=false`, `use_voice:=true`, `debug_dir:=...`, `cmd_vel_topic:=...` |
| `turtlebot_sim.launch` | Solo Gazebo con el TurtleBot2 (`urdf/turtlebot2.urdf.xacro`) |

## Detección

- **Gestos** (`gestures.py`): reglas sobre distancias entre puntos. Si se cumplen varios gana el de
  más prioridad, y para publicarse tiene que aparecer en 7 de los últimos 12 fotogramas.
  - Cuerpo: `cruzar_brazos`, `brazos_abiertos`, `manos_en_caderas`, `manos_juntas`.
  - Manos y cara: `rascarse_cuello`, `morderse_unas`, `manos_en_cara`, `tocarse_cabeza`, `cabeza_inclinada`.
  - Forma de la mano (independiente del giro: compara distancias a la muñeca): `puño`,
    `mano_abierta`, `saludar` (mano abierta de lado a lado), `ok`, `cuernos`, `llamame`,
    `pulgar_arriba`, `pulgar_abajo`, `señalar`, `paz`.
- **Emoción facial** (`facial_expression.py`): HSEmotion sobre la cara recortada con la malla de
  MediaPipe. Se promedian 6 fotogramas y solo se acepta una emoción si su probabilidad llega a 0,45
  y saca 0,15 a la segunda; si no, neutral. Tras 30 fotogramas sin cara: "No detectado".
  `~emotion_backend:=fer` usa FER (TensorFlow) en su lugar.
- **Voz** (`emotions.combine_voice_emotions`): 1) una frase clave de `config/emociones.json`
  dentro del texto manda; 2) si no, el modelo de texto si tiene confianza ≥ 0,40; 3) si el texto es
  neutral, el tono. El silencio no se publica.
- **Fusión**: el avatar (modos 2 y 3) usa la última emoción de la voz si no es neutral, y el robot
  la usa si además tiene menos de 6 s; si no, la de la cara.

## Avatar

Fondo de `media/imagenes/<emoción>.png` (modos 1 y 2) o `media/videos/<emoción>.mp4` (modos 3 y 4;
en el 4 primero `<gesto>.mp4`). Si no existe se usa `no detectado`. Las manos se dibujan con la
profundidad que estima MediaPipe.

## TurtleBot

**Decisión** (`turtlebot_emotion_node.py`): una emoción nueva tiene que mantenerse 3 s (8 s si es
neutral) y el comportamiento actual tiene que llevar al menos 5 s; así no da tirones.
`/emotion_override` se salta esa espera (`auto` vuelve a la cámara, `parar` detiene el robot).
Cada ciclo (10 Hz) limita la aceleración y anula el avance si hay algo a menos de 35 cm en el
pasillo del robot. Sin `/odom` durante 0,5 s, se para.

**Persona**: al arrancar se supone delante, a 1 m. Después se sigue con el `/scan`, aceptando
solo objetos con forma de persona (un cuerpo de 15-60 cm o dos piernas juntas).

**Esconderse** (`turtlebot_behaviors.py`, máquina de estados en `self.st["phase"]`):

1. `search` / `look`: si el `/scan` cubre menos de 300° (cámara de profundidad, ~63°) da una
   vuelta de 360° acumulando los puntos en odometría y arma un barrido virtual.
2. Busca mesas en el barrido (`scan_utils.py`): patas de ≤ 15 cm aisladas; tres en ángulo recto
   con lados de 0,45-1,6 m, la cuarta visible o tapada y nada dentro (4 patas); o dos columnas
   gordas a 0,45-1,8 m (pie central o en T). Elige la más grande con el pasillo de entrada libre.
3. `acquire`: mira hacia la mesa y vuelve a localizar en directo las dos patas del lado de entrada.
4. `goto_pre` → `align` → `enter`: ruta fija por odometría hasta el punto de entrada (rodeando la
   mesa si está en medio), se pone perpendicular y entra por la recta del centro, despacio.
5. `check` / `correct`: comprueba que está entre las patas (10 cm de margen) y corrige hasta 2 veces.
6. Si choca, se atasca o no llega, descarta la mesa y busca otra. Sin mesa, huye de la persona
   (`flee`) y vuelve a buscar; tras 3 intentos se queda quieto (`hidden`).

Con `debug_dir:=<carpeta>` cada vuelta se guarda en `.npz`. Limitación: el sensor solo ve un corte
horizontal, así que un tablero más bajo que el robot solo se detecta al chocar con él.

**Simulación** (`worlds/`): `emotion_room` (una mesa), `emotion_far_table` (mesa al fondo de un
pasillo: la encuentra a la tercera búsqueda), `emotion_furniture` (mesas, sillas y una mesa baja
de 47 cm en la que se atasca) y `emotion_empty` (sin mesa: huye 3 veces y se para). El menú usa
siempre un roscore local para no mezclar la simulación con el robot real.

## Topics

| Topic | Tipo | Contenido |
|---|---|---|
| `/human_state` | String (JSON) | `gesture`, `emotion`, `confidence`, `left/right_hand_landmarks`, `voice_*` |
| `/voice_emotion` | String (JSON) | `text`, `voice_emotion`, `text_emotion`, `final_emotion` |
| `/emotion_override` | String | Emoción forzada, `auto` o `parar` |
| `/turtlebot_emotion/state` | String (JSON) | `emotion`, `behavior`, `phase` |
| `/turtlebot_emotion/map` | String (JSON) | Mapa mental (solo si hay alguien suscrito) |

```bash
rostopic pub -1 /emotion_override std_msgs/String "data: 'enojo'"   # forzar una emoción
rostopic echo /turtlebot_emotion/state                               # ver qué hace el robot
```

## Tests

`../../.venv/bin/python -m pytest test/`: gestos con manos y cuerpos sintéticos, emociones y
reglas de voz, detección de mesas y personas con un lidar simulado, comportamientos con un robot
falso e integridad del paquete (launch, scripts, mundos).

## Anexo: puntos del cuerpo (MediaPipe Pose)

| Índice | Punto | Índice | Punto | Índice | Punto |
|---|---|---|---|---|---|
| 0 | Nariz | 11 / 12 | Hombro izq. / der. | 23 / 24 | Cadera izq. / der. |
| 1-3 / 4-6 | Ojo izq. / der. | 13 / 14 | Codo izq. / der. | 25 / 26 | Rodilla izq. / der. |
| 7 / 8 | Oreja izq. / der. | 15 / 16 | Muñeca izq. / der. | 27 / 28 | Tobillo izq. / der. |
| 9 / 10 | Boca izq. / der. | 17-22 | Meñique, índice y pulgar | 29-32 | Talón y punta del pie |

En cada mano (MediaPipe Hands): 0 muñeca; pulgar 1-4, índice 5-8, corazón 9-12, anular 13-16,
meñique 17-20 (la última de cada dedo es la punta).
