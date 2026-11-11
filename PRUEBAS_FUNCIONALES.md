# Pruebas funcionales

Batería de **30 pruebas** para comprobar que todo el sistema funciona y para evaluarlo: dónde es
fiable, dónde empieza a fallar y por qué. Están pensadas para **grabarse**: cada una dice qué tiene
que verse en el vídeo. Van de lo más básico (instalación) a lo más completo (robot real).

| Bloque | Pruebas | Qué cubre | Necesita |
|---|---|---|---|
| **A** | A1-A3 | Instalación, tests automáticos y menú | PC |
| **B** | B1-B7 | Detección de emociones y gestos con la cámara | PC + webcam |
| **C** | C1-C3 | Emoción por voz | PC + micrófono |
| **D** | D1-D4 | Avatar (los 4 modos) | PC + webcam (+ micrófono) |
| **E** | E1-E7 | TurtleBot2 en Gazebo | PC |
| **F** | F1-F6 | TurtleBot2 real | PC + robot + mesa |

En cada prueba, **"Lo esperable"** es una predicción sacada del código, no un resultado medido.
Si el sistema hace otra cosa, ese es el dato interesante: anotarlo.

## Preparación

- **Instalación** hecha con `./install.sh` y una terminal nueva abierta (ver [README.md](README.md)).
- **Grabación**: grabar la pantalla (por ejemplo con `Ctrl+Alt+Mayús+R` en Ubuntu, o OBS) y, en las
  pruebas del robot y del avatar, también a la persona o al robot con un móvil. Nombrar cada vídeo
  con el código de la prueba (`B3_gestos_mano_cara.mp4`).
- **Cámara**: buena luz de frente, la persona a ~1 m, con la cara y el torso dentro del encuadre.
  La imagen sale en espejo.
- **Ver qué pasa por dentro** (útil en muchas pruebas), en otra terminal:
  ```bash
  rostopic echo /human_state/data -n 1         # gesto y emoción que se publican
  rostopic echo /turtlebot_emotion/state       # emoción, comportamiento y fase del robot
  ```
- **Repeticiones**: las medidas (aciertos, tiempos, distancias) al menos **3 veces**.
- **Seguridad con el robot real**: una persona junto al robot y otra en el teclado, con el panel en
  **parar** a mano. Zona despejada de al menos 3 × 3 m alrededor de la mesa.

---

## A. Instalación y puesta en marcha

### A1. Instalación desde cero

- **Qué mide:** que una persona sin experiencia pueda dejar el sistema funcionando.
- **Cómo:** en un Ubuntu 20.04 recién instalado (o una máquina virtual), seguir solo el apartado
  *Instalación* del README: clonar y `./install.sh`. Cronometrar. Volver a lanzar `./install.sh`
  al terminar.
- **Lo esperable:** termina con *"Instalación completa"* y los tests en verde, en 20-60 minutos
  según la conexión. La segunda ejecución se salta lo instalado y acaba mucho antes. Si se corta
  internet a mitad, al volver a lanzarlo continúa.
- **Grabar:** el comando, los pasos (`==> ...`) y el mensaje final.

### A2. Tests automáticos

- **Qué mide:** la lógica sin cámara ni robot: gestos, emociones y reglas de voz, detección de mesas
  y personas en un lidar simulado, comportamientos del robot e integridad del paquete.
- **Cómo:** `cd src/human_detection && ../../.venv/bin/python -m pytest -v test/`
- **Lo esperable:** todos los tests pasan (63) en menos de un segundo.
- **Grabar:** la salida completa con `-v`.

### A3. Menú de lanzamiento

- **Qué mide:** que el menú es usable y que cada opción abre lo que dice.
- **Cómo:** `rosrun human_detection launch_menu.py`. Moverse con las flechas, con `j`/`k` y con los
  números; entrar en un submenú y volver con `q`/`Esc`; abrir *Avatar → Cara + imágenes* y cerrarlo
  con `Ctrl+C`; salir con *Salir*. Repetir lanzándolo con un roscore ya abierto (`roscore &`) y
  elegir una simulación.
- **Lo esperable:** la opción marcada se resalta; `Ctrl+C` cierra lo lanzado y vuelve al menú (no lo
  cierra). Con un roscore local abierto, la simulación avisa *"Ya hay un ROS en marcha"* antes de
  abrir un segundo Gazebo.
- **Grabar:** la terminal durante toda la navegación.

---

## B. Detección con la cámara

Lanzar *Avatar → Cara + imágenes* y mirar la ventana **ROS Human Node**: arriba sale `Gesto: ...` y
debajo `Emocion: ...` con las tres emociones más probables.

### B1. Las siete emociones faciales

- **Qué mide:** la tasa de acierto del reconocimiento de emociones.
- **Cómo:** poner cada expresión (feliz, triste, enojo, sorpresa, miedo, disgusto y neutral)
  durante 5 s, 3 veces cada una, y anotar qué emoción se queda en pantalla. Repetirlo con 3
  personas distintas.
- **Lo esperable:** feliz, sorpresa y neutral son las más fiables. Miedo, disgusto y tristeza son
  las más difíciles: con una expresión poco marcada sale **neutral**, porque solo se acepta una
  emoción si su probabilidad media llega a 0,45 y le saca 0,15 a la segunda. Confusiones típicas:
  miedo ↔ sorpresa y disgusto ↔ enojo. Tarda ~0,5 s en cambiar (media de 6 fotogramas a ~10 Hz).
- **Grabar:** la cara y la ventana a la vez (pantalla partida o el móvil apuntando a ambas).

### B2. Sin cara, cara parcial y varias personas

- **Qué mide:** cómo se comporta cuando la entrada no es la ideal.
- **Cómo:** (a) salir del encuadre 5 s y volver; (b) taparse media cara con la mano; (c) ponerse de
  perfil; (d) dos personas delante, una feliz y otra seria.
- **Lo esperable:** (a) mantiene la última emoción unos 3 s (30 fotogramas) y luego pasa a *No
  detectado*; al volver, la recupera enseguida. (b) y (c) suelen dar neutral o cambios erráticos.
  (d) MediaPipe sigue a **una sola persona** (normalmente la más visible): solo cuenta su emoción.
- **Grabar:** la ventana en cada caso.

### B3. Gestos del cuerpo

- **Qué mide:** los gestos corporales: `cruzar_brazos`, `brazos_abiertos`, `manos_en_caderas` y
  `manos_juntas`.
- **Cómo:** con el torso entero a la vista (~1,5 m), hacer cada gesto 10 veces durante 2 s y
  anotar cuántas veces aparece. Repetir a 1 m y a 2,5 m.
- **Lo esperable:** el gesto aparece ~1 s después de empezar (tiene que verse en 7 de los últimos
  12 fotogramas) y desaparece ~1 s después de soltarlo. Los umbrales están en píxeles para una
  webcam de 640 × 480 a ~1 m: más lejos, el cuerpo es más pequeño y `brazos_abiertos` cuesta más;
  más cerca, `manos_juntas` salta con facilidad. Cruzar los brazos gana a manos juntas (más
  prioridad).
- **Grabar:** a la persona entera y la ventana.

### B4. Gestos de manos en la cara y la cabeza

- **Qué mide:** `rascarse_cuello`, `morderse_unas`, `manos_en_cara`, `tocarse_cabeza` y
  `cabeza_inclinada`.
- **Cómo:** cada gesto 10 veces durante 2 s. Comparar con las capturas de `docs/capturas/`.
- **Lo esperable:** morderse las uñas y cabeza inclinada tienen la prioridad más alta (9) y ganan a
  manos en la cara cuando se dan a la vez. Las manos pegadas a la cara se pierden a veces (MediaPipe
  no encuentra la mano sobre la piel de la cara), así que estos gestos fallan más que los del
  cuerpo. Inclinar la cabeza salta a partir de ~17°.
- **Grabar:** primer plano de la cara y la ventana.

### B5. Gestos por la forma de la mano

- **Qué mide:** `puño`, `mano_abierta`, `saludar`, `ok`, `cuernos`, `llamame`, `pulgar_arriba`,
  `pulgar_abajo`, `señalar` y `paz`.
- **Cómo:** con la mano a 0,5 m de la cámara, cada gesto 10 veces con cada mano. Repetir girando la
  mano 45° y con la palma y el dorso. Para saludar, mover la mano abierta de lado a lado.
- **Lo esperable:** los gestos por forma (puño, mano abierta, ok, cuernos, llámame, pulgar abajo)
  funcionan aunque la mano esté girada, porque comparan distancias a la muñeca. Pulgar arriba, paz y
  señalar usan posiciones verticales y fallan con la mano girada. Saludar necesita al menos dos
  cambios de sentido en los últimos 20 fotogramas. Confusión más probable: pulgar ↔ puño.
- **Grabar:** la mano y la ventana.

### B6. Condiciones difíciles

- **Qué mide:** la robustez de la detección fuera del caso ideal.
- **Cómo:** repetir 3 emociones (feliz, sorpresa, neutral) y 3 gestos (puño, mano abierta,
  cruzar brazos) con: contraluz (ventana detrás), poca luz, gafas, a 0,5 m, 1,5 m y 3 m.
- **Lo esperable:** la emoción necesita la cara grande y bien iluminada: a 3 m o a contraluz cae a
  neutral o *No detectado*. Los gestos del cuerpo aguantan mejor la distancia que los de la mano.
- **Grabar:** una toma de cada condición con la ventana visible.

### B7. Rendimiento

- **Qué mide:** si el PC procesa a la velocidad prevista (10 fotogramas por segundo).
- **Cómo:** con la detección en marcha, en otra terminal `rostopic hz /human_state`. Medirlo solo,
  con el avatar 3D abierto y con la voz activa (*Voz + cara + vídeos*).
- **Lo esperable:** ~10 Hz. Si baja de ~6 Hz, los gestos tardan más en aparecer (el suavizado es por
  fotogramas). Con la voz activa puede bajar cada 4 s mientras Whisper transcribe.
- **Grabar:** la salida de `rostopic hz` en cada caso.

---

## C. Voz

Lanzar *Avatar → Voz + cara + imágenes*, elegir el micrófono y mirar la terminal: cada 4 s sale
`Voz: <emoción> | Texto: <transcripción>`.

### C1. Frases clave

- **Qué mide:** las frases de `config/emociones.json`, que mandan sobre los modelos.
- **Cómo:** decir, con tono neutro, *"estoy contento"*, *"estoy triste"*, *"te odio"*, *"tengo
  miedo"*, *"no me lo esperaba"* y *"hola, qué tal"*. Después decir *"estoy contento"* con voz
  enfadada.
- **Lo esperable:** feliz, triste, enojo, miedo, sorpresa y neutral, en ese orden, sin importar el
  tono: la frase clave tiene prioridad. Ojo: tiene que contener la frase exacta (*"estoy muy
  contento"* no contiene *"estoy contento"* y va a los modelos). Una frase dicha a caballo entre dos fragmentos de 4 s se
  corta y puede no reconocerse: repetirla.
- **Grabar:** la terminal con las líneas `Voz:` y el audio.

### C2. Frases libres y tono

- **Qué mide:** los modelos de texto (RoBERTa) y de tono (wav2vec 2.0) cuando no hay frase clave.
- **Cómo:** *"hoy ha sido un día horrible"*, *"me ha tocado la lotería"*, *"son las tres y media"*
  (con voz alegre, triste y enfadada) y una palabra suelta gritando (*"¡no!"*).
- **Lo esperable:** las frases con carga emocional clara salen por el texto. Si el texto es neutral,
  decide el tono (solo distingue neutral, feliz, enojo y triste). Con texto alegre y tono triste
  sale *mixta* (el avatar no la muestra). Los modelos están entrenados en inglés, así que fallan más
  con frases en español sin palabras clave.
- **Grabar:** terminal y audio.

### C3. Silencio, ruido y micrófono

- **Qué mide:** que el ruido no invente emociones.
- **Cómo:** (a) 30 s en silencio; (b) música o conversación de fondo; (c) elegir en el menú un
  micrófono que no existe o desconectarlo antes de lanzar.
- **Lo esperable:** (a) *"Voz: silencio (se mantiene la emoción)"*, sin publicar nada. (b) Whisper
  puede transcribir parte de la letra o de la conversación y publicar una emoción falsa. (c) Usa el
  micrófono por defecto y avisa *"Micrófono N no válido"*.
- **Grabar:** terminal.

---

## D. Avatar

La ventana **Avatar 3D** muestra de fondo una imagen o un vídeo y encima las manos detectadas
(cilindros grises y articulaciones rojas).

### D1. Modo 1: cara + imágenes

- **Qué mide:** que la imagen sigue a la emoción facial y que las manos se dibujan en 3D.
- **Cómo:** *Avatar → Cara + imágenes*. Poner cada emoción y mover las manos delante de la cámara,
  acercándolas y alejándolas.
- **Lo esperable:** cambia la imagen con la emoción. Solo hay imagen de feliz, triste, enojo,
  sorpresa y neutral: **miedo y disgusto muestran la imagen *no detectado***. Las manos se mueven en
  tiempo real y cambian de profundidad al acercarlas.
- **Grabar:** la persona y la ventana del avatar a la vez.

### D2. Modo 2: la voz manda

- **Qué mide:** la fusión de cara y voz.
- **Cómo:** *Avatar → Voz + cara + imágenes*. Con cara seria, decir *"estoy contento"*; luego,
  sonriendo, decir *"estoy triste"*; luego callar 20 s.
- **Lo esperable:** la imagen sigue a la voz cuando no es neutral, aunque la cara diga otra cosa. La
  última emoción de la voz **se mantiene** mientras solo haya silencio (no se publica nada); vuelve a
  mandar la cara cuando un fragmento con voz da neutral (p. ej. decir *"hola"*).
- **Grabar:** persona, terminal y avatar.

### D3. Modo 3: vídeos

- **Qué mide:** los vídeos de cada emoción.
- **Cómo:** *Avatar → Voz + cara + vídeos*. Pasar por las 7 emociones (con la cara o con frases
  clave para miedo y disgusto).
- **Lo esperable:** hay vídeo de las 7 emociones; cada uno se repite en bucle y cambia al cambiar
  la emoción.
- **Grabar:** el avatar.

### D4. Modo 4: vídeos de gestos

- **Qué mide:** el avatar guiado por el gesto.
- **Cómo:** *Avatar → Cara + vídeos de gestos*. Rascarse el cuello, luego otro gesto (puño), luego
  nada.
- **Lo esperable:** en este modo la emoción facial está desactivada (*Emocion: Desactivado*).
  Rascarse el cuello reproduce su vídeo; **cualquier otro gesto o ninguno muestra el vídeo *no
  detectado***, porque `rascarse_cuello.mp4` es el único vídeo de gesto. Para añadir más basta con
  copiar `<gesto>.mp4` en `media/videos/`.
- **Grabar:** persona y avatar.

---

## E. TurtleBot2 en simulación

Para controlar el robot a mano: *TurtleBot → Simulación con emociones a elegir*. Se abren Gazebo y
el **mapa mental**. La persona es el cilindro que tiene delante, a 1 m.

### E1. Comportamientos básicos

- **Qué mide:** que cada emoción produce su comportamiento.
- **Cómo:** mundo *Habitación con mesa*, sensor *Lidar 360º*. En el panel, mandar en orden: feliz,
  neutral, triste, miedo, sorpresa y parar, dejando unos 15 s entre cada una.
- **Lo esperable:** feliz → se acerca hasta 0,6 m del centro de la persona y se para; neutral → se
  queda mirándola; triste → se queda a 0,9 m (retrocede si está más cerca); miedo → retrocede hasta
  1,5 m sin dejar de mirarla; sorpresa → una vuelta de 360° y se para; parar → se detiene al
  instante. El mapa mental muestra la emoción, el comportamiento y la persona (magenta).
- **Grabar:** Gazebo, el mapa mental y el panel.

### E2. Esconderse con lidar

- **Qué mide:** el comportamiento principal en el caso fácil.
- **Cómo:** mismo mundo y sensor; mandar **enojo**. Cuando esté debajo, mandar feliz.
- **Lo esperable:** en ~40 s: elige la mesa (rojo en el mapa), va al punto de entrada, se pone
  perpendicular, entra despacio y la terminal del log (o `rostopic echo /turtlebot_emotion/state`)
  llega a la fase `hidden` (*"Escondido: estoy entre las patas"*). Al mandar feliz, **primero sale
  marcha atrás** y luego se acerca a la persona.
- **Grabar:** Gazebo y mapa mental; es el vídeo de referencia del proyecto.

### E3. Esconderse con la cámara de profundidad

- **Qué mide:** el mismo caso con el sensor del robot real (63°, nada a menos de 45 cm y odometría
  con deriva).
- **Cómo:** mundo *Habitación con mesa*, sensor *Cámara de profundidad*; mandar enojo.
- **Lo esperable:** primero da una **vuelta de 360°** acumulando puntos (en gris en el mapa), elige
  la mesa, gira hacia ella y la vuelve a localizar en directo (*"Mesa localizada de nuevo"*) antes de
  ir. Puede descartar primero una mesa falsa (*"No vuelvo a ver la mesa al mirarla"*). Termina
  escondido en 1-2 minutos.
- **Grabar:** Gazebo y mapa mental.

### E4. Mesa lejana

- **Qué mide:** la huida cuando no ve ninguna mesa.
- **Cómo:** mundo *Pasillo con mesa al fondo* (lidar); mandar enojo.
- **Lo esperable:** no ve la mesa: se aleja de la persona por la zona más despejada (fase `flee`),
  vuelve a buscar, huye otra vez y la encuentra en la **tercera búsqueda**.
- **Grabar:** Gazebo y mapa mental.

### E5. Mesa demasiado baja

- **Qué mide:** la recuperación cuando la mesa elegida no sirve.
- **Cómo:** mundo *Mesas, sillas y mesa baja* (lidar); mandar enojo.
- **Lo esperable:** elige primero la mesa baja (47 cm, el robot mide ~45 cm sin margen), se
  **atasca** al entrar (*"Atascado al entrar"*) o choca, sale marcha atrás, la marca como descartada
  (aspas grises en el mapa) y se esconde en otra mesa.
- **Grabar:** Gazebo y mapa mental.

### E6. Sin mesas

- **Qué mide:** que el comportamiento termina aunque no haya solución.
- **Cómo:** mundo *Habitación vacía*; mandar enojo.
- **Lo esperable:** huye y busca 3 veces y luego se queda quieto (*"No encuentro ninguna mesa, me
  quedo quieto"*). Al huir no se acerca a menos de ~40 cm de las paredes.
- **Grabar:** Gazebo y mapa mental.

### E7. Simulación con la cámara: estabilidad de la emoción

- **Qué mide:** la cadena completa cara → emoción → robot y el filtro que evita cambios bruscos.
- **Cómo:** *TurtleBot → Simulación con cámara*, mundo *Habitación con mesa*. (a) Sonreír de forma
  sostenida; (b) alternar sonrisa y cara seria cada segundo; (c) sonreír 2 s y parar.
- **Lo esperable:** (a) el robot se acerca unos 3 s después de empezar a sonreír. (b) y (c) no
  cambian el comportamiento: una emoción tiene que mantenerse 3 s (8 s si es neutral) y cada
  comportamiento dura al menos 5 s.
- **Grabar:** la cara, la ventana de la cámara y Gazebo.

---

## F. TurtleBot2 real

### F1. Arranque con `run.sh`

- **Qué mide:** que el robot se pone en marcha con un solo comando y que los fallos se detectan.
- **Cómo:** (a) `bash run.sh` con todo bien; (b) con una contraseña incorrecta; (c) con la base
  Kobuki apagada.
- **Lo esperable:** (a) abre Terminator con 5 terminales; pasa *Master de ROS*, *Base Kobuki*,
  *Cámara del robot* y *Láser virtual* en menos de 2 minutos y abre el menú. (b) *"Contraseña
  incorrecta"* sin abrir nada. (c) *"Sin odometría en 30 s"*, cierra Terminator y los roslaunch del
  robot.
- **Grabar:** la terminal de `run.sh` y Terminator.

### F2. Comportamientos básicos en el robot real

- **Qué mide:** E1 con el robot de verdad.
- **Cómo:** menú → *Robot real con emociones a elegir*. Persona de pie a 1 m delante del robot.
  Mandar feliz, neutral, triste, miedo, sorpresa y parar. Medir con un metro la distancia final a la
  persona en feliz, triste y miedo.
- **Lo esperable:** el menú comprueba que llegan `/scan` y `/odom` antes de lanzar. Las distancias
  (desde el centro del robot) se acercan a 0,6, 0,9 y 1,5 m. La cámara solo ve 63° por delante: si
  la persona se sale de ese ángulo, el robot usa su última posición conocida. La vuelta de sorpresa
  se mide con la odometría, así que cierra bien (pocos grados de error).
- **Grabar:** el robot y la persona con el móvil, y el mapa mental.

### F3. Esconderse bajo una mesa real

- **Qué mide:** la funcionalidad principal en el mundo real.
- **Cómo:** una mesa de 4 patas más alta que el robot, con 1,5 m libres delante, a 1-2 m del robot.
  Mandar enojo. Repetir 5 veces cambiando la posición inicial, y luego con una mesa de pie central o
  en T y con una silla grande.
- **Lo esperable:** vuelta de 360°, elección de la mesa (mapa mental), relocalización y entrada
  despacio hasta quedar entre las patas. Anotar cuántas veces termina debajo. Fallos posibles:
  patas finas o tapadas que no se ven en la vuelta, travesaños a la altura de la cámara (~30 cm) que
  parecen una pared, y deriva de la odometría al entrar (la cámara no ve a menos de 45 cm). Las
  vueltas quedan en `~/.ros/tfg_vueltas/` para analizarlas.
- **Grabar:** el robot con el móvil (desde un lado de la mesa) y el mapa mental.

### F4. Seguridad: parar, obstáculos y choques

- **Qué mide:** que el robot se puede parar y que no choca a lo loco.
- **Cómo:** (a) en feliz, mandar **parar** mientras avanza y medir lo que recorre hasta pararse;
  (b) en feliz, poner una caja entre el robot y la persona; (c) en enojo, durante la entrada, poner
  una caja delante para que choque con el bumper.
- **Lo esperable:** (a) se para en menos de medio segundo (el panel publica al momento). (b) no
  avanza si la caja está a menos de ~40 cm del centro del robot (por debajo la cámara no ve: ahí
  actúa el bumper). (c) *"Choque con el bumper"*: descarta esa mesa, sale marcha atrás y busca otra.
- **Grabar:** el robot con el móvil.

### F5. Robot real con la cámara del robot

- **Qué mide:** la cadena completa en el robot: su cámara detecta la emoción y él reacciona.
- **Cómo:** menú → *Robot real*. Ponerse delante de la cámara del robot a ~1 m y sostener feliz,
  luego miedo, luego neutral.
- **Lo esperable:** la ventana muestra la imagen de la cámara del robot (comprimida si está
  disponible, para no saturar la WiFi) con el esqueleto. El robot cambia de comportamiento unos 3 s
  después de cada emoción. Con la persona muy cerca la cara se sale del encuadre y la emoción se
  pierde: es más fiable entre 0,8 y 1,5 m.
- **Grabar:** la persona y el robot con el móvil, y la ventana de la cámara.

### F6. Pérdida de conexión y cierre

- **Qué mide:** que el robot se para solo si se pierde el control.
- **Cómo:** con el robot moviéndose (feliz o miedo): (a) `Ctrl+C` en la terminal del menú; (b)
  desactivar la WiFi del PC; (c) `pkill -9 -f turtlebot_emotion_node`.
- **Lo esperable:** (a) se para al instante (el nodo manda velocidad cero al cerrarse). (b) y (c) la
  base Kobuki se para sola al dejar de recibir órdenes (~0,6 s), unos 10 cm a la velocidad máxima.
  Tras (b), al volver la WiFi, comprobar si hay que reiniciar con `run.sh`.
- **Grabar:** el robot con el móvil y el PC.

---

## Puntos fuertes y débiles esperados

**Fuertes:** emociones claras (feliz, sorpresa, neutral) con buena luz; gestos del cuerpo y de forma
de la mano; frases clave de voz; los comportamientos simples; esconderse con lidar en simulación; la
recuperación cuando una mesa falla; y la parada del robot (panel, `Ctrl+C`, corte de red).

**Débiles:** emociones sutiles (miedo, disgusto, tristeza), que caen a neutral; gestos con las manos
sobre la cara; voz sin frases clave o con ruido de fondo (modelos entrenados en inglés); el campo de
visión estrecho de la cámara del robot real (la persona se pierde y las patas cercanas no se ven); y
la deriva de la odometría en recorridos largos.

## Plantilla de resultados

| Prueba | Variante | Intentos | Aciertos | Medida (s / m / Hz) | Vídeo | Notas |
|---|---|---|---|---|---|---|
| A1 | Ubuntu 20.04 limpio | 1 | | | A1_instalacion.mp4 | |
| B1 | Persona 1, feliz | 3 | | | | |
| E2 | Habitación, lidar | 3 | | | | |
| F3 | Mesa de 4 patas | 5 | | | | |
| ... | | | | | | |
