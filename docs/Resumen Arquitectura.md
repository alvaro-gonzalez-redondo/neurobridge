# Conceptual guide/White paper/Architectural overview

## 1. Propósito, público y nicho

El propósito central de esta biblioteca es simular **redes neuronales de disparos (SNN, Spiking Neural Networks)** en un único ordenador equipado con varias GPUs. Su diseño apunta a dos objetivos principales:
1. **Aprovechar al máximo la capacidad de cómputo paralelo de GPUs** para ejecutar simulaciones con gran número de neuronas y sinapsis.    
2. **Reducir al mínimo la latencia de comunicación entre GPUs**, de modo que los resultados puedan emplearse en contextos que requieren respuesta rápida, como la **robótica en tiempo real** o experimentos interactivos.

El público esperado no es el principiante absoluto. Está pensada para:
- Investigadores y desarrolladores con un **nivel intermedio o avanzado en PyTorch**, familiarizados con CUDA, `torch.distributed` y el backend NCCL.    
- Usuarios con **nociones de neurociencia computacional**, capaces de comprender modelos de neurona como _Integrate-and-Fire_ (IF) y reglas de plasticidad como STDP (_Spike-Timing-Dependent Plasticity_).    

El nicho de esta biblioteca se distingue de alternativas conocidas:
- Frente a marcos generales como **NEST** o **Brian2**, esta biblioteca se especializa en **multi-GPU dentro de un solo nodo**, evitando el overhead del cómputo distribuido multinodo.    
- Frente a proyectos como **BindsNET**, aquí el foco está menos en la interfaz amigable y más en la **ejecución eficiente y compacta**, con herramientas como **CUDA Graphs** y **puentes de spikes entre GPUs**.    

En resumen, se posiciona como una herramienta para quienes buscan un **equilibrio entre expresividad y rendimiento**, especialmente cuando la prioridad es ejecutar simulaciones de SNN en tiempo real sobre hardware GPU disponible localmente.

## 2. Modelo de ejecución y escena (árbol de nodos)

La biblioteca organiza la simulación mediante una **estructura jerárquica de nodos**. Cada elemento del modelo —ya sea un grupo de neuronas, una conexión sináptica o un monitor— es un **`Node`**, y puede contener otros nodos hijos.

Existen variantes de esta abstracción:
- **`Node`**: base genérica, sin asignación a un dispositivo concreto.    
- **`GPUNode`**: nodo asociado a una GPU específica, con un atributo `device` que determina dónde se ejecutan sus tensores y operaciones.    
- **`ParentStack`**: mecanismo interno para manejar de forma automática el árbol de nodos. Permite que al declarar un nuevo grupo o conexión dentro de un contexto, este quede asociado al nodo “padre” correcto.    

El ciclo de ejecución se apoya en **dos métodos clave** que cada nodo puede implementar:
- **`_ready()`**: inicialización previa a la simulación. Se recorre de forma recursiva el árbol de nodos para preparar buffers, parámetros y dependencias.    
- **`_process()`**: ejecución de un paso de simulación. También se recorre recursivamente, de modo que cada nodo actualiza su estado en orden.    

Una característica importante es la separación entre:
- El **subárbol “capturable”** (instancia de `CUDAGraphSubTree`), cuyas operaciones se registran en un **CUDA Graph** para ser repetidas eficientemente en cada ciclo.    
- El **resto de nodos**, que realizan tareas accesorias como monitorización o logging, y no forman parte de la captura.    

La evolución temporal se maneja mediante el contador **`LocalCircuit.t`**, un tensor residente en GPU que representa el paso actual de simulación. Este contador avanza en incrementos unitarios, y se combina con un esquema de **indexado circular** de tamaño `delay_max` para los búferes de spikes. Dicho esquema permite modelar retardos sinápticos de hasta `delay_max – 1` pasos, optimizando memoria y acceso.

En conjunto, este modelo jerárquico asegura que:
- Cada componente de la simulación se integre de manera ordenada en el ciclo global.    
- La parte crítica de cómputo quede contenida en un grafo de CUDA, reduciendo el overhead de llamadas sucesivas.    
- El manejo del tiempo y los retardos se mantenga coherente y eficiente en GPU.

## 3. Motor de simulación (`Simulator` y `LocalCircuit`)

El núcleo operativo de la biblioteca lo constituyen **`Simulator`** y **`LocalCircuit`**, responsables de inicializar, coordinar y ejecutar los pasos de simulación.

**Inicialización**
- Se detecta automáticamente el número de GPUs disponibles.
- Si el script se ejecuta con `torchrun` y variables de entorno (`WORLD_SIZE`, `RANK`), se activa el modo distribuido con backend **NCCL**. Cada proceso recibe su `rank` y se asigna a la GPU correspondiente.    
- La clase `LocalCircuit` selecciona el `device` y establece la semilla aleatoria, garantizando reproducibilidad independiente por GPU/rank.    

**Preparación del grafo de ejecución**
- El simulador distingue entre nodos incluidos en el **subárbol capturable** (`CUDAGraphSubTree`) y el resto.    
- Se realiza un **calentamiento** (warm-up) de varios pasos para inicializar kernels y buffers.    
- Opcionalmente, se usa **`torch.compile`** para optimizar funciones.
- Después, se captura un **CUDA Graph** que encapsula el ciclo de `_process()` de todos los nodos capturables.    

**Ejecución**
- Cada llamada a `step()` avanza la simulación un paso de tiempo.    
- Si hay un grafo capturado, se ejecuta directamente con `graph.replay()`, evitando el overhead de llamadas sucesivas a kernels.    
- El contador de tiempo `t` se incrementa y se sincroniza con los búferes circulares de spikes.    

**Finalización**
- Al cerrar la simulación, se liberan de forma segura los grupos distribuidos (`torch.distributed.destroy_process_group()`).
- El objeto `Simulator` limpia recursos y asegura consistencia entre procesos.    

En síntesis, el motor organiza toda la infraestructura técnica para que el usuario pueda centrarse en el modelo neuronal y sináptico, delegando la gestión de GPUs, comunicación entre procesos y captura de ejecución al sistema central.

## 4. Grupos y filtrado (selecciones)

Las **poblaciones neuronales** y otros conjuntos de entidades se organizan en la biblioteca mediante la clase **`Group`** y sus derivados. Su función es manejar colecciones de elementos (p. ej., neuronas o conexiones) de manera vectorizada y flexible.

**Características principales**
- Cada grupo mantiene un atributo `size` que indica el número de elementos.    
- Existe un **vector de filtrado** (`filter`) que especifica qué subconjunto de elementos está activo en cada operación.    
- Los métodos de selección (`where_*`) devuelven copias del grupo con filtros ajustados, sin modificar el original.

**Tipos de selección**
- **`where_id(indices)`**: activa elementos según sus identificadores absolutos dentro del grupo.
- **`where_pos(condition)`** (en `SpatialGroup`): permite seleccionar según coordenadas espaciales (posición en un cubo o cuadrícula).
- **`where_rank(r)`**: filtra elementos asociados a un determinado `rank` en ejecución distribuida.

**Uso práctico**
- Los filtros son **componibles**: se pueden aplicar sucesivamente para refinar la selección.
- Es posible **clonar grupos** para definir filtros independientes, preservando el original como referencia global.
- Esto resulta útil, por ejemplo, para seleccionar subconjuntos de neuronas excitadoras o inhibidoras, o para definir targets específicos de una conexión sináptica.

**Ventaja conceptual**  
Este mecanismo permite tratar poblaciones grandes como tensores globales en GPU, pero con la flexibilidad de trabajar sobre subconjuntos definidos dinámicamente, sin necesidad de crear copias físicas de los datos.

## 5. Modelo neuronal base y variantes

El comportamiento neuronal se define a través de la clase **`NeuronGroup`**, que actúa como plantilla común para los distintos modelos.

**Estructura interna**
- Cada neurona mantiene un **búfer circular de spikes** con dimensiones `[N, delay_max]`, donde `N` es el número de neuronas del grupo.    
- El índice activo dentro del búfer corresponde al tiempo actual `t % delay_max`.    
- Las corrientes externas o internas pueden inyectarse directamente en el paso en curso mediante `add_current_in_step`.    

**Lectura de spikes**
- **`get_spikes()`** devuelve el vector de spikes en el instante actual.    
- **`get_spikes_at(delay)`** permite acceder al estado con un retardo específico, útil para modelar conexiones sinápticas con retrasos en la transmisión.    

**Modelos implementados**
- **`ParrotNeurons`**: no poseen dinámica propia; simplemente reemiten spikes o corrientes recibidas, útiles como repetidores o fuentes intermedias.    
- **`SimpleIFNeurons`**: modelo _Integrate-and-Fire_ sencillo con umbral, reset y decaimiento exponencial de la variable de estado.    
- **`RandomSpikeNeurons`**: generan spikes de manera probabilística según una tasa en Hz, equivalentes a fuentes de Poisson independientes.    
- **`IFNeurons`** (extendidos): incorporan múltiples canales de conductancia con decaimos bi-exponenciales y potenciales de reversión, lo que permite simular entradas excitatorias e inhibitorias más realistas.    

**Puntos clave**
- El API unificado garantiza que cualquier modelo neuronal pueda conectarse de la misma forma a sinapsis y monitores.    
- El uso de búfer circular sincronizado con `delay_max` asegura que las conexiones puedan acceder a spikes pasados sin necesidad de almacenar todo el historial.

## 6. Conexiones sinápticas: dispersas y densas

Las conexiones entre grupos de neuronas se construyen mediante el operador **`>>`**, que devuelve un **`ConnectionOperator`**. Este operador combina:
1. Un **patrón de conectividad** (`all-to-all`, `one-to-one`, `specific`).    
2. Una **clase de sinapsis** que define la dinámica de transmisión y plasticidad.    

**Conexiones dispersas (`ConnectionGroup`)**
- Se representan como listas planas de índices de pre- y post-sinapsis (`idx_pre`, `idx_pos`).    
- Cada conexión almacena sus parámetros: peso (`weight`), retardo (`delay`), canal sináptico (`channel`).    
- La propagación se implementa con operaciones vectorizadas (`index_add_`), aplicando a cada post-sinapsis la suma de corrientes procedentes de los presinápticos activos.    
- El acceso a spikes presinápticos incluye el retardo específico de cada conexión, utilizando el búfer circular del grupo de origen.    

**Conexiones densas (`ConnectionDense`)**
- Se representan como matrices de peso de tamaño `[N_pre, N_pos]`.    
- La propagación se realiza mediante multiplicación matricial (`matmul`), eficiente cuando la conectividad es densa.    
- También existen variantes con plasticidad (`STDPDenseConnection`).    

**Parámetros y su cálculo**  
La inicialización de parámetros se gestiona mediante `_compute_parameter`, que admite:
- Escalares: aplicados uniformemente.    
- Tensores: asignación explícita de valores distintos.    
- Listas o funciones: generación procedural de parámetros.    

**Restricciones estructurales**
- El retardo asignado a cada conexión debe ser **menor que `delay_max`** del grupo presináptico.    
- El canal especificado debe existir en el grupo postsináptico.

En conjunto, esta dualidad disperso/denso permite modelar tanto redes biológicamente plausibles con conectividad irregular como configuraciones experimentales con alta densidad de conexiones, equilibrando flexibilidad y eficiencia computacional.

## 7. Comunicación inter-GPU: `BridgeNeuronGroup`

Cuando la simulación se distribuye en varias GPUs, las poblaciones neuronales deben intercambiar spikes entre dispositivos. Esto se gestiona con **`BridgeNeuronGroup`**, un nodo especializado que actúa como pasarela.

**Función principal**
- Recoge spikes generados en una GPU y los transmite a las demás.
- Inserta los spikes recibidos en los búferes de las poblaciones destino, adelantándolos en el tiempo para compensar la latencia de comunicación.    

**Mecanismo de operación**
1. Durante `n_bridge_steps`, los spikes locales se acumulan en un **búfer temporal (`_write_buffer`)**.    
2. Al completarse el bloque, los spikes se empaquetan (de bool a uint8) para reducir tamaño.    
3. Se lanza una operación colectiva **`dist.all_gather(async_op=True)`** que intercambia la información entre todos los procesos (una GPU por rank).    
4. Los resultados se desempaquetan y se escriben en el **búfer de spikes futuro (`_spike_buffer`)**, con un desplazamiento de tiempo al menos igual a `n_bridge_steps + 1`.    

**Condiciones de validez**
- Se requiere que **`n_bridge_steps < delay_max`** para que los spikes transmitidos aún entren en el anillo temporal válido.    
- La latencia efectiva de la comunicación es al menos `n_bridge_steps` pasos, lo que se debe considerar en aplicaciones de control en tiempo real.    

**Selección de subpoblaciones**
- Con **`where_rank(r)`** se definen subconjuntos de neuronas que serán visibles solo desde un rank específico, lo que facilita construir arquitecturas distribuidas con control fino de direcciones.    

**Modo no distribuido**
- Si la simulación corre en una sola GPU, el puente no usa comunicación colectiva. Los spikes se copian directamente al búfer de futuro, preservando la misma semántica de latencia.

En suma, `BridgeNeuronGroup` es el componente que convierte una simulación local en una **simulación distribuida multi-GPU**, manteniendo sincronía y eficiencia en el intercambio de spikes.

## 8. Monitores y extracción de datos

La biblioteca incluye **monitores** para registrar la actividad neuronal y variables internas sin interrumpir la simulación.

**Tipos principales**

- **`SpikeMonitor`**    
    - Lee los spikes almacenados en los búferes circulares.        
    - Solo extrae datos cuando `t % delay_max == delay_max - 1`, es decir, al cerrarse un ciclo del anillo temporal.        
    - Devuelve pares `(neuron_id, time)` que permiten reconstruir el tren de disparos de cada neurona.        
- **`VariableMonitor`**    
    - Permite registrar el valor de tensores arbitrarios en cada paso.        
    - Filtra por subconjuntos de neuronas definidos mediante `where`.
    - Los datos se acumulan en arrays de CPU, listos para análisis posterior.        
- **`RingBufferSpikeMonitor`**    
    - Diseñado para recolección continua en GPU.        
    - Mantiene un **búfer circular** en GPU que almacena spikes recientes.
    - Periódicamente copia bloques de datos a memoria de CPU **pinned** usando transferencias asíncronas (`non_blocking=True`).        
    - Adecuado para escenarios en los que se requieren lecturas frecuentes o casi en tiempo real sin penalizar el rendimiento global.

**Características comunes**
- Los monitores no participan en la captura de CUDA Graphs, evitando overhead dentro del ciclo crítico de simulación.    
- Se ejecutan de forma controlada en puntos seguros del ciclo (`pos_step`, `on_finish`).    
- Ofrecen un equilibrio entre **rendimiento** y **accesibilidad de datos**, priorizando mantener los cálculos intensivos en GPU y diferir la transferencia a CPU.    

En conjunto, los monitores son la interfaz estándar para observar y analizar la dinámica de la red, proporcionando visibilidad sin comprometer la eficiencia del motor de simulación.

## 9. Ciclo de vida de un experimento (`Experiment`)

La clase **`Experiment`** organiza toda la lógica de una simulación, ofreciendo un marco estándar para construir, ejecutar y controlar experimentos completos.

**Métodos principales**
- **`build_network()`**: debe crear explícitamente el grafo de nodos (grupos neuronales, sinapsis, monitores). Si se usan varias GPUs, aquí también se añade el puente por defecto (`add_default_bridge`).    
- **`on_start()`**: se ejecuta una vez antes de iniciar la simulación, útil para inicializar variables o preparar registros.    
- **`pre_step()`** y **`pos_step()`**: ganchos que se llaman en cada paso de simulación, antes y después de la ejecución del grafo capturado. Permiten insertar intervenciones controladas (ej. inyección de estímulos, logging periódico).    
- **`on_finish()`**: se ejecuta tras completar la simulación, ideal para guardar resultados o liberar recursos.    

**Ejecución**
- El método **`run(steps)`** recorre el número de pasos solicitado:    
    - Inicializa el simulador y el subárbol de captura si aún no se ha hecho.        
    - Ejecuta `on_start()`.        
    - Itera sobre los pasos, llamando a `pre_step()`, al grafo (`graph.replay()` o equivalente) y a `pos_step()`.        
    - Al finalizar, ejecuta `on_finish()`.        
- El control de errores asegura que, aun en caso de interrupción, se limpien los procesos distribuidos.    

**Contextos de autoparentado**
- **`autoparent("graph")`**: todo lo declarado dentro se incluirá en el subárbol capturable por CUDA Graphs.    
- **`autoparent("normal")`**: nodos declarados fuera del grafo crítico, como monitores o utilidades.    

**Ventaja estructural**  
El uso de `Experiment` permite separar claramente la **definición del modelo** de la **ejecución controlada**, estandarizando el flujo de trabajo y facilitando reproducibilidad.

## 10. Utilidades y logging

Además del motor principal, la biblioteca incluye un conjunto de utilidades que facilitan la instrumentación de los experimentos.

**Logging**
- Implementa salida diferenciada por `rank` en ejecuciones distribuidas.    
- Cada proceso puede imprimir mensajes identificados por color, lo que permite distinguir fácilmente la procedencia de la información.    
- El logging se usa tanto en el núcleo (`Simulator`, `Experiment`) como en ejemplos, para dar trazabilidad sin saturar la salida estándar.    

**Visualización y análisis**
- Incluye funciones de **plotting** simples para raster plots o series temporales.    
- La lógica está diseñada para conmutar automáticamente entre **mostrar en pantalla** o **guardar en archivo**, según el entorno de ejecución (interactivo o batch).    
- Existe una utilidad para suavizar trenes de spikes (`smooth_spikes`), que aplica ventanas de convolución a los datos binarios, útil para estimar tasas de disparo.    

**Detección de entorno**
- Funciones auxiliares como **`is_distributed`** permiten saber si la simulación se está ejecutando con múltiples procesos (`torch.distributed`).    
- **`can_use_torch_compile`** detecta si está disponible la compilación con `torch.compile`, activándola solo cuando sea seguro.    

**Función pedagógica**  
Estas utilidades no forman parte del núcleo de la simulación, pero resultan necesarias para convertir la simulación en un **experimento reproducible y analizable**, reduciendo el esfuerzo del usuario en tareas rutinarias.

## 11. Ejemplos incluidos (mapa rápido)

La biblioteca incorpora scripts de ejemplo que muestran cómo construir y ejecutar distintos tipos de simulación. Estos ejemplos cumplen un rol pedagógico: sirven como plantillas mínimas para que el usuario entienda la API y como pruebas de funcionamiento básico.

**Ejemplos principales**

- **`lib_example_01_multipleGPUs.py`**    
    - Demuestra la comunicación de spikes entre GPUs mediante `BridgeNeuronGroup`.        
    - Implementa un patrón simple de “ping-pong” donde la actividad viaja de una GPU a otra.        
- **`lib_example_02_oneGPU.py`**    
    - Ejemplo de simulación con **una sola GPU**.        
    - Incluye sinapsis con plasticidad STDP dispersa y uso de monitores para registrar actividad.        
- **`lib_example_03_twoGPUs.py`**    
    - Construye una red distribuida entre dos GPUs.        
    - Una fuente en GPU0 envía spikes a través del puente hacia un objetivo en GPU1 con STDP.        
    - Muestra cómo combinar multi-GPU y plasticidad.        
- **`lib_example_04_BRN_STDP.py`** y **`lib_example_04_BRN_STDP_multipleGPUs.py`**    
    - Implementan una red excitatoria/inhibitoria de mayor tamaño con plasticidad STDP densa.        
    - Una versión corre en una sola GPU, la otra distribuye la red entre varias.        

**Utilidad práctica**  
Estos ejemplos ilustran patrones de uso reales:
- Cómo inicializar un experimento y organizar el grafo en `build_network`.    
- Cómo añadir monitores y recolectar datos.    
- Cómo configurar conexiones densas o dispersas.
- Cómo ejecutar tanto en modo local como distribuido con `torchrun`.

En conjunto, constituyen un **manual mínimo de referencia** que complementa la documentación y clarifica la semántica de los componentes de la biblioteca.

## 12. Rendimiento y tiempo real

La biblioteca está diseñada con énfasis en **eficiencia** y **baja latencia**, condiciones necesarias para simular redes de spikes en escenarios cercanos al tiempo real.

**Optimización mediante CUDA Graphs**
- El ciclo de simulación se captura en un **CUDA Graph** tras un periodo de calentamiento.    
- De este modo, cada paso se ejecuta con una sola llamada (`graph.replay()`), eliminando el overhead de lanzar múltiples kernels desde CPU.    

**Minimización de transferencias CPU-GPU**
- Los datos críticos (spikes, variables de estado, pesos sinápticos) se mantienen en GPU.    
- Los monitores transfieren información a CPU en bloques controlados, reduciendo el impacto en el rendimiento.    

**Compresión de spikes para comunicación**
- Los spikes se empaquetan de bool a uint8 antes de la transmisión inter-GPU, logrando una compresión 8:1.    
- Esto disminuye el volumen de datos en operaciones colectivas, acelerando la comunicación con `torch.distributed`.    

**Parámetros críticos de latencia**
- **`n_bridge_steps`**: número de pasos agrupados antes de enviar spikes por el puente. A mayor valor, más eficiencia en comunicación, pero también mayor retraso efectivo.    
- **`delay_max`**: tamaño del búfer circular. Debe ser mayor que `n_bridge_steps` para garantizar consistencia temporal.
- **Densidad de conexiones**: el uso de representaciones densas implica alto coste de memoria, mientras que las dispersas son más escalables pero con mayor overhead de índices.
- **Buffers de monitores**: su tamaño determina el equilibrio entre frecuencia de extracción y carga de memoria.    

**Recomendaciones prácticas**
- Ubicar el máximo posible de cómputo dentro del subárbol capturable (`graph`) para aprovechar CUDA Graphs.    
- Evitar asignaciones dinámicas o llamadas a CPU dentro de `_process()`.    
- Ajustar `n_bridge_steps` de acuerdo al retardo tolerable en la aplicación (control en robótica vs. simulación offline).    

En conjunto, estas decisiones técnicas permiten que la biblioteca no solo escale a múltiples GPUs en un mismo host, sino que lo haga con un rendimiento adecuado para aplicaciones que exigen tiempos de respuesta estrictos.

## 13. Limitaciones y puntos de validación

Aunque la biblioteca es potente en su nicho, presenta restricciones y aspectos a verificar en cada uso.

**Limitaciones estructurales**
- La ejecución distribuida se limita a **multi-GPU en un único host**. No se admite actualmente comunicación entre nodos distintos.    
- La funcionalidad de **guardar y restaurar estado (`save_state`)** no está implementada. Cada simulación debe iniciarse desde cero.    
- La generación de máscaras de conectividad aleatoria en los ejemplos se hace de forma explícita en CPU, lo que puede implicar **alto consumo de memoria** para redes muy grandes.    

**Plasticidad STDP**
- En la implementación de conexiones dispersas con STDP, la parte de **depresión sináptica** suma el término `A_minus` en lugar de restar. Esto implica que, si `A_minus > 0`, tanto la potenciación como la “depresión” incrementan el peso.
- Se requiere validar si esto responde a una convención interna (p. ej. `A_minus` negativo) o si se trata de un detalle que el usuario debe ajustar.    
- [ ] #task Comprobar que `A_minus` esté bien implementado en Neurobridge

**Temporización de monitores**
- Los monitores de spikes solo extraen datos cuando se completa un ciclo de `delay_max`. Esto implica que el acceso a spikes tiene una granularidad por bloques, no estrictamente en cada paso.    
- Para aplicaciones que requieran lectura continua (ej. control en bucle cerrado), debe considerarse el uso de `RingBufferSpikeMonitor`.    

**Otros puntos de atención**
- La latencia efectiva de los puentes inter-GPU depende de `n_bridge_steps`. Si se requiere reacción en tiempo real, hay que calcular de antemano el desfase introducido y compararlo con `delay_max`.    
- Al usar conexiones densas, la memoria GPU puede convertirse en un limitante antes que el cómputo.    

En síntesis, la biblioteca ofrece un marco sólido, pero exige al usuario **validar explícitamente** las configuraciones de plasticidad, temporización y consumo de memoria para garantizar que la simulación se comporte como se espera.

## 14. Extensibilidad

La biblioteca está pensada para ser **modular**, de modo que los usuarios puedan añadir nuevos modelos o herramientas sin modificar el núcleo.

**Nuevos modelos neuronales**
- Se crean heredando de **`NeuronGroup`**.    
- Es obligatorio implementar la dinámica de membrana y el manejo del búfer de spikes.    
- Debe respetarse el API de inyección de corrientes (`add_current_in_step`) y de lectura de spikes (`get_spikes`, `get_spikes_at`).    
- Ejemplo de uso: definir un modelo adaptativo con corriente de adaptación lenta, manteniendo compatibilidad con conexiones y monitores.    

**Nuevos tipos de sinapsis**
- Se derivan de **`ConnectionGroup`** (dispersas) o **`ConnectionDense`** (densas).    
- Se implementan dos métodos clave:    
    - **`_init_connection`**: inicializa parámetros de cada sinapsis (peso, retardo, canal).        
    - **`_update`**: define la regla de propagación/plasticidad en cada paso.
- Esto permite añadir fácilmente variantes de STDP, reglas hebbianas o modelos dependientes de tripletes.

**Nuevos monitores**
- Se basan en **`Node`**, con lógica de recolección de datos en `_process` o en los hooks (`pos_step`, `on_finish`).    
- Es importante que las operaciones intensivas en CPU se hagan **fuera del subárbol capturable**, para no romper la ejecución eficiente en CUDA Graphs.    

**Flexibilidad general**
- La separación estricta entre subárbol de grafo y nodos normales facilita integrar código adicional sin comprometer la ruta crítica de simulación.
- La arquitectura jerárquica de nodos asegura que cualquier extensión se acople naturalmente al ciclo de vida (`_ready`, `_process`).    

En conjunto, esta extensibilidad convierte a la biblioteca en una plataforma no cerrada, donde es posible explorar nuevos modelos neuronales, reglas sinápticas o mecanismos de observación sin modificar el motor central.

## 15. Puesta en marcha y ejecución distribuida

La biblioteca está preparada para ejecutarse tanto en **una sola GPU** como en **varias GPUs de un mismo host**.

**Ejecución en modo local (una GPU)**
- Basta con lanzar el script de experimento con Python normal.    
- El simulador detecta el `device` y asigna los tensores a esa GPU.    
- No se inicializa comunicación distribuida.    

**Ejecución en modo distribuido (multi-GPU)**
- Se utiliza **`torchrun`** como lanzador, que crea un proceso por GPU.
- Comando típico:    
    ```bash
    torchrun --nproc_per_node=K script.py
    ```
    donde `K` es el número de GPUs disponibles en el nodo.
- Cada proceso recibe automáticamente sus variables de entorno `WORLD_SIZE` y `RANK`.    
- El backend de comunicación está fijado en **NCCL**, optimizado para hardware NVIDIA.    
- Cada proceso selecciona su GPU con `cuda:rank`.    

**Inicialización automática**
- El simulador detecta si existen variables `WORLD_SIZE`/`RANK` y, en ese caso, entra en modo distribuido.    
- Se crea el grupo de procesos distribuido y se gestiona su destrucción segura al finalizar.    
- El usuario no necesita llamadas manuales a `torch.distributed.init_process_group`.    

**Compatibilidad con puentes (`BridgeNeuronGroup`)**
- En modo distribuido, los puentes gestionan la comunicación colectiva de spikes.    
- En modo local, los puentes siguen operando, pero usando copia directa en GPU para mantener la misma semántica temporal.    

**Resumen**  
La puesta en marcha es sencilla:
- **Un solo GPU** → ejecución directa con Python.    
- **Multi-GPU** en un nodo → ejecución con `torchrun`.    

Este diseño hace posible escalar una simulación sin modificar el código, únicamente cambiando el modo de lanzamiento.