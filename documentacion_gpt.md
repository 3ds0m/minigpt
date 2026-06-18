# Documentación del Modelo GPT en Español

## Resumen General

Este proyecto implementa un modelo GPT (Generative Pre-trained Transformer) simplificado para generar texto estilo Shakespeare. Es una versión educativa que muestra los conceptos fundamentales de los transformers.

## Arquitectura del Modelo

### Hiperparámetros Principales

```python
tamaño_lote = 64          # Secuencias procesadas en paralelo
tamaño_bloque = 256       # Longitud máxima de contexto
n_embedding = 384         # Dimensión de los vectores embedding
n_cabezas = 6             # Número de cabezas de atención
n_capas = 6               # Número de bloques transformer
abandono = 0.2            # Tasa de dropout para regularización
```

## 🔧 Clases y Métodos Principales

### 1. `CabezaAtencion` - Mecanismo de Auto-Atención

**Propósito**: Implementa una cabeza individual de auto-atención que permite al modelo "prestar atención" a diferentes partes de la secuencia de entrada.

#### Métodos:

- **`__init__(tamaño_cabeza)`**: 
  - Inicializa las matrices de consulta (Q), clave (K) y valor (V)
  - Crea la máscara triangular para atención causal
  - Configura dropout para regularización

- **`forward(x)`**:
  - **Entrada**: Tensor (lote, tiempo, canales)
  - **Proceso**: 
    1. Calcula Q, K, V usando transformaciones lineales
    2. Computa puntuaciones de atención: `QK^T / √d`
    3. Aplica máscara causal (no ver el futuro)
    4. Normaliza con softmax
    5. Pondera los valores con las puntuaciones
  - **Salida**: Representación atendida (lote, tiempo, tamaño_cabeza)

### 2. `AtencionMultiCabeza` - Atención Paralela

**Propósito**: Combina múltiples cabezas de atención para capturar diferentes tipos de relaciones en los datos.

#### Métodos:

- **`__init__(num_cabezas, tamaño_cabeza)`**:
  - Crea lista de cabezas de atención independientes
  - Inicializa proyección final para combinar resultados

- **`forward(x)`**:
  - Ejecuta todas las cabezas en paralelo
  - Concatena los resultados
  - Aplica proyección final y dropout

### 3. `RedAdelante` - Procesamiento Individual

**Propósito**: Procesa cada posición de la secuencia de forma independiente usando una red neuronal simple.

#### Arquitectura:
```
Entrada (384) → Linear (1536) → ReLU → Linear (384) → Dropout
```

### 4. `BloqueTransformer` - Unidad Fundamental

**Propósito**: Combina atención y procesamiento feed-forward con conexiones residuales.

#### Métodos:

- **`forward(x)`**:
  - **Paso 1**: `x = x + auto_atencion(norm_capa1(x))` (Comunicación)
  - **Paso 2**: `x = x + red_adelante(norm_capa2(x))` (Computación)
  - Las conexiones residuales (`x + ...`) ayudan al entrenamiento

### 5. `ModeloLenguajeGPT` - Modelo Completo

**Propósito**: Modelo principal que combina todos los componentes.

#### Componentes:

1. **Embeddings**:
   - `tabla_embedding_tokens`: Convierte caracteres a vectores
   - `tabla_embedding_posiciones`: Codifica posiciones en la secuencia

2. **Procesamiento**:
   - Stack de bloques transformer
   - Normalización final
   - Cabeza de salida al vocabulario

#### Métodos Principales:

- **`_inicializar_pesos(modulo)`**:
  - Inicializa pesos con distribución normal (media=0, std=0.02)
  - Crucial para convergencia estable

- **`forward(idx, objetivos=None)`**:
  - **Entrada**: Índices de tokens (B, T)
  - **Proceso**:
    1. Combina embeddings de tokens y posiciones
    2. Pasa por todos los bloques transformer
    3. Aplica normalización final
    4. Proyecta al vocabulario
  - **Salida**: Logits y pérdida (si hay objetivos)

- **`generar(idx, max_tokens_nuevos)`**:
  - **Propósito**: Genera texto nuevo token por token
  - **Proceso**:
    1. Toma contexto actual (limitado a tamaño_bloque)
    2. Predice probabilidades del siguiente token
    3. Muestrea según las probabilidades
    4. Agrega token a la secuencia
    5. Repite hasta generar max_tokens_nuevos

## 🔄 Flujo de Entrenamiento

### 1. Preparación de Datos
```python
def obtener_lote(division):
    # Selecciona secuencias aleatorias del dataset
    # Crea pares (entrada, objetivo) donde objetivo = entrada + 1 posición
```

### 2. Evaluación de Pérdida
```python
def estimar_perdida():
    # Evalúa el modelo en datos de entrenamiento y validación
    # Usa cross-entropy loss para medir qué tan bien predice
```

### 3. Bucle de Entrenamiento
```python
for iteracion in range(max_iteraciones):
    # 1. Obtener lote de datos
    # 2. Calcular predicciones y pérdida
    # 3. Backpropagation (calcular gradientes)
    # 4. Actualizar pesos con optimizador
    # 5. Evaluar progreso periódicamente
```

## Proceso de Generación de Texto

### Paso a Paso:

1. **Contexto Inicial**: Comienza con token especial o secuencia dada
2. **Predicción**: Modelo calcula probabilidades para siguiente carácter
3. **Muestreo**: Selecciona carácter según distribución de probabilidades
4. **Actualización**: Agrega carácter a la secuencia
5. **Repetición**: Continúa hasta generar texto deseado

### Ejemplo:
```
Contexto: "Ser o no"
Predicción: P(" ") = 0.7, P("s") = 0.2, P("t") = 0.1
Muestreo: Selecciona " " (espacio)
Nuevo contexto: "Ser o no "
... continúa generando
```

## Matemáticas Clave

### Auto-Atención:
```
Atención(Q,K,V) = softmax(QK^T / √d_k)V
```

### Conexiones Residuales:
```
salida = x + SubCapa(LayerNorm(x))
```

### Pérdida de Entropía Cruzada:
```
Loss = -Σ y_true * log(y_pred)
```

## Características Importantes

### Fortalezas:
- **Arquitectura estándar**: Mismos principios que GPT-3/4
- **Código limpio**: Fácil de entender y modificar
- **Entrenamiento rápido**: Converge en minutos/horas
- **Generación coherente**: Produce texto estilo Shakespeare

### Limitaciones:
- **Vocabulario pequeño**: Solo 65 caracteres únicos
- **Contexto limitado**: Máximo 256 caracteres
- **Dataset específico**: Solo entrenado en Shakespeare
- **Escala pequeña**: 10M parámetros vs 175B+ de GPT-3

## Experimentos Sugeridos

1. **Modificar hiperparámetros**:
   - Aumentar `n_capas` para modelo más profundo
   - Cambiar `n_cabezas` para diferentes patrones de atención
   - Ajustar `tasa_aprendizaje` para convergencia

2. **Cambiar dataset**:
   - Usar diferentes textos (poesía, código, etc.)
   - Experimentar con tokenización por palabras

3. **Análisis de atención**:
   - Visualizar qué partes del texto atiende el modelo
   - Estudiar patrones aprendidos por diferentes cabezas

4. **Generación controlada**:
   - Implementar temperature sampling
   - Agregar técnicas como top-k o nucleus sampling

## Métricas de Evaluación

- **Pérdida de entrenamiento**: Qué tan bien memoriza el dataset
- **Pérdida de validación**: Qué tan bien generaliza
- **Perplejidad**: exp(pérdida) - medida de "sorpresa" del modelo
- **Calidad subjetiva**: Coherencia y estilo del texto generado

## Conexión con ChatGPT

Este modelo es esencialmente un "ChatGPT microscópico":
- **Misma arquitectura**: Transformer decoder
- **Mismo objetivo**: Predecir siguiente token
- **Misma técnica**: Auto-atención y feed-forward
- **Diferencia principal**: Escala (datos, parámetros, compute)

La comprensión de este código te da las bases para entender modelos mucho más grandes como GPT-3, GPT-4, y otros LLMs modernos.