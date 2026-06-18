# MiniGPT en Español 🤖🇪🇸

Este repositorio contiene la implementación desde cero de un modelo **GPT (Generative Pre-trained Transformer)** simplificado, diseñado y optimizado para la generación de texto a nivel de caracteres. Es un proyecto con fines educativos que ilustra la arquitectura de auto-atención (self-attention) y bloques de Transformers modernos bajo el framework PyTorch.

El modelo está configurado para entrenarse en GPU (mediante CUDA) o CPU, y permite cargar o guardar checkpoints del estado de pesos de forma automatizada.

---

## 📊 Arquitectura del Modelo

La implementación sigue la estructura clásica del decodificador de Transformers (como el propuesto en *Attention Is All You Need*), compuesto por:

1. **Embeddings duales:** Embedding de tokens (caracteres del vocabulario) combinado con embeddings de posición entrenables.
2. **Bloques Transformer apilados:** Cada bloque contiene:
   - **Atención Multi-Cabeza (Multi-Head Attention):** Cabezas paralelas que aprenden afinidades contextuales con máscara causal triangular para evitar mirar tokens futuros.
   - **Conexiones Residuales & LayerNorm:** Estabilizan el entrenamiento profundo.
   - **Feed-Forward Network (GELU):** Capa de computación lineal de expansión/contracción lineal con dropout.
3. **Muestreo por Temperatura:** El generador de texto admite un parámetro de temperatura para controlar la aleatoriedad y creatividad en la predicción.

### Hiperparámetros de la Red
*   **Longitud de Contexto (Context Length):** 512 caracteres
*   **Tamaño de Lote (Batch Size):** 32 secuencias
*   **Dimensión del Embedding:** 512
*   **Número de Cabezas (Attention Heads):** 8
*   **Número de Capas (Layers):** 8
*   **Optimización:** AdamW con programador de decaimiento por coseno (Cosine Annealing LR)

---

## 📁 Estructura de Archivos

*   `gpt_espanol.py`: Script principal de definición del modelo, bucle de entrenamiento, guardado de checkpoints y generación.
*   `bigrama_espanol.py`: Modelo simplificado basado únicamente en frecuencias de bigramas (usado como baseline).
*   `documentacion_gpt.md`: Documentación técnica detallada del modelo en español.
*   `diagrama_gpt.svg`: Diagrama conceptual de la arquitectura de bloques.
*   `input.txt`: Corpus de texto utilizado para el entrenamiento a nivel de caracteres.
*   `testing.ipynb`: Jupyter Notebook interactivo para pruebas y experimentación rápida.

---

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos Previos

Asegúrate de tener instalado Python 3 y las dependencias correspondientes (se recomienda un entorno virtual):

```bash
pip install torch
```

### 1. Entrenar el Modelo

Para iniciar el entrenamiento o reanudar el último checkpoint guardado, ejecuta:

```bash
python gpt_espanol.py
```

El script buscará automáticamente si existe un archivo `modelo_entrenado.pth`. Si lo encuentra, te preguntará si deseas cargarlo o entrenar una nueva red desde cero.

### 2. Generar Texto

Una vez finalizado el entrenamiento, el modelo guardará su estado de pesos en `modelo_entrenado.pth` y generará un ejemplo de texto de 500 caracteres a través de la salida de consola utilizando el muestreo por temperatura.

---

## 🎓 Créditos

Proyecto desarrollado por **Edson Gonzales**. Inspirado en los principios arquitectónicos de los modelos GPT de OpenAI y enfocado en la enseñanza simplificada del procesamiento de lenguaje natural (NLP).
