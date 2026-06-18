import torch
import torch.nn as nn
from torch.nn import functional as F

# hiperparámetros - configuración del modelo bigrama
tamaño_lote = 32  # cuántas secuencias independientes procesaremos en paralelo?
tamaño_bloque = 8  # cuál es la longitud máxima de contexto para predicciones?
max_iteraciones = 3000  # número máximo de pasos de entrenamiento
intervalo_evaluacion = 300  # cada cuántos pasos evaluamos el modelo
tasa_aprendizaje = 1e-2  # qué tan rápido aprende el modelo (más alto que GPT)
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'  # usar GPU si está disponible
iteraciones_eval = 200  # cuántas iteraciones para evaluar pérdida
# ------------

torch.manual_seed(1337)  # semilla para reproducibilidad

# descargar dataset: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    texto = f.read()

# obtener todos los caracteres únicos que aparecen en el texto
caracteres = sorted(list(set(texto)))
tamaño_vocabulario = len(caracteres)

# crear mapeo de caracteres a enteros y viceversa
caracter_a_numero = {ch: i for i, ch in enumerate(caracteres)}
numero_a_caracter = {i: ch for i, ch in enumerate(caracteres)}

# funciones de codificación y decodificación
codificar = lambda s: [caracter_a_numero[c] for c in s]  # codificador: texto → lista de enteros
decodificar = lambda l: ''.join([numero_a_caracter[i] for i in l])  # decodificador: enteros → texto

# dividir datos en entrenamiento y validación
datos = torch.tensor(codificar(texto), dtype=torch.long)
n = int(0.9 * len(datos))  # primeros 90% para entrenamiento, resto para validación
datos_entrenamiento = datos[:n]
datos_validacion = datos[n:]


def obtener_lote(division):
    """
    Genera un lote pequeño de datos de entrada x y objetivos y
    
    Args:
        division (str): 'train' para entrenamiento, 'val' para validación
    
    Returns:
        tuple: (x, y) donde x son las entradas y y son los objetivos
    """
    datos = datos_entrenamiento if division == 'train' else datos_validacion
    ix = torch.randint(len(datos) - tamaño_bloque, (tamaño_lote,))
    x = torch.stack([datos[i:i + tamaño_bloque] for i in ix])
    y = torch.stack([datos[i + 1:i + tamaño_bloque + 1] for i in ix])
    x, y = x.to(dispositivo), y.to(dispositivo)
    return x, y


@torch.no_grad()
def estimar_perdida():
    """
    Evalúa la pérdida del modelo en datos de entrenamiento y validación
    sin calcular gradientes (más eficiente)
    
    Returns:
        dict: Diccionario con pérdidas promedio de entrenamiento y validación
    """
    resultado = {}
    modelo.eval()  # poner modelo en modo evaluación
    for division in ['train', 'val']:
        perdidas = torch.zeros(iteraciones_eval)
        for k in range(iteraciones_eval):
            X, Y = obtener_lote(division)
            logits, perdida = modelo(X, Y)
            perdidas[k] = perdida.item()
        resultado[division] = perdidas.mean()
    modelo.train()  # volver a modo entrenamiento
    return resultado


class ModeloLenguajeBigrama(nn.Module):
    """
    Modelo de lenguaje bigrama súper simple
    
    Este modelo predice el siguiente carácter basándose únicamente en el carácter actual.
    No tiene memoria de contexto más largo, por lo que es muy limitado pero fácil de entender.
    """

    def __init__(self, tamaño_vocabulario):
        """
        Inicializa el modelo bigrama
        
        Args:
            tamaño_vocabulario (int): Número de caracteres únicos en el vocabulario
        """
        super().__init__()
        # cada token lee directamente los logits para el siguiente token desde una tabla de búsqueda
        self.tabla_embedding_tokens = nn.Embedding(tamaño_vocabulario, tamaño_vocabulario)

    def forward(self, idx, objetivos=None):
        """
        Pase hacia adelante del modelo
        
        Args:
            idx (torch.Tensor): Índices de tokens de entrada (B,T)
            objetivos (torch.Tensor, opcional): Índices objetivo para calcular pérdida (B,T)
        
        Returns:
            tuple: (logits, perdida) donde logits son las predicciones y perdida es la pérdida calculada
        """
        # idx y objetivos son tensores (B,T) de enteros
        logits = self.tabla_embedding_tokens(idx)  # (B,T,C) donde C = tamaño_vocabulario

        if objetivos is None:
            perdida = None
        else:
            # reorganizar tensores para calcular pérdida de entropía cruzada
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # aplanar para cross_entropy
            objetivos = objetivos.view(B * T)
            perdida = F.cross_entropy(logits, objetivos)

        return logits, perdida

    def generar(self, idx, max_tokens_nuevos):
        """
        Genera nuevos tokens usando el modelo entrenado
        
        Args:
            idx (torch.Tensor): Contexto inicial (B, T)
            max_tokens_nuevos (int): Cuántos tokens generar
        
        Returns:
            torch.Tensor: Secuencia extendida con tokens generados (B, T + max_tokens_nuevos)
        """
        # idx es un array (B, T) de índices en el contexto actual
        for _ in range(max_tokens_nuevos):
            # obtener las predicciones
            logits, perdida = self(idx)
            # enfocarse solo en el último paso de tiempo
            logits = logits[:, -1, :]  # se convierte en (B, C)
            # aplicar softmax para obtener probabilidades
            probabilidades = F.softmax(logits, dim=-1)  # (B, C)
            # muestrear de la distribución
            idx_siguiente = torch.multinomial(probabilidades, num_samples=1)  # (B, 1)
            # agregar índice muestreado a la secuencia en ejecución
            idx = torch.cat((idx, idx_siguiente), dim=1)  # (B, T+1)
        return idx


# crear e inicializar el modelo
modelo = ModeloLenguajeBigrama(tamaño_vocabulario)
m = modelo.to(dispositivo)

# crear optimizador PyTorch
optimizador = torch.optim.AdamW(modelo.parameters(), lr=tasa_aprendizaje)

print("=== INICIANDO ENTRENAMIENTO DEL MODELO BIGRAMA ===")
print(f"Tamaño del vocabulario: {tamaño_vocabulario} caracteres")
print(f"Dispositivo: {dispositivo}")
print(f"Parámetros del modelo: {sum(p.numel() for p in m.parameters())} parámetros")

# bucle de entrenamiento
for iteracion in range(max_iteraciones):
    # evaluar la pérdida en conjuntos de entrenamiento y validación de vez en cuando
    if iteracion % intervalo_evaluacion == 0:
        perdidas = estimar_perdida()
        print(f"paso {iteracion}: pérdida entrenamiento {perdidas['train']:.4f}, "
              f"pérdida validación {perdidas['val']:.4f}")

    # muestrear un lote de datos
    xb, yb = obtener_lote('train')

    # evaluar la pérdida
    logits, perdida = modelo(xb, yb)
    optimizador.zero_grad(set_to_none=True)  # limpiar gradientes
    perdida.backward()  # calcular gradientes (backpropagation)
    optimizador.step()  # actualizar parámetros

print("\n=== ENTRENAMIENTO COMPLETADO ===")

# generar texto desde el modelo
contexto = torch.zeros((1, 1), dtype=torch.long, device=dispositivo)
texto_generado = decodificar(m.generar(contexto, max_tokens_nuevos=500)[0].tolist())

print("\n--- TEXTO GENERADO POR EL MODELO BIGRAMA ---")
print(texto_generado)
print("\n--- FIN DEL TEXTO GENERADO ---")

print(f"\nNOTA: Este modelo bigrama es muy simple y solo considera el carácter anterior")
print(f"para predecir el siguiente. El texto generado será menos coherente que GPT.")
print(f"Es útil como punto de partida para entender modelos de lenguaje más complejos.")