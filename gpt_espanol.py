import torch
import torch.nn as nn
from torch.nn import functional as F

# hiperparámetros - configuración del modelo
tamaño_lote = 32  # cuántas secuencias independientes procesaremos en paralelo?
longitud_bloque = 512  # cuál es la longitud máxima de contexto para predicciones?
max_iteraciones = 8000  # número máximo de pasos de entrenamiento
intervalo_evaluacion = 400  # cada cuántos pasos evaluamos el modelo
tasa_aprendizaje = 1e-4  # qué tan rápido aprende el modelo
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'  # usar GPU si está disponible
iteraciones_eval = 200  # cuántas iteraciones para evaluar pérdida
n_embedding = 512  # dimensión de los embeddings
n_cabezas = 8  # número de cabezas de atención
n_capas = 8  # número de capas transformer
abandono = 0.1  # tasa de dropout para regularización
# ------------

torch.manual_seed(1337)  # semilla para reproducibilidad

# cargar el dataset de texto
with open('input.txt', 'r', encoding='utf-8') as f:
    texto = f.read()

# obtener todos los caracteres únicos del texto
caracteres = sorted(list(set(texto)))
tamaño_vocabulario = len(caracteres)

# crear mapeo de caracteres a números y viceversa
caracter_a_numero = {ch: i for i, ch in enumerate(caracteres)}
numero_a_caracter = {i: ch for i, ch in enumerate(caracteres)}

# funciones de codificación y decodificación
codificar = lambda s: [caracter_a_numero[c] for c in s]  # texto → números
decodificar = lambda l: ''.join([numero_a_caracter[i] for i in l])  # números → texto

# dividir datos en entrenamiento y validación
datos = torch.tensor(codificar(texto), dtype=torch.long)
n = int(0.9 * len(datos))  # 90% para entrenamiento, 10% para validación
datos_entrenamiento = datos[:n]
datos_validacion = datos[n:]


def obtener_lote(division):
    """Genera un lote pequeño de datos de entrada x y objetivos y"""
    datos = datos_entrenamiento if division == 'train' else datos_validacion
    ix = torch.randint(len(datos) - longitud_bloque, (tamaño_lote,))
    x = torch.stack([datos[i:i + longitud_bloque] for i in ix])
    y = torch.stack([datos[i + 1:i + longitud_bloque + 1] for i in ix])
    x, y = x.to(dispositivo), y.to(dispositivo)
    return x, y


@torch.no_grad()
def estimar_perdida():
    """Evalúa la pérdida del modelo en datos de entrenamiento y validación"""
    resultado = {}
    modelo.eval()  # modo evaluación
    for division in ['train', 'val']:
        perdidas = torch.zeros(iteraciones_eval)
        for k in range(iteraciones_eval):
            X, Y = obtener_lote(division)
            logits, perdida = modelo(X, Y)
            perdidas[k] = perdida.item()
        resultado[division] = perdidas.mean()
    modelo.train()  # volver a modo entrenamiento
    return resultado


class CabezaAtencion(nn.Module):
    """Una cabeza de auto-atención"""

    def __init__(self, tamaño_cabeza):
        super().__init__()
        # matrices para calcular consultas, claves y valores
        self.clave = nn.Linear(n_embedding, tamaño_cabeza, bias=False)
        self.consulta = nn.Linear(n_embedding, tamaño_cabeza, bias=False)
        self.valor = nn.Linear(n_embedding, tamaño_cabeza, bias=False)
        
        # máscara triangular para atención causal (no ver el futuro)
        self.register_buffer('mascara_triangular', 
                           torch.tril(torch.ones(longitud_bloque, longitud_bloque)))
        self.abandono = nn.Dropout(abandono)

    def forward(self, x):
        """
        Procesa la entrada a través de la cabeza de atención
        Entrada: (lote, tiempo, canales)
        Salida: (lote, tiempo, tamaño_cabeza)
        """
        B, T, C = x.shape
        k = self.clave(x)      # (B,T,tamaño_cabeza)
        q = self.consulta(x)   # (B,T,tamaño_cabeza)
        
        # calcular puntuaciones de atención ("afinidades")
        pesos = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # escalado
        pesos = pesos.masked_fill(self.mascara_triangular[:T, :T] == 0, float('-inf'))
        pesos = F.softmax(pesos, dim=-1)  # convertir a probabilidades
        pesos = self.abandono(pesos)
        
        # realizar agregación ponderada de los valores
        v = self.valor(x)  # (B,T,tamaño_cabeza)
        salida = pesos @ v  # (B,T,tamaño_cabeza)
        return salida


class AtencionMultiCabeza(nn.Module):
    """Múltiples cabezas de auto-atención en paralelo"""

    def __init__(self, num_cabezas, tamaño_cabeza):
        super().__init__()
        self.cabezas = nn.ModuleList([CabezaAtencion(tamaño_cabeza) 
                                    for _ in range(num_cabezas)])
        self.proyeccion = nn.Linear(tamaño_cabeza * num_cabezas, n_embedding)
        self.abandono = nn.Dropout(abandono)

    def forward(self, x):
        """Combina las salidas de todas las cabezas de atención"""
        salida = torch.cat([cabeza(x) for cabeza in self.cabezas], dim=-1)
        salida = self.abandono(self.proyeccion(salida))
        return salida


class RedAdelante(nn.Module):
    """Red neuronal simple: lineal → activación → lineal"""

    def __init__(self, n_embedding):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),  # expansión 4x
            nn.GELU(),  # función de activación
            nn.Linear(4 * n_embedding, n_embedding),  # contracción
            nn.Dropout(abandono),  # regularización
        )

    def forward(self, x):
        return self.red(x)


class BloqueTransformer(nn.Module):
    """Bloque Transformer: comunicación (atención) seguida de computación (red adelante)"""

    def __init__(self, n_embedding, n_cabezas):
        super().__init__()
        tamaño_cabeza = n_embedding // n_cabezas
        self.auto_atencion = AtencionMultiCabeza(n_cabezas, tamaño_cabeza)
        self.red_adelante = RedAdelante(n_embedding)
        self.norm_capa1 = nn.LayerNorm(n_embedding)  # normalización de capa
        self.norm_capa2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        """
        Procesa entrada a través del bloque transformer
        Usa conexiones residuales: x = x + f(x)
        """
        # comunicación entre tokens (auto-atención)
        x = x + self.auto_atencion(self.norm_capa1(x))
        # procesamiento individual de tokens (red adelante)
        x = x + self.red_adelante(self.norm_capa2(x))
        return x


class ModeloLenguajeGPT(nn.Module):
    """Modelo completo de lenguaje tipo GPT"""

    def __init__(self):
        super().__init__()
        # tabla de embeddings para tokens (caracteres)
        self.tabla_embedding_tokens = nn.Embedding(tamaño_vocabulario, n_embedding)
        # tabla de embeddings para posiciones
        self.tabla_embedding_posiciones = nn.Embedding(longitud_bloque, n_embedding)
        # stack de bloques transformer
        self.bloques = nn.Sequential(*[BloqueTransformer(n_embedding, n_cabezas=n_cabezas) 
                                     for _ in range(n_capas)])
        self.norm_final = nn.LayerNorm(n_embedding)  # normalización final
        self.cabeza_lenguaje = nn.Linear(n_embedding, tamaño_vocabulario)  # salida al vocabulario

        # inicialización mejorada de pesos (importante para convergencia)
        self.apply(self._inicializar_pesos)

    def _inicializar_pesos(self, modulo):
        """Inicializa los pesos del modelo de manera óptima"""
        if isinstance(modulo, nn.Linear):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            if modulo.bias is not None:
                torch.nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, nn.Embedding):
            torch.nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

    def forward(self, idx, objetivos=None):
        """
        Pase hacia adelante del modelo
        idx: índices de tokens de entrada (B,T)
        objetivos: índices objetivo para calcular pérdida (B,T)
        """
        B, T = idx.shape

        # obtener embeddings de tokens y posiciones
        emb_tokens = self.tabla_embedding_tokens(idx)  # (B,T,C)
        emb_posiciones = self.tabla_embedding_posiciones(torch.arange(T, device=dispositivo))  # (T,C)
        x = emb_tokens + emb_posiciones  # combinar embeddings (B,T,C)
        
        # pasar por todos los bloques transformer
        x = self.bloques(x)  # (B,T,C)
        x = self.norm_final(x)  # normalización final (B,T,C)
        logits = self.cabeza_lenguaje(x)  # proyectar al vocabulario (B,T,vocab_size)

        if objetivos is None:
            perdida = None
        else:
            # calcular pérdida de entropía cruzada
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            objetivos = objetivos.view(B * T)
            perdida = F.cross_entropy(logits, objetivos)

        return logits, perdida

    def generar(self, idx, max_tokens_nuevos, temperatura= 1.2):
        """
        Genera nuevos tokens usando el modelo entrenado
        idx: contexto inicial (B, T)
        max_tokens_nuevos: cuántos tokens generar
        """
        for _ in range(max_tokens_nuevos):
            # recortar contexto al tamaño máximo del bloque
            idx_condicionado = idx[:, -longitud_bloque:]
            # obtener predicciones
            logits, perdida = self(idx_condicionado)
            # enfocarse solo en el último paso de tiempo
            logits = logits[:, -1, :]/temperatura  # (B, C)
            # aplicar softmax para obtener probabilidades
            probabilidades = F.softmax(logits, dim=-1)  # (B, C)
            # muestrear de la distribución
            idx_siguiente = torch.multinomial(probabilidades, num_samples=1)  # (B, 1)
            # agregar índice muestreado a la secuencia
            idx = torch.cat((idx, idx_siguiente), dim=1)  # (B, T+1)
        return idx


# crear e inicializar el modelo
modelo = ModeloLenguajeGPT()
m = modelo.to(dispositivo)

# mostrar número de parámetros
print(f"El modelo tiene {sum(p.numel() for p in m.parameters())/1e6:.1f}M parámetros")

# crear optimizador
optimizador = torch.optim.AdamW(modelo.parameters(), lr=tasa_aprendizaje)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizador, T_max=max_iteraciones
)
scheduler.step()

# verificar si existe un modelo entrenado
import os
modelo_guardado = 'modelo_entrenado.pth'
entrenar_nuevo = True

if os.path.exists(modelo_guardado):
    print(f"\nSe encontró un modelo entrenado: {modelo_guardado}")
    respuesta = input("¿Quieres cargar el modelo existente? (s/n): ").lower().strip()
    
    if respuesta == 's' or respuesta == 'si':
        print("Cargando modelo entrenado...")
        checkpoint = torch.load(modelo_guardado, map_location=dispositivo)
        
        # cargar estado del modelo
        modelo.load_state_dict(checkpoint['modelo_estado'])
        optimizador.load_state_dict(checkpoint['optimizador_estado'])
        
        # cargar vocabulario y hiperparámetros
        caracteres_guardados = checkpoint['caracteres']
        tamaño_vocabulario_guardado = checkpoint['tamaño_vocabulario']
        
        print(f"Modelo cargado exitosamente!")
        print(f"Vocabulario: {tamaño_vocabulario_guardado} caracteres")
        entrenar_nuevo = False
    else:
        print("Entrenando nuevo modelo...")

if entrenar_nuevo:
    # bucle de entrenamiento
    for iteracion in range(max_iteraciones):
        # evaluar pérdida periódicamente
        if iteracion % intervalo_evaluacion == 0 or iteracion == max_iteraciones - 1:
            perdidas = estimar_perdida()
            print(f"paso {iteracion}: pérdida entrenamiento {perdidas['train']:.4f}, "
                  f"pérdida validación {perdidas['val']:.4f}")

        # obtener lote de datos
        xb, yb = obtener_lote('train')

        # evaluar pérdida y hacer backpropagation
        logits, perdida = modelo(xb, yb)
        optimizador.zero_grad(set_to_none=True)
        perdida.backward()
        optimizador.step()

    # guardar el modelo entrenado
    print(f"\nGuardando modelo entrenado en {modelo_guardado}...")
    torch.save({
        'modelo_estado': modelo.state_dict(),
        'optimizador_estado': optimizador.state_dict(),
        'caracteres': caracteres,
        'tamaño_vocabulario': tamaño_vocabulario,
        'longitud_bloque': longitud_bloque,
        'n_embedding': n_embedding,
        'n_cabezas': n_cabezas,
        'n_capas': n_capas,
        'abandono': abandono
    }, modelo_guardado)
    print("Modelo guardado exitosamente!")

# generar texto desde el modelo entrenado
contexto = torch.zeros((1, 1), dtype=torch.long, device=dispositivo)
texto_generado = decodificar(m.generar(contexto, max_tokens_nuevos=500)[0].tolist())
print("\n--- TEXTO GENERADO ---")
print(texto_generado)

# descomentar para guardar texto largo en archivo
# open('mas_texto.txt', 'w').write(decodificar(m.generar(contexto, max_tokens_nuevos=10000)[0].tolist()))