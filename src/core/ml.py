"""
Modulo de Machine Learning y Metodos Numericos Puros para DaZe.
Implementado sin dependencias externas (sin numpy, scipy ni scikit-learn).
"""


def _raiz(n, tolerancia=1e-12):
    """Raiz cuadrada por algoritmo iterativo de Newton-Raphson."""
    if n < 0:
        raise ValueError("No se puede calcular raiz de numero negativo")
    if n == 0:
        return 0.0
    x = n / 2.0 if n >= 1 else 1.0
    for _ in range(100):
        sig = 0.5 * (x + n / x)
        if abs(x - sig) < tolerancia:
            return sig
        x = sig
    return x


def correlacion_pearson(x, y):
    """Calcula el coeficiente de correlacion r de Pearson en [-1.0, 1.0]."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)
    media_x = sum(x) / n
    media_y = sum(y) / n

    num = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
    den_x = sum((xi - media_x) ** 2 for xi in x)
    den_y = sum((yi - media_y) ** 2 for yi in y)

    denominador = _raiz(den_x * den_y)
    if denominador == 0:
        return 0.0
    return num / denominador


# Alias de correlacion
correlacion = correlacion_pearson


class RegresionLineal:
    """Modelo de regresion lineal simple (y = mx + b) por Minimos Cuadrados Ordinarios."""

    def __init__(self):
        self.m = 0.0
        self.b = 0.0
        self.r2 = 0.0

    def __getitem__(self, clave):
        if hasattr(self, clave):
            return getattr(self, clave)
        raise KeyError(clave)

    def ajustar(self, x, y):
        n = len(x)
        if n < 2 or n != len(y):
            raise ValueError("Se requieren al menos 2 pares de datos para ajustar regresion.")

        media_x = sum(x) / n
        media_y = sum(y) / n

        num = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
        den = sum((xi - media_x) ** 2 for xi in x)

        self.m = (num / den) if den != 0 else 0.0
        self.b = media_y - (self.m * media_x)

        ss_tot = sum((yi - media_y) ** 2 for yi in y)
        ss_res = sum((yi - (self.m * xi + self.b)) ** 2 for xi, yi in zip(x, y))
        if ss_tot == 0:
            self.r2 = 1.0 if ss_res == 0 else 0.0
        else:
            self.r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))

        return self

    def predecir(self, x_val):
        return self.m * x_val + self.b


def regresion_lineal(x, y):
    """Ajusta una regresion lineal simple y retorna metricas en un diccionario."""
    modelo = RegresionLineal()
    modelo.ajustar(x, y)
    return {
        "m": modelo.m,
        "b": modelo.b,
        "r2": modelo.r2,
        "modelo": modelo
    }


def predecir_regresion(modelo, x_val):
    """Estima el valor de y para un x_val dado a partir de un modelo de regresion."""
    if isinstance(modelo, dict):
        if "m" in modelo and "b" in modelo:
            return modelo["m"] * x_val + modelo["b"]
        elif "modelo" in modelo:
            return modelo["modelo"].predecir(x_val)
    elif hasattr(modelo, "predecir"):
        return modelo.predecir(x_val)
    raise ValueError("Objeto de modelo no valido para inferencia")


class KMeans:
    """Algoritmo de clustering no supervisado en D dimensiones."""

    def __init__(self, k=2, max_iter=20):
        self.k = k
        self.max_iter = max_iter
        self.centroides = []
        self.etiquetas = []

    def __getitem__(self, clave):
        if hasattr(self, clave):
            return getattr(self, clave)
        raise KeyError(clave)

    def _distancia_euclidiana(self, p1, p2):
        return _raiz(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def ajustar(self, datos):
        if not datos or self.k <= 0:
            return self

        k = min(self.k, len(datos))
        self.centroides = [list(datos[i]) for i in range(k)]
        self.etiquetas = [0] * len(datos)
        dim = len(datos[0])

        for _ in range(self.max_iter):
            nuevas_etiquetas = []
            for punto in datos:
                distancias = [self._distancia_euclidiana(punto, c) for c in self.centroides]
                min_idx = distancias.index(min(distancias))
                nuevas_etiquetas.append(min_idx)

            clusters = {i: [] for i in range(k)}
            for idx, c_idx in enumerate(nuevas_etiquetas):
                clusters[c_idx].append(datos[idx])

            nuevos_centroides = []
            for i in range(k):
                if clusters[i]:
                    n_puntos = len(clusters[i])
                    nuevo_c = [sum(p[d] for p in clusters[i]) / n_puntos for d in range(dim)]
                else:
                    nuevo_c = self.centroides[i]
                nuevos_centroides.append(nuevo_c)

            self.centroides = nuevos_centroides
            if nuevas_etiquetas == self.etiquetas:
                break
            self.etiquetas = nuevas_etiquetas

        return self


def agrupar_kmedias(datos, k=2, max_iter=20):
    """Ejecuta agrupamiento K-Means y retorna centroides y etiquetas."""
    km = KMeans(k=k, max_iter=max_iter)
    km.ajustar(datos)
    return {
        "centroides": km.centroides,
        "etiquetas": km.etiquetas,
        "modelo": km
    }


# Alias sin guiones bajos para uso comodo en el DSL
regresionlineal = regresion_lineal
predecir = predecir_regresion
agruparkmedias = agrupar_kmedias
kmedias = agrupar_kmedias


def _mlp_exp(x):
    if x < -50:
        return 0.0
    if x > 50:
        x = 50.0
    if x < 0:
        return 1.0 / _mlp_exp(-x)
    k = 0
    val = float(x)
    while val > 1.0:
        val /= 2.0
        k += 1
    suma = 1.0
    termino = 1.0
    for i in range(1, 30):
        termino = (termino * val) / i
        suma += termino
    for _ in range(k):
        suma *= suma
    return suma


def _mlp_sig(x):
    if x >= 0:
        return 1.0 / (1.0 + _mlp_exp(-x))
    ex = _mlp_exp(x)
    return ex / (1.0 + ex)


class PerceptronMulticapa:
    def __init__(self, ocultas=4, lr=0.8, epocas=3000):
        self.ocultas = ocultas
        self.lr = float(lr)
        self.epocas = int(epocas)
        self.w1 = [
            [0.45, -0.35],
            [-0.55, 0.65],
            [0.30, 0.50],
            [-0.40, -0.60],
        ]
        self.b1 = [0.1, -0.1, 0.2, -0.2]
        self.w2 = [0.5, -0.6, 0.4, -0.5]
        self.b2 = 0.05
        self.historial_error = []

    def _forward(self, x):
        h = []
        for j in range(self.ocultas):
            z = self.b1[j] + self.w1[j][0] * x[0] + self.w1[j][1] * x[1]
            h.append(_mlp_sig(z))
        z_out = self.b2 + sum(self.w2[j] * h[j] for j in range(self.ocultas))
        return h, _mlp_sig(z_out)

    def entrenar(self, X, Y):
        self.historial_error = []
        paso = max(1, self.epocas // 12)
        for ep in range(1, self.epocas + 1):
            err_tot = 0.0
            for x, y in zip(X, Y):
                h, y_hat = self._forward(x)
                err = y - y_hat
                err_tot += 0.5 * (err ** 2)
                d_out = err * y_hat * (1.0 - y_hat)
                d_h = [d_out * self.w2[j] * h[j] * (1.0 - h[j]) for j in range(self.ocultas)]
                for j in range(self.ocultas):
                    self.w2[j] += self.lr * d_out * h[j]
                self.b2 += self.lr * d_out
                for j in range(self.ocultas):
                    self.w1[j][0] += self.lr * d_h[j] * x[0]
                    self.w1[j][1] += self.lr * d_h[j] * x[1]
                    self.b1[j] += self.lr * d_h[j]
            mse = err_tot / len(X)
            if ep == 1 or ep % paso == 0 or ep == self.epocas:
                self.historial_error.append({"epoca": ep, "error": round(mse, 6)})
        return self

    def estimar(self, x):
        _, y_hat = self._forward(x)
        return round(y_hat, 6)

    def clasificar(self, x):
        return 1 if self.estimar(x) >= 0.5 else 0


def entrenar_mlp(X, Y, epocas=3000, lr=0.8):
    mlp = PerceptronMulticapa(ocultas=4, lr=lr, epocas=epocas).entrenar(X, Y)
    return {
        "modelo": mlp,
        "historial_error": mlp.historial_error,
        "error_final": mlp.historial_error[-1]["error"],
        "arquitectura": "2-4-1 (Sigmoide)"
    }


def estimar_mlp(modelo, x):
    mlp = modelo["modelo"] if isinstance(modelo, dict) else modelo
    if x and isinstance(x[0], list):
        return [mlp.estimar(p) for p in x]
    return mlp.estimar(x)


def clasificar_mlp(modelo, x):
    mlp = modelo["modelo"] if isinstance(modelo, dict) else modelo
    if x and isinstance(x[0], list):
        return [mlp.clasificar(p) for p in x]
    return mlp.clasificar(x)
