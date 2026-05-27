# %% [markdown]
# # Perceptrón: Segmentación de Colores en Imagen de Flor
# 
# Entrenamos el perceptrón con pixeles extraídos automáticamente mediante umbrales de color para segmentar la flor en: pétalos (blanco), centro (amarillo), hojas (verde) y fondo/sombras (oscuro).

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import os

# %%
class Perceptron:
    def __init__(self, n_entradas, tasa_aprendizaje=0.01, epocas_maximas=100):
        self.pesos = np.random.uniform(-0.5, 0.5, n_entradas)
        self.umbral = np.random.uniform(-0.5, 0.5)
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas_maximas = epocas_maximas
        self.historial_errores = []

    def funcion_activacion(self, suma):
        return 1 if suma >= 0 else 0

    def entrenar(self, X, Y):
        for epoca in range(self.epocas_maximas):
            errores_epoca = 0
            for i in range(len(X)):
                entradas = X[i]
                salida_deseada = Y[i]
                
                suma = np.dot(entradas, self.pesos) + self.umbral
                salida_obtenida = self.funcion_activacion(suma)
                
                error = salida_deseada - salida_obtenida
                if error != 0:
                    errores_epoca += 1
                    self.pesos += self.tasa_aprendizaje * error * entradas
                    self.umbral += self.tasa_aprendizaje * error
                    
            self.historial_errores.append(errores_epoca)
            if errores_epoca == 0:
                break

    def activacion_cruda(self, entradas):
        return np.dot(entradas, self.pesos) + self.umbral

def predecir_multiclase(perceptrones, entradas):
    activaciones = [p.activacion_cruda(entradas) for p in perceptrones]
    return np.argmax(activaciones)

# %% [markdown]
# ## 1. Cargar la imagen y extraer muestras de entrenamiento

# %%
ruta_imagen = 'imagen_flor.png'
if not os.path.exists(ruta_imagen):
    print(f"Error: No se encuentra {ruta_imagen}")

img_pil = Image.open(ruta_imagen).convert('RGB')
# Redimensionar conservando el aspect ratio
img_pil.thumbnail((400, 400))
img_array = np.array(img_pil)

img_normalizada = img_array / 255.0
img_plana = img_normalizada.reshape(-1, 3)

# Definimos clases y sus colores de visualización
nombres_clases = ['Pétalos', 'Centro', 'Hojas', 'Sombras']
colores_clases = {
    'Pétalos': '#FFFFFF',  # Blanco
    'Centro': '#FFD700',   # Amarillo
    'Hojas': '#228B22',    # Verde
    'Sombras': '#1A1A1A'   # Oscuro
}

# Extraer muestras automáticamente usando reglas heurísticas de color
X_train = []
Y_clases = []

for pixel in img_plana:
    r, g, b = pixel
    # 0: Pétalos (Blanco/Gris claro)
    if r > 0.7 and g > 0.7 and b > 0.7:
        if np.random.rand() < 0.1: # Tomar solo 10% para no saturar
            X_train.append(pixel)
            Y_clases.append(0)
    # 1: Centro (Amarillo)
    elif r > 0.7 and g > 0.6 and b < 0.5:
        X_train.append(pixel)
        Y_clases.append(1)
    # 2: Hojas (Verde)
    elif g > r + 0.1 and g > b + 0.1 and g < 0.8:
        if np.random.rand() < 0.2:
            X_train.append(pixel)
            Y_clases.append(2)
    # 3: Sombras/Fondo Oscuro
    elif r < 0.3 and g < 0.3 and b < 0.3:
        if np.random.rand() < 0.1:
            X_train.append(pixel)
            Y_clases.append(3)

X_train = np.array(X_train)
Y_clases = np.array(Y_clases)

print(f"Dimensiones de la imagen: {img_array.shape}")
print(f"Muestras de entrenamiento totales: {len(X_train)}")
for i, nombre in enumerate(nombres_clases):
    print(f"  {nombre}: {np.sum(Y_clases == i)} muestras")

# %% [markdown]
# ## 2. Entrenar Perceptrones (One-vs-All)

# %%
perceptrones = []

for i in range(len(nombres_clases)):
    print(f"Entrenando perceptrón para '{nombres_clases[i]}'...")
    Y_binario = np.where(Y_clases == i, 1, 0)
    
    p = Perceptron(n_entradas=3, tasa_aprendizaje=0.01, epocas_maximas=50)
    p.entrenar(X_train, Y_binario)
    perceptrones.append(p)

aciertos = sum(predecir_multiclase(perceptrones, X_train[i]) == Y_clases[i] for i in range(len(X_train)))
print(f"Precisión en entrenamiento: {(aciertos/len(X_train))*100:.1f}%")

# %% [markdown]
# ## 3. Segmentar toda la imagen

# %%
predicciones = np.array([predecir_multiclase(perceptrones, pixel) for pixel in img_plana])
mapa_segmentado = predicciones.reshape(img_array.shape[:2])
print("¡Segmentación completada!")

# %% [markdown]
# ## 4. Visualizar Resultados

# %%
cmap_custom = ListedColormap([colores_clases[c] for c in nombres_clases])

fig, ax = plt.subplots(1, 3, figsize=(20, 6))

ax[0].imshow(img_array)
ax[0].set_title('Imagen Original', fontsize=14, fontweight='bold')
ax[0].axis('off')

im_seg = ax[1].imshow(mapa_segmentado, cmap=cmap_custom)
ax[1].set_title('Segmentación por Perceptrón', fontsize=14, fontweight='bold')
ax[1].axis('off')

ax[2].imshow(img_array)
ax[2].imshow(mapa_segmentado, cmap=cmap_custom, alpha=0.5)
ax[2].set_title('Superposición', fontsize=14, fontweight='bold')
ax[2].axis('off')

import matplotlib.patches as mpatches
parches = [mpatches.Patch(color=colores_clases[c], label=c) for c in nombres_clases]
ax[1].legend(handles=parches, loc='lower right')

plt.suptitle("Segmentación de Colores de la Flor", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Pesos Finales

# %%
print("PESOS FINALES DE CADA PERCEPTRÓN")
for i, p in enumerate(perceptrones):
    print(f"{nombres_clases[i]:<12}: R={p.pesos[0]:+.4f}, G={p.pesos[1]:+.4f}, B={p.pesos[2]:+.4f} | Umbral={p.umbral:+.4f}")
