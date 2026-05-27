# Segmentación de Imágenes con Perceptrón

Este proyecto implementa un Perceptrón Multiclase (arquitectura One-vs-All) desde cero, utilizando Python y NumPy. El objetivo principal es realizar tareas de clustering y segmentación de imágenes basándose únicamente en el espacio de color RGB.

## Características y Fases del Proyecto

El desarrollo se ha dividido en tres experimentos principales, incrementando progresivamente la complejidad de los datos:

### 1. Datos Sintéticos (`perceptron_colores.py` / `.ipynb`)
- **Descripción**: Implementación base del perceptrón para clasificación binaria y posterior extensión a multiclase (One-vs-All).
- **Proceso**: Se entrenó el modelo con matrices de colores artificiales (rojo, verde, azul) con ruido gaussiano añadido para simular variabilidad. 
- **Objetivo**: Validar el funcionamiento matemático del perceptrón, el cálculo del error (delta) y la actualización de pesos.

### 2. Imagen Real: Paisaje / Playa (`perceptron_imagen_real.py` / `.ipynb`)
- **Descripción**: Aplicación del modelo a una pintura realista (`imagen_playa.png`).
- **Proceso**: Se definieron Regiones de Interés (ROI) estáticas para extraer píxeles de entrenamiento de 4 clases distintas:
  - Cielo
  - Vegetación
  - Arena
  - Agua / Mar
- **Resultados**: El modelo logró procesar más de 32,000 píxeles, obteniendo una precisión aproximada del 67.7%. Demostró capacidad para separar grandes masas de color pese a las texturas y degradados naturales.

### 3. Imagen Real: Flor con Muestreo Heurístico (`perceptron_imagen_flor.py` / `.ipynb`)
- **Descripción**: Refinamiento del proceso de extracción de datos aplicado a una fotografía de una flor (`imagen_flor.png`).
- **Proceso**: Se implementó una técnica de **muestreo heurístico** automatizado. En lugar de definir regiones fijas manualmente por coordenadas, el algoritmo filtra píxeles en toda la imagen en base a umbrales de intensidad y diferencias relativas entre los canales R, G y B. 
- **Clases Extraídas**:
  - Pétalos
  - Centro de la flor
  - Hojas (Fondo verde)
  - Fondo oscuro
- **Resultados**: Mayor representatividad en el conjunto de entrenamiento, permitiendo una segmentación más precisa y robusta sin requerir la delimitación manual de rectángulos de interés.

## Estructura del Proyecto

El proyecto incluye tanto scripts de Python puro (`.py`) como notebooks interactivos de Jupyter (`.ipynb`) para facilitar la ejecución y la visualización de resultados (gráficos de convergencia de error, mapas de segmentación y diagramas de distribución RGB).

- Archivos principales de código:
  - `perceptron_colores.*`
  - `perceptron_imagen_real.*`
  - `perceptron_imagen_flor.*`
- Archivos de testing automatizado (entornos sin interfaz gráfica):
  - `test_run.py`, `test_real.py`, `test_flor.py`

## Requisitos

- Python 3.x
- Jupyter Notebook
- Bibliotecas: `numpy`, `matplotlib`, `Pillow` (PIL)

## Ejecución

Para visualizar los resultados de forma interactiva y probar el entrenamiento en tiempo real, abra cualquiera de los archivos `.ipynb` en Jupyter Notebook o JupyterLab:

```bash
jupyter notebook
```
