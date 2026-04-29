# Proyecto de Detección de Residuos para Reciclaje

Este proyecto se centra en la detección y clasificación de objetos de basura en diversos entornos para ayudar en los esfuerzos de reciclaje. Utilizamos el conjunto de datos **TACO (Trash Annotations in Context)**, que contiene imágenes de alta calidad de basura en fondos diversos.

## Objetivo del Proyecto
Desarrollar un sistema de visión por computadora robusto capaz de identificar basura y categorizarla en cinco grupos principales de reciclaje:
- **Amarillo**: Plásticos, latas y metales.
- **Azul**: Papel y cartón.
- **Verde**: Vidrio.
- **Marrón**: Residuos orgánicos.
- **Gris**: Resto / otros.

---

## 1. Pipeline de Procesamiento de Datos

Para garantizar la consistencia entre los diferentes modelos (SSD y YOLO), implementamos una estrategia de procesamiento de datos unificada:

### Mapeo de Categorías
El dataset original de TACO contiene 60 categorías. Las mapeamos en nuestros 5 grupos objetivo para simplificar la tarea de clasificación y mejorar el rendimiento del modelo.

### Pasos de Procesamiento de Imágenes
- **Filtrado**: Se excluyen las imágenes que no contienen ninguna anotación para nuestros 5 grupos objetivo.
- **Redimensionado**: Las imágenes se redimensionan a un tamaño uniforme (por ejemplo, $300 \times 300$ para SSD300) manteniendo la relación de aspecto o utilizando escalado/relleno según sea necesario.
- **División**: El conjunto de datos se divide en **Entrenamiento (80%)**, **Validación (10%)** y **Test (10%)** utilizando una semilla aleatoria fija (42) para garantizar la reproducibilidad y una comparación justa entre experimentos.

### Conversión de Etiquetas
- **SSD**: Las anotaciones se convierten del formato COCO al formato **Pascal VOC XML**.
- **YOLO**: Las anotaciones se convierten al formato **YOLO TXT normalizado**.

---

## 2. Modelos de Detección

### Experimento 1: YOLO (You Only Look Once)
Implementado utilizando el framework `ultralytics` (YOLOv8). Este modelo proporciona una línea base de alta velocidad y precisión para la detección de objetos.

- **Arquitectura y Backbone**: 
  - Basado en **YOLOv8 Nano (yolov8n)** y **Small (yolov8s)**.
  - Utiliza el backbone **CSPDarknet**, que emplea convoluciones Cross-Stage Partial para mejorar el flujo de gradientes y reducir el costo computacional.
- **Configuración de Imagen**: Redimensionado dinámico a **640x640** píxeles, utilizando *letterboxing* para mantener la relación de aspecto original.
- **Aumentos (Augmentations)**: Entrenamiento robusto mediante hiperparámetros de aumentos controlados:
  - **HSV (h=0.005, s=0.25, v=0.20)**: Ajustes aleatorios de tono, saturación y brillo.
  - **Transformaciones Afines**: Traslación (0.03) y Escala (0.10).
  - **Flips**: Volteo horizontal aleatorio (50% de probabilidad).
  - *Nota*: Se desactivaron intencionalmente Mosaic y Mixup en las pruebas base para evaluar la consistencia del detector en imágenes individuales.
- **Estrategia de Datos**: 
  - **Oversampling**: Implementación de una técnica de sobremuestreo personalizada para clases minoritarias, garantizando que el modelo vea suficientes ejemplos de residuos menos frecuentes (como vidrio o papel).
  - **División Estratificada**: División Train/Val/Test basada en la clase dominante de cada imagen para mantener la representatividad estadística.

#### Resultados Clave (YOLO)
| Run | Épocas | mAP50 | Precisión | Recall | mAP50-95 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **yolo8n_oversampling_soft_20** | 20 | **0.2867** | 0.3659 | 0.3662 | 0.2007 |
| yolo8n_stratified_oversampling_20 | 20 | 0.2867 | 0.3659 | 0.3662 | 0.2007 |
| yolo11n_oversampling_30 | 30 | 0.2831 | 0.3431 | 0.3718 | 0.2051 |
| yolo8s_oversampling_soft_25 | 25 | 0.2788 | 0.2885 | 0.3635 | 0.2005 |
| yolo11n_oversampling_30_aug | 30 | 0.2746 | 0.3254 | 0.3558 | 0.1914 |
| yolo8n_oversampling_soft_50 | 50 | 0.2695 | 0.3494 | 0.2970 | 0.1982 |
| yolo8s_oversampling_soft_50 | 50 | 0.2674 | 0.3937 | 0.2894 | 0.1903 |
| yolo26n_oversampling_30 | 30 | 0.2530 | 0.3326 | 0.2821 | 0.1717 |
| yolo11n_oversampling_30_aug_soft | 30 | 0.2512 | 0.3410 | 0.2894 | 0.1809 |
| yolo8n_oversampling_soft_30_aug_batch_32 | 25 | 0.2509 | 0.3882 | 0.2767 | 0.1693 |
| yolo8n_oversampling_soft_30_aug | 30 | 0.2428 | 0.2762 | 0.2872 | 0.1717 |
| yolo8n_4clases_30 | 30 | 0.2191 | 0.2893 | 0.2448 | 0.1504 |
| yolo8n_4clases_20 | 20 | 0.2025 | 0.2292 | 0.2911 | 0.1420 |
| yolo8n_20epochs | 20 | 0.2012 | 0.4399 | 0.2497 | 0.1498 |
| yolo8s_20epochs | 20 | 0.1955 | 0.4859 | 0.2237 | 0.1328 |
| yolo8n_aug_soft_30 | 30 | 0.1850 | 0.4453 | 0.2129 | 0.1341 |
| yolo8n_aug_30 | 30 | 0.1790 | 0.4903 | 0.2055 | 0.1262 |

---

### Experimento 2: SSD (Single Shot MultiBox Detector) - VGG16
Implementado en PyTorch utilizando la arquitectura `ssd300_vgg16`.
- **Backbone**: VGG16 (pre-entrenado en ImageNet).
- **Aumentos (Augmentations)**: Uso de `albumentations`: Volteo horizontal, brillo/contraste, y transformaciones afines.
- **Optimización**: Ajuste de hiperparámetros mediante búsqueda en cuadrícula (grid search).

#### Resultados Clave (SSD)
**Ajuste de Hiperparámetros (Grid Search):**
| Configuración | LR | WD | Mejor Val Loss | Resultado |
|:---|:---|:---|:---|:---|
| Config 1 | 0.01 | 0.0005 | 8900.53 | Gradientes explotaron (NaN) |
| Config 2 | 0.01 | 0.0001 | 4.34e+23 | Gradientes explotaron (NaN) |
| Config 3 | 0.001 | 0.0005 | 5.6106 | Estable |
| **Config 4** | **0.001** | **0.0001** | **5.5501** | **Mejor configuración** |
| Config 5 | 0.0005 | 0.0005 | 5.6806 | Estable pero lento |
| Config 6 | 0.0005 | 0.0001 | 5.7204 | Estable pero lento |

**Entrenamiento Final (Mejor Modelo):**
| Métrica | Valor |
|:---|:---|
| Épocas | 50 |
| Mejor Época (Val Loss) | 18 |
| **Mejor Val Loss** | **5.3060** |
| Final Train Loss | 3.2787 |
| Final Val Loss | 5.3804 |

### Experimento 3: MobileNet-SSD (Planificado)
Como tercer experimento, planeamos implementar un cabezal SSD sobre un backbone **MobileNet** (por ejemplo, MobileNetV2 o V3).
- **Motivación**: MobileNet es significativamente más ligero y rápido que VGG16, lo que lo hace ideal para su despliegue en dispositivos móviles o aplicaciones integradas (edge computing).
- **Objetivo**: Evaluar el compromiso entre la precisión de detección y la velocidad de inferencia, buscando una versión "ligera" de nuestro detector de reciclaje.

---

## Cómo Ejecutar

### Requisitos
- Python 3.10+
- PyTorch
- Torchvision
- Albumentations
- Ultralytics (para YOLO)
- PIL, NumPy, tqdm

### Preparación
1. **Generar Dataset Común**:
   ```bash
   python YOLO_implementation/src/data_common.py
   ```
2. **Exportar para SSD**:
   ```bash
   python SSD_implementation/prepare_ssd_dataset.py
   ```
3. **Exportar para YOLO**:
   ```bash
   python YOLO_implementation/src/export_yolo.py
   ```

### Entrenamiento

#### SSD
Ejecuta el notebook de entrenamiento:
`SSD_implementation/train_ssd_model.ipynb`

#### YOLO
Puedes entrenar desde la consola utilizando el script de entrenamiento:
```bash
python YOLO_implementation/src/train_yolo.py --model yolov8n.pt --epochs 20 --imgsz 640
```
