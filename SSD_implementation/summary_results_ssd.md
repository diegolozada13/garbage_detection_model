# Resumen de Resultados - Experimento 2 (SSD VGG16)

Este documento resume los resultados del ajuste de hiperparámetros y el entrenamiento final del modelo SSD300 con backbone VGG16.

## 1. Ajuste de Hiperparámetros (Grid Search)

Se probaron diferentes combinaciones de Tasa de Aprendizaje (LR) y Decaimiento de Peso (WD) durante 10 épocas cada una (con early stopping).

| Configuración | LR | WD | Mejor Val Loss | Resultado |
|:--------------|:---|:---|:---------------|:----------|
| Config 1 | 0.01 | 0.0005 | 8900.53 | Gradientes explotaron (NaN) |
| Config 2 | 0.01 | 0.0001 | 4.34e+23 | Gradientes explotaron (NaN) |
| Config 3 | 0.001 | 0.0005 | 5.6106 | Estable |
| **Config 4** | **0.001** | **0.0001** | **5.5501** | **Mejor configuración** |
| Config 5 | 0.0005 | 0.0005 | 5.6806 | Estable pero lento |
| Config 6 | 0.0005 | 0.0001 | 5.7204 | Estable pero lento |

---

## 2. Entrenamiento Final (Mejor Modelo)

Utilizando la mejor configuración encontrada (**LR=0.001, WD=0.0001**), se procedió a un entrenamiento largo de 50 épocas.

| Métrica | Valor |
|:--------|:------|
| Épocas | 50 |
| Mejor Época (Val Loss) | 18 |
| **Mejor Val Loss** | **5.3060** |
| Final Train Loss | 3.2787 |
| Final Val Loss | 5.3804 |

---

## 3. Observaciones Clave

1.  **Inestabilidad con LR alto**: Las configuraciones con $LR=0.01$ fallaron inmediatamente debido a que los gradientes se volvieron demasiado grandes (NaN/Inf), lo que sugiere que para SSD300 en este dataset, el límite superior de aprendizaje es más bajo que en YOLO.
2.  **Convergencia**: El modelo alcanzó su mejor punto de validación relativamente pronto (época 18), y a partir de ahí la pérdida de validación se estabilizó alrededor de 5.3-5.4, mientras que la de entrenamiento siguió bajando hasta 3.2. Esto indica un **ligero sobreentrenamiento (overfitting)** que podría mitigarse con aumentos de datos más agresivos (como el Mosaic de YOLO).
3. **Probar metodos usados en YOLO**
