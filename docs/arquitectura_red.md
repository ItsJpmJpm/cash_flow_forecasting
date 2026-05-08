# Arquitectura de la Red Neuronal - MLP para Cash Flow Forecasting

## 1. Introducción Teórica

### 1.1 ¿Por qué una Red Neuronal?

Las redes neuronales son capaces de:
- **Capturar relaciones no lineales** entre features financieras
- **Aprender patrones complejos** que modelos estadísticos tradicionales no detectan
- **Generalizar** a partir de ejemplos históricos
- **Adaptarse** a cambios en el comportamiento del mercado

### 1.2 MLP (Multi-Layer Perceptron)

Es una red neuronal feedforward donde cada capa está completamente conectada a la siguiente:

```
Input → Hidden 1 → Hidden 2 → Output
```

---

## 2. Arquitectura Detallada

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT LAYER (12 features)                   │
├─────────────────────────────────────────────────────────────────┤
│  mes (1-12)                                                      │
│  trimestre (Q1-Q4 encoded)                                       │
│  dia_nomina (0/1)                                               │
│  tendencia_mes (0-35 normalizado)                              │
│  estacionalidad (factor 0.7-1.3)                                │
│  variacion_mes_anterior (%)                                     │
│  ratio_cobros_pagos (0.5-2.0)                                   │
│  dias_cartera_promedio (30-60)                                 │
│  num_clientes_activos (40-80)                                  │
│  proyectos_pendientes (10-30)                                  │
│  ingresos_mes_anterior (€ normalizado)                         │
│  egresos_mes_anterior (€ normalizado)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    [Fully Connected Dense]
                    Activation: ReLU
                    Units: 64
                              ↓
                         DROPOUT 0.3
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      HIDDEN LAYER 1 (64 units)                    │
├─────────────────────────────────────────────────────────────────┤
│  ReLU activation                                                │
│  Kernel initializer: He normal                                  │
│  L2 regularization: 0.001                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    [Fully Connected Dense]
                    Activation: ReLU
                    Units: 32
                              ↓
                         DROPOUT 0.3
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      HIDDEN LAYER 2 (32 units)                   │
├─────────────────────────────────────────────────────────────────┤
│  ReLU activation                                                │
│  Kernel initializer: He normal                                  │
│  L2 regularization: 0.001                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    [Fully Connected Dense]
                    Activation: Linear
                    Units: 1
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER (1 value)                     │
├─────────────────────────────────────────────────────────────────┤
│  cash_flow_predicho (€)                                         │
│  Activation: None (linear para regresión)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Hiperparámetros

### 3.1 Tabla de Parámetros

| Parámetro | Valor | Justificación |
|-----------|-------|----------------|
| Learning Rate | 0.001 | Valor por defecto de Adam, estable |
| Learning Rate Decay | 1e-6 | Decaimiento gradual para estabilidad |
| Epochs | 500 | Con early stopping (patience=50) |
| Batch Size | 32 | Balance precisión vs velocidad |
| Optimizer | Adam | Mejor para datos financieros con ruido |
| Loss Function | MSE | Error cuadrático medio estándar |
| Validation Split | 0.2 | 36 meses es poco, 80/20 óptimo |
| Patience | 50 | Suficiente para converger sin overfit |
| Min Delta | 0.001 | Cambio mínimo para early stop |

### 3.2 Regularización

| Técnica | Valor | Propósito |
|---------|-------|-----------|
| Dropout Layer 1 | 0.3 | Previene co-adaptación |
| Dropout Layer 2 | 0.3 | Previene overfitting |
| L2 Kernel Regularizer | 0.001 | Penaliza pesos grandes |
| Constraint Max Norm | 3.0 | Estabiliza convergencia |

---

## 4. Preprocesamiento de Datos

### 4.1 Normalización

```python
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()  # Features
scaler_y = StandardScaler()  # Target

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1))
```

### 4.2 Features Engineering

#### Features temporales
- **mes**: One-hot encoding o categorical
- **trimestre**: One-hot encoding (Q1-Q4)
- **dia_nomina**: Si el mes tiene nómina (siempre, día 25)

#### Features de ratio
- **ratio_cobros_pagos**: ingresos_30d / egresos_30d
- **dias_cartera**: Promedio ponderado de antigüedad facturas

#### Features de tendencia
- **media_movil_3_meses**: Suavizado de corto plazo
- **variacion_vs_mes_anterior**: Crecimiento/decrecimiento

### 4.3 División Train/Test

```python
# Para series temporales NO usamos random split
# Usamos los primeros 24 meses para training
# Los últimos 12 meses para validación/test

train_size = 24  # 2021 + 2022
X_train = X[:train_size]
X_test = X[train_size:]
```

---

## 5. Funciones de Activación

### 5.1 ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)

Ventajas:
- Computacionalmente eficiente
- No problemas de vanishing gradient para positivos
- Sparcity natural (neuronas apagadas)
```

### 5.2 Linear (Output)

Para regresión usamos activación linear (sin función):
```
f(x) = x
```

---

## 6. Backpropagation y Optimización

### 6.1 Adam Optimizer

Adam combina las ventajas de:
- **AdaGrad**: Funciona bien con features sparse
- **RMSProp**: Funciona bien con ruido no estacionario

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)
```

### 6.2 Early Stopping

```python
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    min_delta=0.001
)
```

---

## 7. Comparación con Baseline ARIMA

### 7.1 ARIMA(p,d,q)

| Aspecto | ARIMA | MLP |
|---------|-------|-----|
| linealidad | Solo lineal | Lineal + no lineal |
| Estacionalidad | SARIMA nativa | Feature engineering |
| Interpretabilidad | Alta | Media |
| Datos necesarios | 50+ observaciones | 30+ suficiente |
| Overfitting risk | Bajo | Medio (mitigado con dropout) |
| Velocidad | Rápido | Medio |

### 7.2 Cuándo usar cada uno

| Escenario | Modelo Recomendado |
|-----------|---------------------|
| Datos muy ruidosos | ARIMA |
| Patrones complejos/multifactor | MLP |
| Estacionalidad fuerte | ARIMA o MLP con features |
| Series cortas (<24 obs) | ARIMA |
| Series largas (>36 obs) | MLP |

---

## 8. Métricas de Error

### 8.1 MSE (Mean Squared Error)

```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### 8.2 MAE (Mean Absolute Error)

```python
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

### 8.3 MAPE (Mean Absolute Percentage Error)

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

---

## 9. Referencias Teóricas

1. ** Hochreiter, S. & Schmidhuber, J. (1997)** - Long Short-Term Memory
2. ** Bishop, C. (1995)** - Neural Networks for Pattern Recognition
3. ** Hyndman, R.J. & Athanasopoulos, G. (2021)** - Forecasting: Principles and Practice
4. ** Goodfellow, I. et al. (2016)** - Deep Learning (MIT Press)