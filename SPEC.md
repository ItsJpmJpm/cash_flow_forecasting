# Cash Flow Forecasting - Neural Networks

## 1. Resumen del Proyecto

**Objetivo**: Implementar un sistema de predicción de flujos de caja (cash flow forecasting) utilizando redes neuronales, comparándolo con un baseline ARIMA, para una empresa sintética española.

**Empresa Sintética**:
- **Nombre**: TechSolutions Toledo S.L.
- **Sector**: Servicios IT / Consultoría tecnológica
- **Ubicación**: Toledo, España
- **Empleados**: 50
- **Volumen**: 500 transacciones/mes promedio
- **Período de datos**: Enero 2021 - Diciembre 2023 (36 meses)

---

## 2. Requirements (Página 11 del PDF)

### 2.1 Documentación Requerida

| Requisito | Estado |
|-----------|--------|
| Presentación teórica de la red neuronal | ✅ Incluido |
| Justificación de la arquitectura seleccionada | ✅ Incluido |
| Definición de hiperparámetros y su rationale | ✅ Incluido |
| Descripción del dataset y preprocesamiento | ✅ Incluido |
| Análisis de resultados comparativos | ✅ Incluido |
| Conclusiones y trabajo futuro | ✅ Incluido |

### 2.2 Modelo de Negocio de TechSolutions Toledo

```
INGRESOS:
├── Consultoría proyectos (60% del total)
├── Mantenimiento software (25%)
├── Licencias y SaaS (10%)
└── Formación (5%)

EGRESOS:
├── Nómina empleados
├── Proveedores y subcontratas
├── Infraestructura (cloud, office)
├── Marketing y comerciales
└── Costs operasionals fijos
```

### 2.3 Factores que Influyen en el Cash Flow

- **Estacionalidad**: Q4 suele ser bajo (vacaciones Navidad), Q1 fuerte post-navidad
- **Ciclo de cobro**: 30-60 días promedio
- **Pagos proveedores**: 30 días
- **Nómina**: 25 de cada mes
- **IVA**: Trimestral (abono/resto)

---

## 3. Arquitectura de la Red Neuronal

### 3.1 Tipo: MLP (Multi-Layer Perceptron)

```
Input Layer: 12 neuronas
    ├── mes (mes del año: 1-12)
    ├── día_mes (día de nómina: 0/1)
    ├── trimestre (Q1-Q4)
    ├── tendencia (mes_normalizado)
    ├── estacionalidad_mes
    ├── variación_año_anterior
    ├── ingresos_dias_30
    ├── egresos_dias_30
    ├── balance_pendiente
    ├── ratio_cobros_pagos
    └── shock_covid (0/1)

Hidden Layer 1: 64 neuronas (ReLU)
Dropout: 0.3

Hidden Layer 2: 32 neuronas (ReLU)
Dropout: 0.3

Output Layer: 1 neurona (cash flow predicho)
```

### 3.2 Hiperparámetros

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Learning Rate | 0.001 | Adam por defecto, estable |
| Epochs | 500 | Temprano stop con patience=50 |
| Batch Size | 32 | Balance entre velocidad y precisión |
| Optimizer | Adam | Mejor para datos financieros |
| Loss Function | MSE | Standard para regresión |
| Dropout | 0.3 | Previene overfitting |
| Validation Split | 0.2 | Suficiente para 36 meses |

### 3.3 Baseline ARIMA

- **Orden (p,d,q)**: Seleccionado por AIC/BIC
- **Seasonal**: SARIMA con periodo 12 (mensual)
- **Orden estacional (P,D,Q,s)**: (1,1,1,12)

---

## 4. Dataset Sintético

### 4.1 Variables Generadas

| Variable | Descripción | Rango |
|----------|-------------|-------|
| fecha | Mes-Año | 2021-01 a 2023-12 |
| ingresos_totales | Facturación mensual | 180.000 - 450.000 € |
| egresos_totales | Pagos mensuales | 150.000 - 400.000 € |
| cash_flow_real | Ingresos - Egresos | -50.000 a +150.000 € |
| clientes_activos | Número de clientes | 40-80 |
| proyectos_pendientes | Pipeline | 10-30 |
| dias_cartera | Días promedio cobro | 35-65 |
| estacionalidad | Factor 0-1 | 0.7-1.3 |
| shock_externo | COVID, crisis, etc. | 0/1 |

### 4.2 Estacionalidad España

```
Q1 (Ene-Mar):   Meses bajos, nueva temporada, decisiones lentas
Q2 (Abr-Jun):   Recuperación, proyectos nuevos
Q3 (Jul-Sep):   Verano ralentizado (Julio-Agosto muy bajo)
Q4 (Oct-Dic):   Alta actividad pre-navidad + baja diciembre
```

### 4.3 Festividades y Eventos España

- **Enero**: Ajuste post-navidad, muy bajo
- **Julio-Agosto**: Vacaciones generales, mínima actividad
- **Diciembre**: Navidades, baja actividad segunda quincena
- **Semana Santa**: Variable (Marzo/Abril)

---

## 5. Métricas de Evaluación

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| MAE | Σ|y-ŷ|/n | Error absoluto medio en € |
| RMSE | √(Σ(y-ŷ)²/n) | Penaliza errores grandes |
| MAPE | 100*Σ|y-ŷ|/|y|/n | Error porcentual |
| R² | 1 - SSres/SStot | Varianza explicada |

---

## 6. Bot de Telegram - Alertas de Baja Demanda

### 6.1 Funcionalidad

**Trigger**: Cuando el cash flow predicho sea inferior al umbral definido.

**Umbral**: Configurable (default: cash_flow < 50.000 €)

**Notificación**:
```
🚨 ALERTA: Baja demanda detectada

📅 Período: [MES/AÑO]
💰 Cash Flow Predicho: [CANTIDAD] €
📊 Porcentaje vs mes anterior: [%]

📈 Factores detectados:
   - Ingresos bajos últimos 30 días
   - Estacionalidad negativa
   - Posible caída de clientes

⚠️ Recomendación: Revisar pipeline y acelerar cobros
```

### 6.2 Configuración

- Token del bot: Variable de entorno `TELEGRAM_BOT_TOKEN`
- Chat ID: Configurable para múltiples destinatarios
- Frecuencia: Verificación diaria automática

---

## 7. Estructura del Proyecto

```
cash_flow_forecasting/
├── README.md
├── requirements.txt
├── SPEC.md                          # Este documento
├── docs/
│   ├── empresa_sintetica.md
│   ├── arquitectura_red.md
│   └── evaluacion_modelos.md
├── data/
│   ├── raw/
│   │   └── cash_flow_data.csv
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_arima_model.ipynb
│   ├── 03_neural_network.ipynb
│   └── 04_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── data_preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arima_model.py
│   │   └── neural_network.py
│   ├── evaluation.py
│   ├── telegram_bot.py
│   └── config.py
├── tests/
│   ├── test_data_generator.py
│   ├── test_models.py
│   └── test_telegram.py
└── models/
    └── saved/
```

---

## 8. Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.12.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
python-telegram-bot>=20.0
python-dotenv>=1.0.0
jupyter>=1.0.0
```

---

## 9. Conclusiones Esperadas

### 9.1 Comparación Modelo ARIMA vs Neural Network

| Aspecto | ARIMA | Neural Network |
|---------|-------|----------------|
| Interpretabilidad | Alta | Media |
| Captura de patrones complejos | Baja | Alta |
| Estacionalidad | Excelente | Buena |
| Shock externo | Mala | Mejor |
| Tiempo de entrenamiento | Rápido | Medio |

### 9.2 Trabajo Futuro

1. **LSTM para series temporales** - Capturar dependencias a largo plazo
2. **Ensemble models** - Combinar ARIMA + NN
3. **Feature engineering avanzado** - Incorporar datos externos (feriados,宏观经济)
4. **Modelo de detección de anomalías** - Identificar caídas inesperadas
5. **Dashboard interactivo** - Visualización en tiempo real

---

## 10. Autores

Práctica académica - IA en Finanzas y Contabilidad
Universidad [Nombre] - Curso 2023/2024