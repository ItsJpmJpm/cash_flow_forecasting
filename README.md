# 💰 Cash Flow Forecasting - Neural Networks

## 📋 Descripción

Sistema de predicción de flujos de caja utilizando redes neuronales (MLP) y ARIMA como baseline, para la empresa ficticia **TechSolutions Toledo S.L.**

## 🏢 Empresa Sintética

- **Nombre**: TechSolutions Toledo S.L.
- **Sector**: Servicios IT / Consultoría tecnológica
- **Ubicación**: Toledo, España
- **Empleados**: 50
- **Período**: Enero 2021 - Diciembre 2023

## 📁 Estructura del Proyecto

```
cash_flow_forecasting/
├── SPEC.md                 # Especificación completa del proyecto
├── README.md               # Este archivo
├── requirements.txt       # Dependencias Python
├── src/
│   ├── config.py           # Configuración global
│   ├── data_generator.py  # Generador de datos sintéticos
│   ├── data_preprocessing.py
│   ├── models/
│   │   ├── arima_model.py
│   │   └── neural_network.py
│   ├── evaluation.py
│   └── telegram_bot.py     # Bot de alertas
├── notebooks/              # Jupyter notebooks de análisis
├── data/                   # Datos CSV
└── docs/                   # Documentación adicional
```

## 🚀 Uso Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Generar datos sintéticos

```python
from src.data_generator import generate_cash_flow_data

df = generate_cash_flow_data()
df.to_csv("data/raw/cash_flow_data.csv", index=False)
```

### 3. Entrenar modelos

```python
from src.models.arima_model import train_arima
from src.models.neural_network import train_mlp

# ARIMA baseline
arima_results = train_arima(df)

# Neural Network
nn_model = train_mlp(df)
```

### 4. Comparar y evaluar

```python
from src.evaluation import compare_models, calculate_metrics

results = compare_models(df, arima_model, nn_model)
```

### 5. Configurar Bot de Telegram

```bash
# Crear archivo .env
echo "TELEGRAM_BOT_TOKEN=tu_token_aqui" > .env
echo "ALERT_THRESHOLD=50000" >> .env  # Umbral de alerta en €
```

## 📊 Resultados Esperados

### Métricas de Evaluación

| Modelo | MAE (€) | RMSE (€) | MAPE (%) | R² |
|--------|---------|----------|----------|-----|
| ARIMA | ~12.000 | ~18.000 | ~8% | ~0.75 |
| Neural Network | ~10.000 | ~15.000 | ~6% | ~0.82 |

## 🔔 Bot de Telegram

El bot envía alertas cuando el cash flow predicho está por debajo del umbral configurado.

**Comandos**:
- `/start` - Iniciar el bot
- `/forecast` - Solicitar predicción actual
- `/alert` - Configurar umbral de alerta
- `/status` - Ver estado actual

## 📚 Documentación

- [SPEC.md](SPEC.md) - Especificación completa del proyecto
- [docs/arquitectura_red.md](docs/arquitectura_red.md) - Arquitectura de la red neuronal
- [docs/empresa_sintetica.md](docs/empresa_sintetica.md) - Detalles de la empresa ficticia
- [docs/evaluacion_modelos.md](docs/evaluacion_modelos.md) - Análisis de resultados

## 📝 Requisitos (Página 11 PDF)

✅ Presentación teórica de la red neuronal
✅ Justificación de la arquitectura seleccionada
✅ Definición de hiperparámetros y rationale
✅ Descripción del dataset y preprocesamiento
✅ Análisis de resultados comparativos
✅ Conclusiones y trabajo futuro

## 👥 Autores

Práctica académica - IA en Finanzas y Contabilidad