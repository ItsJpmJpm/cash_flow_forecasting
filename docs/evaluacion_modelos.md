# Evaluación y Comparación de Modelos

## 1. Marco de Evaluación

### 1.1 Metodología

```
Dataset: 36 meses (2021-2023)
├── Training: 24 meses (2021-2022)
├── Validation: 6 meses (01/2023 - 06/2023)
└── Test: 6 meses (07/2023 - 12/2023)

Evaluación: Walk-forward validation (monthly forecast)
```

### 1.2 Criterios de Éxito

| Criterio | Target |
|----------|--------|
| MAPE | < 10% |
| MAE | < 15.000 € |
| RMSE | < 20.000 € |
| R² | > 0.70 |
| Direction Accuracy | > 75% |

---

## 2. Métricas de Evaluación

### 2.1 Definiciones

| Métrica | Fórmula | Descripción |
|---------|---------|-------------|
| **MAE** | Σ|yᵢ - ŷᵢ|/n | Error absoluto medio en euros |
| **RMSE** | √(Σ(yᵢ-ŷᵢ)²/n) | Raíz del error cuadrático medio |
| **MAPE** | 100·Σ|yᵢ-ŷᵢ|/|yᵢ|/n | Error porcentual absoluto medio |
| **R²** | 1 - SS_res/SS_tot | Coeficiente de determinación |
| **Direction** | % correct direction | Accuracy de tendencia (subida/bajada) |
| **Theil's U** | √Σ(y-ŷ)²/√Σ(y-y_prev)² | Comparación vs naive |

### 2.2 Cálculo de Métricas

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
```

---

## 3. Comparación ARIMA vs Neural Network

### 3.1 Tabla de Resultados Esperados

| Métrica | ARIMA | Neural Network | Ganador |
|---------|-------|----------------|---------|
| MAE (€) | 12.500 | 9.800 | NN |
| RMSE (€) | 18.200 | 14.500 | NN |
| MAPE (%) | 8.5% | 6.2% | NN |
| R² | 0.72 | 0.85 | NN |
| Direction Accuracy | 70% | 78% | NN |
| Training Time (s) | 2.3 | 45.2 | ARIMA |

### 3.2 Análisis por Componente

#### Captura de Estacionalidad

```
ARIMA: ⭐⭐⭐⭐⭐
  - SARIMA captura estacionalidad nativa
  - Period=12 (mensual) muy preciso

Neural Network: ⭐⭐⭐⭐☆
  - Feature engineering de estacionalidad
  - Necesita más datos para aprender patrón
```

#### Respuesta a Shocks

```
ARIMA: ⭐⭐
  - Lento en ajustar a cambios abruptos
  - Asume平稳idad

Neural Network: ⭐⭐⭐⭐
  - Aprende patrones de shocks pasados
  - Feature "shock_externo" ayuda
```

#### Interpretabilidad

```
ARIMA: ⭐⭐⭐⭐⭐
  - Coeficientes interpretables
  - AIC/BIC para selección de parámetros

Neural Network: ⭐⭐
  - "Caja negra"
  - SHAP values para explicabilidad parcial
```

---

## 4. Visualización de Resultados

### 4.1 Gráficos Requeridos

| Gráfico | Descripción |
|---------|-------------|
| Time Series Plot | Predicción vs Real con intervalo de confianza |
| Scatter Plot | Predicción vs Real (regression line) |
| Residual Analysis | Histograma y ACF de residuos |
| Error Distribution | Error absoluto por mes |
| Feature Importance | Para neural network (si usa SHAP) |

### 4.2 Código de Visualización

```python
import matplotlib.pyplot as plt

# Plot 1: Time Series Comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ARIMA
axes[0].plot(dates_test, y_test, 'b-', label='Real', linewidth=2)
axes[0].plot(dates_test, y_pred_arima, 'r--', label='ARIMA', linewidth=2)
axes[0].fill_between(dates_test, ci_lower, ci_upper, alpha=0.3)
axes[0].set_title('ARIMA - Cash Flow Forecast')
axes[0].legend()

# Neural Network
axes[1].plot(dates_test, y_test, 'b-', label='Real', linewidth=2)
axes[1].plot(dates_test, y_pred_nn, 'g--', label='MLP', linewidth=2)
axes[1].set_title('Neural Network - Cash Flow Forecast')
axes[1].legend()

plt.tight_layout()
plt.savefig('docs/comparison_forecast.png', dpi=300)
```

---

## 5. Análisis de Errores

### 5.1 Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| MAE alto en meses bajos | Estacionalidad no captada | Añadir features de estacionalidad |
| Lag sistemático | Modelo no diferencia bien | Probar con más lags |
| Overfitting | Demasiados epochs | Early stopping más agresivo |
| Drift | Tendencia no captada | Diferenciación adicional |

### 5.2 Diagnosis

```python
# Análisis de residuos
residuals = y_test - y_pred

# Ljung-Box test (autocorrelación)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10])

# Normalidad de residuos (Jarque-Bera)
from scipy.stats import jarque_bera
jb_stat, jb_pvalue = jarque_bera(residuals)
```

---

## 6. Conclusiones

### 6.1 Resumen de Resultados

| Aspecto | Conclusión |
|---------|------------|
| Accuracy general | NN supera a ARIMA en todas las métricas |
| Estacionalidad | ARIMA ligeramente mejor |
| Shocks externos | NN captura mejor patrones complejos |
| Robustez | NN más sensible a outliers |
| Speed | ARIMA 20x más rápido |

### 6.2 Recomendación Final

**Neural Network** para predicción operativa diaria.
**ARIMA** para benchmark y explicación a stakeholders.

Modelo híbrido (ensemble) podría mejorar aún más.

---

## 7. Trabajo Futuro

### 7.1 Mejoras Inmediatas

1. **Hyperparameter tuning**: Grid search o Bayesian optimization
2. **Feature selection**: PCA o recursive feature elimination
3. **Cross-validation**: Expandable window validation

### 7.2 Modelos Avanzados

| Modelo | Descripción | Beneficio |
|--------|-------------|-----------|
| LSTM | Long Short-Term Memory | Captura dependencias largas |
| GRU | Gated Recurrent Units | Más rápido que LSTM |
| Transformer | Attention mechanism | Paralelización |
| Prophet | Facebook's model | Estacionalidad automática |
| Ensemble | Combinar modelos | Robustez |

### 7.3 Integraciones

1. **Real-time data**: Conectar a ERP/CRM
2. **Alertas automáticas**: Sistema de阈值 dinámico
3. **Dashboard**: Streamlit o Dash para visualización
4. **API**: Exponer predictions como servicio