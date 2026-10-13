# Motor de simulación Monte Carlo — Potencial geotérmico del volcán Azufral

Aplicación web (Python y Streamlit) que estima de forma preliminar el potencial geotérmico de un sistema
hidrotermal con el método volumétrico del USGS y simulación Monte Carlo. Los valores por defecto corresponden
al volcán Azufral (Nariño, Colombia), con rangos documentados en el trabajo de grado.

**Trabajo de grado:** Ingeniería de Sistemas, Universidad CESMAG.

## Qué hace
- Entrega P90, P50 y P10 de potencia eléctrica (MWe) y de calor almacenado (EJ).
- Histograma, curva de excedencia, análisis de convergencia y sensibilidad (coeficiente de Spearman).
- Contraste con un método independiente (densidad de potencia).
- Exporta la configuración (JSON) y los resultados (CSV) para repetir y auditar cualquier ejecución.

## Cómo usarla
1. Al abrir la página se ejecuta sola la configuración documentada.
2. En el panel izquierdo puede cambiar la distribución, la semilla o cualquier rango (mínimo, moda, máximo).
3. Pulse **INICIAR SIMULACIÓN** para recalcular.

> Los resultados dependen de los rangos de entrada documentados; son una estimación de superficie, sin validación con datos de campo.

## Ejecutarla en el computador
```
pip install -r requirements.txt
streamlit run MonteCarlo_v9.py
```
Se recomienda Python 3.12.

## Archivos
- `MonteCarlo_v9.py`: la aplicación.
- `requirements.txt`: librerías con las versiones con que se probó.
- `.streamlit/config.toml`: tema de color claro.
