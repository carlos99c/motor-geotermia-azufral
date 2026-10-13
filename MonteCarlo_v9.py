import streamlit as st
import numpy as np
import pandas as pd
import json
import io
import re
import unicodedata
import plotly.graph_objects as go
from scipy import stats

# =============================================================================
# NOTAS DE VERSIÓN (v2) — cambios respecto al script original
# -----------------------------------------------------------------------------
# Se añaden 6 mejoras metodológicas solicitadas, preservando la arquitectura,
# el flujo de la interfaz (sidebar -> validación -> motor -> dashboard) y el
# estilo del script original:
#   1) (retirada en la v8) Correlación entre variables vía Iman-Conover.
#   2) Reporte del calor almacenado (Q) en Exajulios (EJ) como salida propia.
#   3) Curva de probabilidad de excedencia (CDF invertida / "S-curve").
#   4) Análisis de convergencia de la simulación (media y percentiles vs N).
#   5) Método de verificación independiente (densidad de potencia, MWe/km²).
#   6) Significancia estadística (p-valor) del coeficiente de Spearman.
# La lógica de cálculo físico se separó en funciones puras (calcular_potencia,
# calor_almacenado_EJ) para que sean testeables fuera de Streamlit, sin
# reestructurar el resto del archivo.
#
# NOTAS DE VERSIÓN (v3) — mejoras inspiradas en Crystal Ball / Paucarani
# -----------------------------------------------------------------------------
#   7) Corrección de la distribución Normal: se reemplaza el recorte
#      (np.clip) por una Normal Truncada real (scipy.stats.truncnorm), que
#      no genera picos artificiales de probabilidad en los límites.
#   8) Calculadora de "Certeza" (rango de probabilidad personalizado),
#      equivalente al recurso homónimo de Crystal Ball.
#   9) Vista alternativa de sensibilidad "Contribución a la Varianza (%)",
#      calculada como en Crystal Ball (ρ² normalizado a 100%, con signo),
#      para comparar en el mismo formato que el estudio de Paucarani.
#  10) Gráfico de superposición (overlay) entre el método volumétrico y el
#      método de verificación independiente (densidad de potencia).
#
# NOTAS DE VERSIÓN (v4) — generalización a otros sistemas geotérmicos
# -----------------------------------------------------------------------------
#  11) El nombre del sistema geotérmico es un parámetro (barra lateral y JSON):
#      títulos, historial y nombres de archivo ya no dicen "Azufral" fijo.
#  12) La comparación con antecedentes es opcional y editable: se muestra solo
#      si se activa, y la lista vive en ANTECEDENTES_POR_DEFECTO o en la clave
#      "antecedentes" del JSON. Se eliminó el texto que dictaminaba si los
#      rangos de entrada "ameritan revisión": el motor no emite veredictos.
#  13) Gráfico de sensibilidad: la vista principal muestra únicamente el
#      coeficiente de Spearman (sin porcentajes), con etiquetas legibles fuera
#      de las barras. La vista "Contribución a la Varianza (%)" queda oculta
#      salvo que se cambie MOSTRAR_VISTA_CONTRIBUCION a True.
#  15) Tarjetas KPI: se quitaron los rótulos "Probable", "Promedio" y "Máximo", que
#      no corresponden a P90, P50 ni P10 (el máximo es otra fila de la tabla).
#  14) La variable Tf se rotula "Temperatura de Referencia" (T0 del método
#      volumétrico), no "de rechazo". Los nombres de columna del CSV no
#      cambian (Temp_Rechazo_C) para no romper análisis ya hechos con ellos.
#
# NOTAS DE VERSIÓN (v5) — configuración por defecto con respaldo documental
# -----------------------------------------------------------------------------
#  16) Solo cambian los tripletes que el software trae por defecto; el código de
#      cálculo y la interfaz no se modificaron. Cada valor tiene su fuente en el
#      comentario que acompaña a su línea (SGC = Servicio Geológico Colombiano,
#      2021, "Actualización del modelo conceptual descriptivo del sistema
#      geotérmico del Volcán Azufral"). Un archivo .json cargado desde la barra
#      lateral sigue reemplazando estos valores, de modo que las configuraciones
#      guardadas con la versión anterior reproducen los resultados anteriores.
# =============================================================================

#  17) (v6) Área por defecto 3-4-5 km² (antes 3-3-3): mínimo del SGC, máximo del análogo
#      Sumikawa (Holmes y Dirkzwager, 2025) y moda como promedio (Halcon et al., 2021).
#      Las demás variables no cambian respecto de la v5.

#  18) (v7) CORRECCIÓN DE FÓRMULA: el factor de planta pasa al denominador (potencia = Q·Rf·η / (Pf·t)).
#      Verificado: con los datos de Paucarani el caso base da 73.0116 MWe, igual al publicado por Flores et al. (2018).
#      Las versiones anteriores daban 65.8929. Los resultados de v1-v6 no son comparables con los de la v7.

#  19) (v8) Se RETIRÓ el módulo de correlación entre variables (Iman-Conover) y la casilla
#      "Activar muestreo correlacionado": no se había verificado y no era necesario. El motor
#      muestrea todas las variables de forma independiente, y esa independencia se comprueba
#      sobre los datos exportados (matriz de correlación del objetivo 2). Un .json guardado
#      con las claves de correlación de versiones anteriores se carga sin error: se ignoran.
#      Los resultados son idénticos a los de la v7 con la correlación desactivada.

#  20) (v9) CAMBIOS:
#      - El archivo JSON ahora guarda y restaura la distribución elegida (clave "tipo_dist"); antes se perdía al recargar.
#      - Histograma: se agrega la línea del P50 y las etiquetas de P90, P50 y P10 van en una franja libre sobre las barras.
#      - Curva de excedencia: las etiquetas de los tres puntos van debajo de la curva, con guía, y la figura lleva título.
#      - Tema de color: fondo azul-gris suave, gráficos y tablas sobre paneles blancos, texto oscuro de alto contraste.
#      - Todas las figuras comparten un estilo común (función estilo_figura) y se muestran sin el tema de Streamlit (theme=None).
#      - Redacción: «validación cruzada» pasa a «contraste cruzado», coherente con el informe (verificación frente a validación).
#      - Versión web: al abrir la página se ejecuta sola la configuración documentada (no queda la pantalla vacía) y se agrega
#        una guía corta para el jurado; la línea AUTORIA permite mostrar el nombre del autor bajo el título.
#      Los resultados numéricos son idénticos a los de la v8.

# --- OPCIONES DE VISUALIZACIÓN ---
MOSTRAR_VISTA_CONTRIBUCION = False   # True reactiva la vista de porcentajes

# --- ANTECEDENTES (opcionales). Reemplazar al usar otro sistema geotérmico ---
ANTECEDENTES_POR_DEFECTO = [
    {"estudio": "Zapata (2015)", "mwe": "89.5", "ej": "—"},
    {"estudio": "Alfaro et al. (2021) — escenario optimista", "mwe": "81.9", "ej": "9.6"},
    {"estudio": "Alfaro et al. (2021) — escenario pesimista", "mwe": "16.41", "ej": "2.5"},
    {"estudio": "SGC (2021) — rutina GEOKERI", "mwe": "26.19", "ej": "2.7"},
]


def nombre_archivo(texto):
    """Convierte un nombre (p. ej. 'Volcán Azufral') en un texto seguro para archivos."""
    t = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_") or "sistema"


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Simulación Monte Carlo - Potencial Geotérmico", layout="wide")

# Estilos CSS personalizados para tarjetas de métricas P90, P50 y P10
st.markdown("""
    <style>
    /* ===== TEMA DE COLOR (v9) =====
       Fondo azul-gris suave para el área principal y un tono un poco más oscuro para el panel lateral.
       Los gráficos y las tablas van sobre paneles BLANCOS, así todo número y etiqueta conserva el máximo contraste.
       Para cambiar la paleta, edite solo estos colores:
         Verde suave: #EAF4EF / #F6FBF8 / #DCEBE3     Crema: #F7F3EA / #FCFAF5 / #EDE6D6 */
    .stApp { background: linear-gradient(180deg, #EAF1F8 0%, #F6F9FC 100%); }
    [data-testid="stSidebar"] { background-color: #DCE7F3; }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3, h4, h5 { color: #1F3864; }
    [data-testid="stCaptionContainer"], .stCaption { color: #3B4756 !important; }
    [data-testid="stPlotlyChart"] {
        background-color: #FFFFFF; border-radius: 10px; padding: 8px 8px 2px 8px;
        border: 1px solid #C9D6E6; box-shadow: 0 1px 4px rgba(31,56,100,0.10);
    }
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF; border-radius: 8px; border: 1px solid #C9D6E6;
    }
    [data-testid="stExpander"] { background-color: #F6F9FC; border: 1px solid #C9D6E6; border-radius: 8px; }
    .kpi-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-top: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .kpi-card-p90 { border-top-color: #5C6BC0; background-color: #EEF2FF; }
    .kpi-card-p50 { border-top-color: #FF8A65; background-color: #FFF3E0; }
    .kpi-card-p10 { border-top-color: #2ECC71; background-color: #E8F8F5; }
    .kpi-card-ej  { border-top-color: #8E24AA; background-color: #F3E5F5; }
    .kpi-title {
        font-size: 14px;
        font-weight: 600;
        color: #3B4756;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: bold;
        color: #1A252C;
    }
    </style>
""", unsafe_allow_html=True)

# Inicialización de estados de sesión
if 'historial' not in st.session_state:
    st.session_state.historial = []
if 'df_actual' not in st.session_state:
    st.session_state.df_actual = None
# v9 (versión web): al abrir la página por primera vez se ejecuta sola la configuración documentada,
# para que quien visite el enlace vea resultados de inmediato sin tener que pulsar nada.
if 'primera_carga' not in st.session_state:
    st.session_state.primera_carga = True

# Texto opcional de autoría (se muestra bajo el título). Déjelo vacío o complételo, por ejemplo:
# AUTORIA = "Trabajo de grado — Ingeniería de Sistemas, Universidad CESMAG — Autor: Nombre Apellido"
AUTORIA = ""

# Estilo común de las figuras (v9): fondo blanco, texto oscuro y ejes legibles.
def estilo_figura(fig):
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", size=13, color="#1A252C"),
    )
    fig.update_xaxes(linecolor="#8896A8", gridcolor="#E3E8EF",
                     tickfont=dict(size=12, color="#1A252C"), title_font=dict(size=13, color="#1A252C"))
    fig.update_yaxes(linecolor="#8896A8",
                     tickfont=dict(size=12, color="#1A252C"), title_font=dict(size=13, color="#1A252C"))
    return fig

# Función auxiliar para generar 3 inputs en una sola fila (Min, Moda, Max)
def crear_inputs_triplet(titulo, key_prefix, min_def, mod_def, max_def, step_val, params_dict, fmt="%.2f"):
    st.markdown(f"**{titulo}**")
    c1, c2, c3 = st.columns(3)
    val_min = c1.number_input("Min", value=float(params_dict.get(f"{key_prefix}_min", min_def)), step=step_val, format=fmt, key=f"{key_prefix}_min")
    val_mod = c2.number_input("Moda", value=float(params_dict.get(f"{key_prefix}_mod", mod_def)), step=step_val, format=fmt, key=f"{key_prefix}_mod")
    val_max = c3.number_input("Max", value=float(params_dict.get(f"{key_prefix}_max", max_def)), step=step_val, format=fmt, key=f"{key_prefix}_max")
    return val_min, val_mod, val_max

# --- SIDEBAR: CONTROLES Y PARÁMETROS DEL MODELO (PARAMETRIZACIÓN VOLCÁN AZUFRAL) ---
with st.sidebar:
    st.header("📂 Gestión de Datos")
    archivo_json = st.file_uploader(
        "Cargar Configuración (.json)",
        type=["json"],
        help="Sube un archivo de configuración previamente guardado."
    )

    params = {}
    auto_run = False
    if archivo_json:
        try:
            params = json.load(archivo_json).get("parametros", {})
            st.success("✅ Datos Sincronizados")
            auto_run = True
        except Exception:
            st.error("❌ Error en lectura del JSON")

    st.markdown("---")
    st.header("⚙️ Parámetros de Simulación")
    nombre_sitio = st.text_input(
        "Nombre del sistema geotérmico",
        value=str(params.get("nombre_sitio", "Volcán Azufral")),
        help="Se usa en los títulos, el historial y los nombres de archivo. "
             "Los valores por defecto de los parámetros corresponden al Volcán Azufral."
    ).strip() or "Sistema geotérmico"

    OPCIONES_DIST = ["Triangular", "Normal", "Uniforme"]
    tipo_dist = st.selectbox(
        "Distribución Probabilística",
        OPCIONES_DIST,
        index=OPCIONES_DIST.index(params.get("tipo_dist")) if params.get("tipo_dist") in OPCIONES_DIST else 0,
        help="Modelado estadístico de la incertidumbre geológica del reservorio. "
             "Triangular: recomendada para rangos de opinión experta (min-moda-max), "
             "coherente con el lenguaje del modelo conceptual del SGC. Normal: útil si "
             "se asume simetría alrededor de la moda (implementada como Normal Truncada "
             "real dentro de [min, max], sin picos artificiales en los bordes). "
             "Uniforme: escenario de máxima incertidumbre."
    )

    n_iter = st.number_input(
        "Número de Iteraciones",
        min_value=1000, max_value=100000,
        value=int(params.get("iteraciones", 10000)),
        step=1000
    )

    seed = st.number_input(
        "Semilla Aleatoria (Seed)",
        min_value=1, max_value=999999,
        value=int(params.get("seed", 42))
    )

    # 1. GEOMETRÍA DEL RESERVORIO (AZUFRAL)
    st.subheader("📍 Geometría del Reservorio")
    a_min, a_mod, a_max = crear_inputs_triplet("Área del Reservorio (km²)", "A", 3.0, 4.0, 5.0, 0.1, params)
    # Mínimo 3 km²: SGC (2021, pág. 81), "un área similar a la capa sello de 3 km2"; se toma como piso porque el informe
    # dice que el reservorio probablemente se extiende lateralmente en todas direcciones (pág. 74) y que no se pueden fijar sus límites.
    # Máximo 5 km²: campo Sumikawa (Japón), análogo de arco volcánico con reservorio hidrotermal: 3-4-5 km² (Holmes y Dirkzwager, 2025, Tabla 2).
    # Moda 4 km²: promedio de mínimo y máximo, criterio de Halcon et al. (2021) para el valor más probable.
    # Ojo: Halcon et al. toman un valor único de resistividad como extensión MÁXIMA; esa lectura equivale a fijar el área en 3 km² (3.0, 3.0, 3.0).
    h_min, h_mod, h_max = crear_inputs_triplet("Espesor del Reservorio (km)", "h", 1.0, 1.5, 2.0, 0.1, params)
    # Moda 1.5 km: SGC (2021, pág. 74). Mínimo 1.0 y máximo 2.0: Halcon et al. (2021, WGC 2020+1), triangular 1000-1500-2000 m.
    # Nota: entre el techo del reservorio (1.8 km) y el tope de la fuente de calor (2.9 km) hay 1.1 km; el espesor de 1.5 km del propio SGC
    # ya lo excede, por lo que el máximo no se amplía más allá de los 2.0 km de Halcon et al. Es un supuesto declarado, no un límite geométrico.

    # 2. PROPIEDADES TÉRMICAS Y FLUIDOS (AZUFRAL)
    st.subheader("🌡️ Condiciones Térmicas y Fluidos")
    tr_min, tr_mod, tr_max = crear_inputs_triplet("Temperatura Reservorio (°C)", "Tr", 230.0, 255.0, 280.0, 1.0, params)
    # Mínimo 230 y máximo 280 °C: SGC (2021, pág. 74), rango más confiable por coincidencia de geotermómetros acuosos y de gases.
    # Moda 255 °C: promedio del rango (regla de Halcon et al., 2021).
    tf_min, tf_mod, tf_max = crear_inputs_triplet("Temperatura de Referencia (°C)", "Tf", 15.0, 15.0, 15.0, 1.0, params)
    # Tf = 15 °C constante. Fuente: convención del Circular 790 del USGS (Williams et al., 2008); Holmes y Dirkzwager (2025)
    # recomiendan tratarla como constante emparejada con el factor de recobro.

    # 3. PROPIEDADES TERMOFÍSICAS Y RECOBRO
    with st.expander("🛠️ Propiedades Termofísicas y Recobro"):
        cr_min, cr_mod, cr_max = crear_inputs_triplet("Cap. Cal. Roca (kJ/kg°C)", "Cr", 0.85, 0.95, 1.05, 0.01, params)
        cf_min, cf_mod, cf_max = crear_inputs_triplet("Cap. Cal. Fluido (kJ/kg°C)", "Cf", 4.68, 4.92, 5.29, 0.01, params)
        # Agua líquida saturada a 230, 255 y 280 °C (tablas de vapor IAPWS-IF97), coherente con el rango de Tr.
        rhor_min, rhor_mod, rhor_max = crear_inputs_triplet("Densidad Roca (kg/m³)", "rhor", 2450.0, 2550.0, 2650.0, 10.0, params)
        rhof_min, rhof_mod, rhof_max = crear_inputs_triplet("Densidad Fluido (kg/m³)", "rhof", 750.0, 791.0, 827.0, 10.0, params)
        # Agua líquida saturada a 280, 255 y 230 °C (tablas de vapor IAPWS-IF97), coherente con el rango de Tr.
        phi_min, phi_mod, phi_max = crear_inputs_triplet("Porosidad (Fracción)", "phi", 0.03, 0.05, 0.08, 0.005, params, fmt="%.3f")
        rf_min, rf_mod, rf_max = crear_inputs_triplet("Factor de Recobro (Rf)", "Rf", 0.08, 0.14, 0.20, 0.01, params)
        # Rango 0.08-0.20: Williams et al. (2008, USGS OFR 2008-1296), reservorios dominados por fracturas, con T0 = 15 °C.
        # Moda 0.14: promedio del rango (el USGS usa distribución uniforme). Alternativa con respaldo: 0.03-0.10-0.17 (Sanyal et al.).
        eta_min, eta_mod, eta_max = crear_inputs_triplet("Eficiencia Planta (η)", "eta", 0.075, 0.085, 0.098, 0.005, params, fmt="%.3f")
        # Relación eficiencia-entalpía (Zarrouk y Moon, 2014, según Halcon et al., 2021; Ciriaco et al., 2020, según Holmes y
        # Dirkzwager, 2025): 7.5 %, 8.5 % y 9.8 % a 200, 225 y 260 °C. Rango clásico del USGS (1975): 8-12 %.
        pf_min, pf_mod, pf_max = crear_inputs_triplet("Factor de Planta (Pf)", "Pf", 0.85, 0.90, 0.95, 0.01, params)

    # PARÁMETRO FIJO: VIDA ÚTIL
    st.subheader("⏱️ Operación")
    vida_util = st.number_input("Vida Útil Planta (Años)", min_value=1, max_value=100, value=int(params.get("vida_util", 30)))

    # -------------------------------------------------------------------
    # NUEVO (Vacío 5): MÉTODO DE VERIFICACIÓN INDEPENDIENTE
    # -------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧪 Método de Verificación Independiente")
    st.caption(
        "Estimación cruzada por densidad de potencia (MWe/km²), típica de "
        "estudios de reconocimiento de sistemas volcánicos de alta entalpía "
        "análogos, para contrastar el resultado del método volumétrico."
    )
    usar_metodo2 = st.checkbox(
        "Ejecutar método de verificación (densidad de potencia)",
        value=bool(params.get("usar_metodo2", True))
    )
    dp_min, dp_mod, dp_max = crear_inputs_triplet(
        "Densidad de Potencia (MWe/km²)", "Dp", 5.0, 10.0, 15.0, 0.5, params
    )

    # -------------------------------------------------------------------
    # ANTECEDENTES (opcional): comparación con estimaciones previas
    # -------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📚 Antecedentes (opcional)")
    mostrar_antecedentes = st.checkbox(
        "Mostrar comparación con estimaciones previas",
        value=bool(params.get("mostrar_antecedentes", False)),
        help="Muestra una tabla de referencia con estudios previos del mismo sistema. "
             "La lista se edita en ANTECEDENTES_POR_DEFECTO (código) o en la clave "
             "'antecedentes' del JSON. Para otro sistema geotérmico, reemplazarla."
    )
    antecedentes = params.get("antecedentes", ANTECEDENTES_POR_DEFECTO)

    # VALIDACIÓN DE PARÁMETROS DE ENTRADA
    errores_validacion = []
    variables_triplet = [
        ("Área", a_min, a_mod, a_max),
        ("Espesor", h_min, h_mod, h_max),
        ("Temp. Reservorio", tr_min, tr_mod, tr_max),
        ("Temp. Referencia", tf_min, tf_mod, tf_max),
        ("Cap. Cal. Roca", cr_min, cr_mod, cr_max),
        ("Cap. Cal. Fluido", cf_min, cf_mod, cf_max),
        ("Densidad Roca", rhor_min, rhor_mod, rhor_max),
        ("Densidad Fluido", rhof_min, rhof_mod, rhof_max),
        ("Porosidad", phi_min, phi_mod, phi_max),
        ("Factor Recobro", rf_min, rf_mod, rf_max),
        ("Eficiencia", eta_min, eta_mod, eta_max),
        ("Factor de Planta", pf_min, pf_mod, pf_max),
        ("Densidad de Potencia", dp_min, dp_mod, dp_max),
    ]

    for nombre, v_min, v_mod, v_max in variables_triplet:
        if v_min > v_max:
            errores_validacion.append(f"{nombre}: El Mínimo no puede ser mayor que el Máximo.")
        if tipo_dist == "Triangular" and not (v_min <= v_mod <= v_max) and not (v_min == v_max):
            errores_validacion.append(f"{nombre}: La Moda debe estar entre el Mínimo y el Máximo.")

    if tr_min <= tf_max:
        errores_validacion.append("La Temperatura del Reservorio Mínima debe superar la Temp. de Referencia Máxima.")

    if errores_validacion:
        st.error("⚠️ **Inconsistencia en los Parámetros:**")
        for err in errores_validacion:
            st.write(f"- {err}")
        st.stop()

    # EXPORTAR CONFIGURACIÓN JSON COMPLETA
    config_a_descargar = {
        "parametros": {
            "nombre_sitio": nombre_sitio,
            "tipo_dist": tipo_dist,
            "iteraciones": n_iter, "seed": seed, "vida_util": vida_util,
            "A_min": a_min, "A_mod": a_mod, "A_max": a_max,
            "h_min": h_min, "h_mod": h_mod, "h_max": h_max,
            "Tr_min": tr_min, "Tr_mod": tr_mod, "Tr_max": tr_max,
            "Tf_min": tf_min, "Tf_mod": tf_mod, "Tf_max": tf_max,
            "Cr_min": cr_min, "Cr_mod": cr_mod, "Cr_max": cr_max,
            "Cf_min": cf_min, "Cf_mod": cf_mod, "Cf_max": cf_max,
            "rhor_min": rhor_min, "rhor_mod": rhor_mod, "rhor_max": rhor_max,
            "rhof_min": rhof_min, "rhof_mod": rhof_mod, "rhof_max": rhof_max,
            "phi_min": phi_min, "phi_mod": phi_mod, "phi_max": phi_max,
            "Rf_min": rf_min, "Rf_mod": rf_mod, "Rf_max": rf_max,
            "eta_min": eta_min, "eta_mod": eta_mod, "eta_max": eta_max,
            "Pf_min": pf_min, "Pf_mod": pf_mod, "Pf_max": pf_max,
            # --- nuevos parámetros ---
            "usar_metodo2": usar_metodo2,
            "Dp_min": dp_min, "Dp_mod": dp_mod, "Dp_max": dp_max,
            "mostrar_antecedentes": mostrar_antecedentes,
            "antecedentes": antecedentes,
        }
    }
    st.download_button(
        label="📥 Descargar Configuración (.json)",
        data=json.dumps(config_a_descargar, indent=4),
        file_name=f"config_{nombre_archivo(nombre_sitio)}_geotermia.json",
        mime="application/json",
        use_container_width=True
    )

    st.markdown("---")
    ejecutar = st.button("🚀 INICIAR SIMULACIÓN", use_container_width=True)


# =============================================================================
# --- MOTOR DE SIMULACIÓN MONTE CARLO ---
# =============================================================================
def generar_vector(dist_type, v_min, v_mod, v_max, n_samples):
    # --- CORRECCIÓN (bug detectado en prueba de validación con Paucarani) ---
    # La versión original solo comparaba v_min == v_max para decidir si la
    # variable era constante, y luego recortaba la moda con np.clip(v_mod,
    # v_min, v_max). Eso es correcto cuando los tres valores del triplete
    # son iguales (p. ej. Densidad del Fluido = 926/926/926), pero falla
    # cuando min y max coinciden entre sí pero la moda es distinta (p. ej.
    # Rf = 0.10/0.15/0.10 o Pf = 0.90/0.95/0.90, como en el caso de
    # Paucarani): la variable se trataba como constante, y aunque no lo
    # fuera, el recorte posterior forzaba la moda de vuelta a ese mismo
    # valor, descartando en silencio la moda ingresada por el usuario.
    # Se corrige en dos pasos: (1) solo se trata como constante si los TRES
    # valores del triplete coinciden; y (2) en vez de recortar la moda hacia
    # un rango que podría no contenerla, se amplía el límite que corresponda
    # (mínimo o máximo) para que el rango sí incluya la moda, preservando así
    # el valor que el usuario realmente ingresó. Para cualquier triplete ya
    # bien formado (mín <= moda <= máx, como en todas las variables de
    # Azufral), esta ampliación no cambia nada: el mínimo y el máximo quedan
    # exactamente igual que antes.
    if v_min == v_max == v_mod:
        return np.full(n_samples, v_min)

    v_min = min(v_min, v_mod)
    v_max = max(v_max, v_mod)

    if dist_type == "Triangular":
        return np.random.triangular(v_min, v_mod, v_max, n_samples)
    elif dist_type == "Normal":
        # --- CORRECCIÓN (hallazgo adicional, distribución Normal) ---
        # La versión original generaba una normal completa y le aplicaba
        # np.clip(): esto no es una normal truncada, sino un "aplastamiento"
        # que apila artificialmente probabilidad exactamente en los límites
        # min/max (un pico ficticio que no existe en la distribución real).
        # Se reemplaza por scipy.stats.truncnorm, que sí redistribuye la
        # masa de probabilidad correctamente dentro de [min, max], de modo
        # que el área bajo la curva truncada integre a 1 sin picos falsos
        # en los bordes.
        std = (v_max - v_min) / 6 if v_max > v_min else 0.0001
        a = (v_min - v_mod) / std
        b = (v_max - v_mod) / std
        return stats.truncnorm.rvs(a, b, loc=v_mod, scale=std, size=n_samples)
    else:  # Uniforme
        return np.random.uniform(v_min, v_max, n_samples)


# -----------------------------------------------------------------------
# NUEVO (Vacío 7, parcial): funciones puras de cálculo físico, separadas
# de la lógica reactiva de Streamlit para que sean testeables de forma
# independiente (p. ej. con pytest) sin depender de session_state ni de
# la interfaz. No se reestructura el resto del archivo: siguen viviendo
# en el mismo script y se invocan igual que antes.
# -----------------------------------------------------------------------
def calcular_potencia_y_calor(A, h, Tr, Tf, Cr, Cf, rho_r, rho_f, Phi, Rf, eta, Pf, vida_util):
    """Ecuación volumétrica USGS (heat-in-place). Retorna potencia (MWe),
    calor almacenado total (EJ) y calor recuperable (EJ)."""
    volumen_m3 = (A * 1e6) * (h * 1000)
    rho_c_equiv = (1 - Phi) * (rho_r * Cr) + Phi * (rho_f * Cf)
    q_termico_kj = volumen_m3 * rho_c_equiv * (Tr - Tf)

    segundos_vida_util = vida_util * 365.25 * 24 * 3600
    # El factor de planta (Pf) DIVIDE: la capacidad instalada es la energía eléctrica total entre (vida útil × factor de planta).
    # Es la forma de Halcon et al. (2021, Ec. 2), Holmes y Dirkzwager (2025, Ecs. 2 y 4) y Flores et al. (2018).
    # Las versiones v1 a v6 multiplicaban por Pf, lo que subestimaba la potencia en 1/Pf² (≈ 11 % con Pf = 0.95; ≈ 23 % con 0.90).
    potencia_mw = (q_termico_kj * Rf * eta / Pf) / segundos_vida_util / 1000

    # 1 EJ = 1e18 J = 1e15 kJ  ->  EJ = kJ * 1e-15
    calor_almacenado_ej = q_termico_kj * 1e-15
    calor_recuperable_ej = calor_almacenado_ej * Rf

    return potencia_mw, calor_almacenado_ej, calor_recuperable_ej


if ejecutar or auto_run or st.session_state.primera_carga:
    st.session_state.primera_carga = False
    np.random.seed(seed)

    # Muestreo estocástico de las variables (distribuciones marginales)
    A = generar_vector(tipo_dist, a_min, a_mod, a_max, n_iter)
    h = generar_vector(tipo_dist, h_min, h_mod, h_max, n_iter)
    Tr = generar_vector(tipo_dist, tr_min, tr_mod, tr_max, n_iter)
    Tf = generar_vector(tipo_dist, tf_min, tf_mod, tf_max, n_iter)
    Cr = generar_vector(tipo_dist, cr_min, cr_mod, cr_max, n_iter)
    Cf = generar_vector(tipo_dist, cf_min, cf_mod, cf_max, n_iter)
    rho_r = generar_vector(tipo_dist, rhor_min, rhor_mod, rhor_max, n_iter)
    rho_f = generar_vector(tipo_dist, rhof_min, rhof_mod, rhof_max, n_iter)
    Phi = generar_vector(tipo_dist, phi_min, phi_mod, phi_max, n_iter)
    Rf = generar_vector(tipo_dist, rf_min, rf_mod, rf_max, n_iter)
    eta = generar_vector(tipo_dist, eta_min, eta_mod, eta_max, n_iter)
    Pf = generar_vector(tipo_dist, pf_min, pf_mod, pf_max, n_iter)

    potencia_mw, calor_almacenado_ej, calor_recuperable_ej = calcular_potencia_y_calor(
        A, h, Tr, Tf, Cr, Cf, rho_r, rho_f, Phi, Rf, eta, Pf, vida_util
    )

    df = pd.DataFrame({
        'ID_Simulacion': np.arange(1, n_iter + 1),
        'Area_km2': np.round(A, 2),
        'Espesor_km': np.round(h, 2),
        'Temp_Reservorio_C': np.round(Tr, 2),
        'Temp_Rechazo_C': np.round(Tf, 2),
        'CapCal_Roca_kJ_kgC': np.round(Cr, 3),
        'CapCal_Fluido_kJ_kgC': np.round(Cf, 3),
        'Densidad_Roca_kg_m3': np.round(rho_r, 1),
        'Densidad_Fluido_kg_m3': np.round(rho_f, 1),
        'Porosidad_Fraccion': np.round(Phi, 3),
        'Factor_Recobro_Rf': np.round(Rf, 3),
        'Eficiencia_Planta_eta': np.round(eta, 3),
        'Factor_Planta_Pf': np.round(Pf, 3),
        'Potencia_MW': np.round(potencia_mw, 2),
        # --- NUEVO (Vacío 2): calor almacenado como salida explícita ---
        'Calor_Almacenado_EJ': np.round(calor_almacenado_ej, 4),
        'Calor_Recuperable_EJ': np.round(calor_recuperable_ej, 4),
    })

    # --- NUEVO (Vacío 5): método de verificación independiente ---
    if usar_metodo2:
        Dp = generar_vector(tipo_dist, dp_min, dp_mod, dp_max, n_iter)
        # Se reutiliza el mismo vector de Área ya muestreado (comparación
        # directa, controlando por la misma incertidumbre geométrica).
        potencia_mw_metodo2 = A * Dp
        df['Densidad_Potencia_MWe_km2'] = np.round(Dp, 2)
        df['Potencia_MW_Metodo2'] = np.round(potencia_mw_metodo2, 2)

    st.session_state.df_actual = df
    st.session_state.n_iter_actual = n_iter

    p9, p5, p1 = np.percentile(potencia_mw, [10, 50, 90])
    st.session_state.historial.append({
        "Escenario": f"{nombre_sitio} - {tipo_dist} ({n_iter}) - Seed: {seed}",
        "P90": round(p9, 2),
        "P50": round(p5, 2),
        "P10": round(p1, 2)
    })


# =============================================================================
# --- DASHBOARD Y RESULTADOS ---
# =============================================================================
if st.session_state.df_actual is not None:
    df = st.session_state.df_actual
    n_iter_actual = st.session_state.get("n_iter_actual", len(df))
    potencia = df['Potencia_MW']
    calor_ej = df['Calor_Almacenado_EJ']

    p90, p50, p10 = np.percentile(potencia, [10, 50, 90])
    p90_ej, p50_ej, p10_ej = np.percentile(calor_ej, [10, 50, 90])

    st.title(f"🌋 Simulación Monte Carlo - Potencial Geotérmico ({nombre_sitio})")
    st.caption("Evaluación probabilística mediante el Método Volumétrico del USGS")
    if AUTORIA:
        st.caption(AUTORIA)

    with st.expander("📖 Guía rápida para el jurado (clic para abrir)"):
        st.markdown(
            """
**Qué muestra esta página.** Una estimación preliminar del potencial geotérmico del volcán Azufral. En lugar de un único número,
la simulación Monte Carlo repite el cálculo 10.000 veces con valores al azar dentro de los rangos documentados de cada parámetro
y entrega una **distribución de resultados posibles**.

**Cómo leer los resultados.**
- **P90, P50 y P10** son valores de potencia (MWe): P90 es el valor que se supera con 90 % de probabilidad (estimación conservadora),
  P50 es la mediana y P10 es el que se supera con solo 10 % (estimación optimista).
- El **histograma** muestra cuántas simulaciones cayeron en cada intervalo de potencia; en naranja, el 80 % central.
- La **curva de excedencia** indica la probabilidad de superar cualquier valor de potencia.
- El **análisis de convergencia** comprueba que 10.000 iteraciones son suficientes.
- El **tornado** muestra qué parámetros influyen más en la potencia.

**Qué puede probar.** En el panel izquierdo puede cambiar la distribución, la semilla o cualquier rango (mínimo, moda y máximo)
y pulsar **INICIAR SIMULACIÓN**. También puede descargar la configuración y los resultados.

**Importante.** Los resultados dependen de los rangos de entrada documentados en el trabajo; son una estimación de superficie,
sin validación con datos de campo.
"""
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # TARJETAS KPI (P90, P50, P10) — Potencia eléctrica
    col_p90, col_p50, col_p10 = st.columns(3)

    with col_p90:
        st.markdown(f"""
            <div class="kpi-card kpi-card-p90">
                <div class="kpi-title">P90 (Conservador)</div>
                <div class="kpi-value">{p90:.2f} MWe</div>
            </div>
        """, unsafe_allow_html=True)

    with col_p50:
        st.markdown(f"""
            <div class="kpi-card kpi-card-p50">
                <div class="kpi-title">P50 (Mediana)</div>
                <div class="kpi-value">{p50:.2f} MWe</div>
            </div>
        """, unsafe_allow_html=True)

    with col_p10:
        st.markdown(f"""
            <div class="kpi-card kpi-card-p10">
                <div class="kpi-title">P10 (Optimista)</div>
                <div class="kpi-value">{p10:.2f} MWe</div>
            </div>
        """, unsafe_allow_html=True)

    # --- NUEVO (Vacío 2): TARJETAS KPI — Calor almacenado en EJ ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🔥 Calor Almacenado (EJ)")
    st.caption("Energía térmica total contenida en el reservorio por encima de la temperatura de referencia (1 EJ = 10¹⁸ J), antes de aplicar recobro y eficiencia.")
    col_ej90, col_ej50, col_ej10 = st.columns(3)
    with col_ej90:
        st.markdown(f"""
            <div class="kpi-card kpi-card-ej">
                <div class="kpi-title">P90 Calor Almacenado</div>
                <div class="kpi-value">{p90_ej:.3f} EJ</div>
            </div>
        """, unsafe_allow_html=True)
    with col_ej50:
        st.markdown(f"""
            <div class="kpi-card kpi-card-ej">
                <div class="kpi-title">P50 Calor Almacenado</div>
                <div class="kpi-value">{p50_ej:.3f} EJ</div>
            </div>
        """, unsafe_allow_html=True)
    with col_ej10:
        st.markdown(f"""
            <div class="kpi-card kpi-card-ej">
                <div class="kpi-title">P10 Calor Almacenado</div>
                <div class="kpi-value">{p10_ej:.3f} EJ</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # NUEVO (Crystal Ball): CALCULADORA DE CERTEZA (rango de probabilidad
    # personalizado) — réplica del recurso "Certainty" de Crystal Ball, que
    # en Paucarani se ve como "Certeza Baja/Alta" sobre el histograma. A
    # diferencia de P90/P50/P10 (fijos), aquí el usuario define cualquier
    # rango [Xmin, Xmax] de potencia y el modelo calcula la probabilidad de
    # que el resultado real caiga dentro de ese rango.
    # =========================================================================
    st.markdown("##### 🎯 Calculadora de Certeza (rango personalizado)")
    st.caption(
        "Equivalente al recurso 'Certainty' de Crystal Ball: define cualquier "
        "rango de potencia y obtén la probabilidad asociada, en vez de limitarte "
        "a los percentiles fijos P90/P50/P10. Útil, por ejemplo, para responder "
        "'¿qué probabilidad hay de superar el mínimo viable de planta?'."
    )
    col_cert1, col_cert2, col_cert3 = st.columns([1, 1, 1])
    pot_min_data = float(np.min(potencia))
    pot_max_data = float(np.max(potencia))
    with col_cert1:
        umbral_min = st.number_input(
            "Límite inferior (MWe)", min_value=pot_min_data, max_value=pot_max_data,
            value=float(round(p90, 2)), step=1.0, key="cert_min"
        )
    with col_cert2:
        umbral_max = st.number_input(
            "Límite superior (MWe)", min_value=pot_min_data, max_value=pot_max_data,
            value=float(round(p10, 2)), step=1.0, key="cert_max"
        )
    with col_cert3:
        if umbral_min <= umbral_max:
            certeza_pct = float(np.mean((potencia >= umbral_min) & (potencia <= umbral_max)) * 100)
            st.markdown(f"""
                <div class="kpi-card" style="border-top-color:#26A69A; background-color:#E0F2F1;">
                    <div class="kpi-title">Probabilidad en el rango</div>
                    <div class="kpi-value">{certeza_pct:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("El límite inferior debe ser menor o igual al superior.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- HISTOGRAMA PROBABILÍSTICO CON DOBLE EJE Y ---
    n_bins = 50
    counts, bin_edges = np.histogram(potencia, bins=n_bins)

    probabilidad_relativa = counts / n_iter_actual

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    colors = [
        '#FF8A65' if (bin_edges[i] >= p90 and bin_edges[i+1] <= p10) else '#5C6BC0'
        for i in range(len(counts))
    ]

    max_count = max(counts)
    max_y2 = int(np.ceil((max_count + 10) / 50.0) * 50)
    max_y1 = max_y2 / n_iter_actual
    FACTOR_Y = 1.45      # franja libre sobre las barras, para que las etiquetas no se solapen con ellas
    ALTURA_ETIQ = 1.12   # altura de las etiquetas (como múltiplo de la altura de las barras)

    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        x=bin_centers,
        y=probabilidad_relativa,
        width=bin_width,
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=0.4,
        name="Probabilidad",
        hovertext=[f"Rango: {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f} MWe<br>Frecuencia: {counts[i]}<br>Probabilidad: {probabilidad_relativa[i]*100:.2f}%" for i in range(len(counts))],
        hoverinfo='text'
    ))

    fig1.add_trace(go.Scatter(
        x=bin_centers,
        y=counts,
        mode='markers',
        marker=dict(opacity=0),
        yaxis='y2',
        showlegend=False,
        hoverinfo='skip'
    ))

    y_etq = max_y1 * ALTURA_ETIQ
    for valor_p, rotulo_p, color_p, grosor_p, estilo_p in [
        (p90, "P90", "#1E88E5", 2, "dash"),
        (p50, "P50 (mediana)", "#1A252C", 3, "solid"),
        (p10, "P10", "#1E88E5", 2, "dash"),
    ]:
        fig1.add_shape(type="line", x0=valor_p, x1=valor_p, y0=0, y1=y_etq, xref="x", yref="y",
                       line=dict(color=color_p, width=grosor_p, dash=estilo_p))
        fig1.add_annotation(x=valor_p, y=y_etq, xref="x", yref="y",
                            text=f"<b>{rotulo_p}</b><br>{valor_p:.2f} MWe", showarrow=False,
                            yanchor="bottom", bgcolor="#FFFFFF", bordercolor=color_p, borderwidth=1.5,
                            borderpad=4, font=dict(size=13, color=color_p))

    fig1.update_layout(
        title={
            'text': f"<b>Distribución de Potencial Eléctrico Generable ({nombre_sitio})</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color='#1A252C')
        },
        xaxis=dict(
            title="Potencia Eléctrica Estimada (MWe)",
            tickformat=".2f",
            showgrid=True,
            gridcolor='#EAEAEA'
        ),
        yaxis=dict(
            title="<b>Probabilidad</b>",
            side="left",
            range=[0, max_y1 * FACTOR_Y],
            tickformat=".2f",
            showgrid=True,
            gridcolor='#EAEAEA'
        ),
        yaxis2=dict(
            title="<b>Frecuencia (# Iteraciones)</b>",
            side="right",
            overlaying="y",
            range=[0, max_y2 * FACTOR_Y],
            showgrid=False
        ),
        template="plotly_white",
        height=540,
        margin=dict(l=60, r=60, t=60, b=50),
        bargap=0.02,
        showlegend=False
    )
    estilo_figura(fig1)

    # --- SECCIÓN DE HISTOGRAMA Y TABLA DE RESULTADOS OBTENIDOS ---
    col_fig, col_stats = st.columns([2, 1])

    with col_fig:
        st.plotly_chart(
            fig1,
            use_container_width=True,
            theme=None,
            config={
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': f'histograma_potencial_{nombre_archivo(nombre_sitio)}',
                    'height': 600,
                    'width': 850,
                    'scale': 1
                }
            }
        )

        st.caption(
            "Naranja: 80 % central de los resultados (entre P90 y P10). Azul: colas, con 10 % a cada lado. "
            "Línea oscura: P50 (mediana)."
        )

    with col_stats:
        # CÁLCULO DE CASO BASE DETERMINÍSTICO (USANDO LAS MODAS INTRODUCIDAS)
        caso_base_val, caso_base_ej, _ = calcular_potencia_y_calor(
            a_mod, h_mod, tr_mod, tf_mod, cr_mod, cf_mod,
            rhor_mod, rhof_mod, phi_mod, rf_mod, eta_mod, pf_mod, vida_util
        )

        # CÁLCULO DE ESTADÍSTICOS DESCRIPTIVOS COMPLETOS
        media_val = np.mean(potencia)
        mediana_val = np.median(potencia)
        std_val = np.std(potencia, ddof=1)
        varianza_val = np.var(potencia, ddof=1)
        sesgo_val = stats.skew(potencia)
        curtosis_val = stats.kurtosis(potencia, fisher=False)
        cv_val = std_val / media_val if media_val != 0 else 0
        min_val = np.min(potencia)
        max_val = np.max(potencia)
        ancho_rango_val = max_val - min_val
        error_std_medio_val = std_val / np.sqrt(len(potencia))

        df_res_obtenidos = pd.DataFrame({
            'Resultados Obtenidos': [
                'Pruebas',
                'Caso base',
                'Media',
                'Mediana',
                'Modo',
                'Desviación estándar',
                'Varianza',
                'Sesgo',
                'Curtosis',
                'Coeficiente de variación',
                'Mínimo',
                'Máximo',
                'Ancho de rango',
                'Error estándar medio'
            ],
            'Valor': [
                f"{len(potencia):,}",
                f"{caso_base_val:.4f}",
                f"{media_val:.4f}",
                f"{mediana_val:.4f}",
                "---",
                f"{std_val:.4f}",
                f"{varianza_val:.4f}",
                f"{sesgo_val:.4f}",
                f"{curtosis_val:.2f}",
                f"{cv_val:.4f}",
                f"{min_val:.4f}",
                f"{max_val:.4f}",
                f"{ancho_rango_val:.4f}",
                f"{error_std_medio_val:.4f}"
            ]
        })

        st.markdown("#### 📋 Resultados Obtenidos")
        st.dataframe(df_res_obtenidos, hide_index=True, use_container_width=True, height=500)

    st.markdown("---")

    # =========================================================================
    # NUEVO (Vacío 3): CURVA DE PROBABILIDAD DE EXCEDENCIA (S-CURVE)
    # =========================================================================
    st.subheader("📈 Curva de Probabilidad de Excedencia")
    st.caption(
        "Complementa al histograma: para cada valor de potencia en el eje Y, "
        "el eje X indica la probabilidad de que el potencial real del sistema "
        "sea igual o superior a dicho valor. Es la forma estándar de reportar "
        "reservas/recursos bajo incertidumbre (convención P90–P50–P10)."
    )
    potencia_ordenada = np.sort(potencia.values)
    n = len(potencia_ordenada)
    rangos = np.arange(1, n + 1)
    # Probabilidad de que X sea MAYOR O IGUAL al valor ordenado (excedencia)
    prob_excedencia = 100 * (1 - (rangos - 0.5) / n)

    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(
        x=prob_excedencia,
        y=potencia_ordenada,
        mode='lines',
        line=dict(color="#1E88E5", width=3),
        name="Curva de excedencia",
        hovertemplate="Probabilidad de excedencia: %{x:.1f}%%<br>Potencia: %{y:.2f} MWe<extra></extra>"
    ))
    for valor, etiqueta, prob_x, color in [
        (p90, "P90", 90, "#5C6BC0"),
        (p50, "P50", 50, "#FF8A65"),
        (p10, "P10", 10, "#2ECC71"),
    ]:
        fig_cdf.add_trace(go.Scatter(
            x=[prob_x], y=[valor], mode='markers',
            marker=dict(size=12, color=color, line=dict(color="white", width=1.5)),
            showlegend=False,
            hoverinfo='skip'
        ))
        # La etiqueta va DEBAJO de la curva (zona libre) con una guía corta hacia su punto.
        fig_cdf.add_annotation(
            x=prob_x, y=valor, text=f"<b>{etiqueta}</b><br>{valor:.2f} MWe",
            showarrow=True, arrowhead=0, arrowwidth=1.3, arrowcolor=color, ax=0, ay=58,
            bgcolor="#FFFFFF", bordercolor=color, borderwidth=1.5, borderpad=4,
            font=dict(size=13, color="#1A252C")
        )

    fig_cdf.update_layout(
        title=dict(text="<b>Curva de probabilidad de excedencia</b>", x=0.5, xanchor="center", font=dict(size=17, color="#1F3864")),
        xaxis=dict(title="Probabilidad de excedencia (%)", autorange="reversed", range=[100, 0]),
        yaxis=dict(title="Potencia Eléctrica Estimada (MWe)", range=[0, float(potencia_ordenada.max()) * 1.08]),
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=60, t=60, b=50),
        showlegend=False
    )
    estilo_figura(fig_cdf)
    st.plotly_chart(fig_cdf, use_container_width=True, theme=None)

    st.markdown("---")

    # =========================================================================
    # NUEVO (Vacío 4): ANÁLISIS DE CONVERGENCIA DE LA SIMULACIÓN
    # =========================================================================
    st.subheader("🔬 Análisis de Convergencia")
    st.caption(
        "Muestra cómo se estabilizan la media y los percentiles P90/P50/P10 a "
        "medida que aumenta el número de iteraciones utilizadas (calculados "
        "sobre los primeros k valores generados). Si las líneas aún oscilan "
        "de forma apreciable cerca de N, el número de iteraciones configurado "
        "podría ser insuficiente para garantizar resultados estables."
    )
    checkpoints = np.unique(np.geomspace(max(50, int(n_iter_actual * 0.02)), n_iter_actual, 25).astype(int))
    valores = potencia.values
    conv_media = [np.mean(valores[:k]) for k in checkpoints]
    conv_p90 = [np.percentile(valores[:k], 10) for k in checkpoints]
    conv_p50 = [np.percentile(valores[:k], 50) for k in checkpoints]
    conv_p10 = [np.percentile(valores[:k], 90) for k in checkpoints]

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=checkpoints, y=conv_media, mode='lines', name='Media', line=dict(color="#1A252C", width=2, dash='dot')))
    fig_conv.add_trace(go.Scatter(x=checkpoints, y=conv_p90, mode='lines', name='P90', line=dict(color="#5C6BC0", width=2)))
    fig_conv.add_trace(go.Scatter(x=checkpoints, y=conv_p50, mode='lines', name='P50', line=dict(color="#FF8A65", width=2)))
    fig_conv.add_trace(go.Scatter(x=checkpoints, y=conv_p10, mode='lines', name='P10', line=dict(color="#2ECC71", width=2)))
    fig_conv.update_layout(
        xaxis=dict(title="Número de iteraciones acumuladas", type="log"),
        yaxis=dict(title="Potencia Eléctrica (MWe)"),
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=20, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    estilo_figura(fig_conv)
    st.plotly_chart(fig_conv, use_container_width=True, theme=None)

    st.markdown("---")

    # SENSIBILIDAD SPEARMAN COMPLETA (TORNADO CON TODAS LAS VARIABLES)
    st.subheader("📊 Análisis Complementario y Exportación")
    col_g2, col_exp = st.columns([2, 1])

    with col_g2:
        st.markdown("#### 🌪️ Sensibilidad de Parámetros (Tornado)")

        # --- NUEVO (Crystal Ball): vista de sensibilidad seleccionable ---
        # Crystal Ball (y por tanto el estudio del volcán Paucarani, hecho con
        # esa herramienta) no muestra el coeficiente de Spearman crudo, sino
        # la vista "Contribution to Variance": el cuadrado de cada coeficiente
        # de correlación de rango, normalizado para que la suma de todas las
        # variables sea 100%, conservando el signo original para indicar
        # dirección. Se ofrece como alternativa para comparar directamente
        # con el formato de Paucarani, sin perder la vista estadística cruda.
        if MOSTRAR_VISTA_CONTRIBUCION:
            vista_sensibilidad = st.radio(
                "Vista de sensibilidad",
                ["Correlación de Spearman", "Contribución a la Varianza (%)"],
                horizontal=True,
                help="Contribución a la Varianza = signo(ρ) × ρ² / Σρ² × 100."
            )
        else:
            vista_sensibilidad = "Correlación de Spearman"
        st.caption(
            "Cada barra es el coeficiente de correlación de Spearman (ρ) entre una variable de "
            "entrada y la potencia resultante, calculado sobre todas las iteraciones. "
            "ρ cercano a +1: cuando la variable sube, la potencia sube. "
            "ρ cercano a 0: la variable casi no influye. "
            "ρ negativo: cuando la variable sube, la potencia baja."
        )

        var_map = {
            'Area_km2': 'Área (A)',
            'Espesor_km': 'Espesor (h)',
            'Temp_Reservorio_C': 'Temp. Reservorio (Tr)',
            'Temp_Rechazo_C': 'Temp. Referencia (Tf)',
            'CapCal_Roca_kJ_kgC': 'Cap. Cal. Roca (Cr)',
            'CapCal_Fluido_kJ_kgC': 'Cap. Cal. Fluido (Cf)',
            'Densidad_Roca_kg_m3': 'Densidad Roca (ρr)',
            'Densidad_Fluido_kg_m3': 'Densidad Fluido (ρf)',
            'Porosidad_Fraccion': 'Porosidad (Φ)',
            'Factor_Recobro_Rf': 'Factor Recobro (Rf)',
            'Eficiencia_Planta_eta': 'Eficiencia Planta (η)',
            'Factor_Planta_Pf': 'Factor Planta (Pf)'
        }

        variables = list(var_map.keys())

        # --- NUEVO (Vacío 6): correlación de Spearman CON p-valor ---
        correlaciones, p_valores = [], []
        for v in variables:
            try:
                r, p_val = stats.spearmanr(df[v], df['Potencia_MW'])
                if np.isnan(r):
                    r, p_val = 0.0, 1.0
            except Exception:
                r, p_val = 0.0, 1.0
            correlaciones.append(r)
            p_valores.append(p_val)

        df_corr = pd.DataFrame({
            'Variable': [var_map[v] for v in variables],
            'Correlacion': correlaciones,
            'p_valor': p_valores,
        })
        df_corr['Significativo (p<0.05)'] = df_corr['p_valor'] < 0.05

        # --- NUEVO (Crystal Ball): cálculo de Contribución a la Varianza ---
        suma_r2 = np.sum(np.square(df_corr['Correlacion']))
        if suma_r2 > 0:
            df_corr['Contribucion_%'] = (
                np.sign(df_corr['Correlacion']) * np.square(df_corr['Correlacion']) / suma_r2 * 100
            )
        else:
            df_corr['Contribucion_%'] = 0.0

        usar_contribucion = MOSTRAR_VISTA_CONTRIBUCION and (vista_sensibilidad == "Contribución a la Varianza (%)")
        columna_x = 'Contribucion_%' if usar_contribucion else 'Correlacion'
        titulo_eje_x = "Contribución a la Varianza (%)" if usar_contribucion else "Coeficiente de correlación de Spearman (ρ)"
        formato_val = "{:+.1f}%" if usar_contribucion else "{:+.2f}"

        df_corr = df_corr.sort_values(by=columna_x, ascending=True)
        valores_x = df_corr[columna_x]
        margen = 0.10 * max(abs(valores_x.max()), abs(valores_x.min()), 1e-9)

        if usar_contribucion:
            textos_hover = [
                f"<b>{var}</b><br>Spearman ρ = {r:.3f}<br>Contribución = {c:.1f}%<br>p-valor = {pv:.2e}"
                for var, r, c, pv in zip(df_corr['Variable'], df_corr['Correlacion'], df_corr['Contribucion_%'], df_corr['p_valor'])
            ]
        else:
            textos_hover = [
                f"<b>{var}</b><br>Spearman ρ = {r:.3f}<br>p-valor = {pv:.2e}"
                for var, r, pv in zip(df_corr['Variable'], df_corr['Correlacion'], df_corr['p_valor'])
            ]

        fig_torn = go.Figure()
        fig_torn.add_trace(go.Bar(
            y=df_corr['Variable'],
            x=valores_x,
            orientation='h',
            text=[formato_val.format(v) for v in valores_x],
            textposition='outside',      # etiqueta fuera de la barra: legible aun en barras muy cortas
            cliponaxis=False,
            marker=dict(
                color=valores_x,
                colorscale='Viridis',
                line=dict(color='black', width=0.8)
            ),
            hoverinfo='text',
            hovertext=textos_hover
        ))

        fig_torn.update_layout(
            title="Sensibilidad de la potencia (MWe) a cada variable de entrada",
            xaxis_title=titulo_eje_x,
            xaxis=dict(
                range=[min(0, valores_x.min()) - 4 * margen, max(0, valores_x.max()) + 4 * margen],
                zeroline=True, zerolinecolor="#444444", zerolinewidth=1
            ),
            template="plotly_white",
            height=480,
            margin=dict(l=20, r=40, t=40, b=20)
        )

        estilo_figura(fig_torn)
        st.plotly_chart(
            fig_torn,
            use_container_width=True,
            theme=None,
            config={
                'toImageButtonOptions': {
                    'format': 'svg',
                    'filename': f'tornado_sensibilidad_{nombre_archivo(nombre_sitio)}',
                    'height': 500,
                    'width': 700,
                    'scale': 1
                }
            }
        )

        # --- NUEVO (Vacío 6): tabla de p-valores + advertencia metodológica ---
        with st.expander("🔎 Ver significancia estadística (p-valores)"):
            cols_tabla = ['Variable', 'Correlacion', 'p_valor', 'Significativo (p<0.05)']
            nombres_tabla = {'Correlacion': 'Spearman (ρ)'}
            formatos_tabla = {'Spearman (ρ)': '{:.3f}', 'p_valor': '{:.2e}'}
            if MOSTRAR_VISTA_CONTRIBUCION:
                cols_tabla.insert(2, 'Contribucion_%')
                nombres_tabla['Contribucion_%'] = 'Contrib. Varianza (%)'
                formatos_tabla['Contrib. Varianza (%)'] = '{:.1f}'
            st.dataframe(
                df_corr[cols_tabla]
                    .rename(columns=nombres_tabla)
                    .sort_values('Spearman (ρ)', key=abs, ascending=False)
                    .style.format(formatos_tabla),
                hide_index=True, use_container_width=True
            )
            st.caption(
                "⚠️ Con miles de iteraciones, casi cualquier correlación distinta de "
                "cero resultará estadísticamente significativa (p < 0.05), incluso si "
                "su magnitud es pequeña. El p-valor confirma que la relación no es "
                "producto del azar, pero la **magnitud** del coeficiente (no el "
                "p-valor) es la que debe usarse para priorizar variables en la "
                "discusión geológica."
            )

    with col_exp:
        st.markdown("#### 📥 Descargas e Historial")
        output = io.StringIO()
        df.to_csv(output, index=False, sep=';', decimal=',')

        st.download_button(
            label="📄 Descargar CSV Resultados (Excel)",
            data=output.getvalue(),
            file_name=f"Simulacion_Geotermica_{nombre_archivo(nombre_sitio)}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("**Resumen de Escenarios Ejecutados:**")
        if st.session_state.historial:
            st.dataframe(pd.DataFrame(st.session_state.historial).tail(5), use_container_width=True)
            if st.button("Limpiar Historial", use_container_width=True):
                st.session_state.historial = []
                st.rerun()

    st.markdown("---")

    # =========================================================================
    # NUEVO (Vacío 5): COMPARACIÓN CON MÉTODO DE VERIFICACIÓN INDEPENDIENTE
    # =========================================================================
    if 'Potencia_MW_Metodo2' in df.columns:
        st.subheader("🧪 Contraste con Método Independiente (Densidad de Potencia)")
        st.caption(
            "El método volumétrico USGS (columna izquierda) y el método de "
            "densidad de potencia (columna derecha) son metodológicamente "
            "independientes entre sí; ambos comparten únicamente la incertidumbre "
            "del área del reservorio. Su coincidencia relativa es un contraste "
            "cruzado indicativo, no una validación del recurso."
        )
        p90_m2, p50_m2, p10_m2 = np.percentile(df['Potencia_MW_Metodo2'], [10, 50, 90])
        df_comparacion = pd.DataFrame({
            'Percentil': ['P90', 'P50', 'P10'],
            'Método Volumétrico USGS (MWe)': [f"{p90:.2f}", f"{p50:.2f}", f"{p10:.2f}"],
            'Método Densidad de Potencia (MWe)': [f"{p90_m2:.2f}", f"{p50_m2:.2f}", f"{p10_m2:.2f}"],
        })
        st.dataframe(df_comparacion, hide_index=True, use_container_width=True)

        # =====================================================================
        # NUEVO (Crystal Ball): GRÁFICO DE SUPERPOSICIÓN (Overlay Chart)
        # -----------------------------------------------------------------------
        # Crystal Ball permite superponer dos o más distribuciones de pronóstico
        # en un mismo eje para comparar visualmente su forma, dispersión y
        # solapamiento — mucho más informativo que comparar solo tres percentiles
        # en una tabla. Aquí se superponen las distribuciones completas del
        # método volumétrico y del método de densidad de potencia.
        # =====================================================================
        st.markdown("#### 🔀 Superposición de Distribuciones (Overlay)")
        st.caption(
            "Ambas distribuciones completas, superpuestas con transparencia. "
            "Cuanto mayor sea el solapamiento entre las dos formas, mayor es la "
            "coherencia cruzada entre los dos métodos independientes; si están "
            "muy separadas, alguno de los dos (o ambos) requiere revisar sus "
            "supuestos de entrada."
        )
        pot_m1 = df['Potencia_MW']
        pot_m2 = df['Potencia_MW_Metodo2']
        bins_comunes = np.histogram_bin_edges(
            np.concatenate([pot_m1, pot_m2]), bins=50
        )

        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Histogram(
            x=pot_m1, xbins=dict(start=bins_comunes[0], end=bins_comunes[-1], size=bins_comunes[1] - bins_comunes[0]),
            histnorm='probability', name='Método Volumétrico USGS',
            marker=dict(color='rgba(30, 136, 229, 0.55)'),
            hovertemplate="Volumétrico: %{x:.1f} MWe<br>Probabilidad: %{y:.3f}<extra></extra>"
        ))
        fig_overlay.add_trace(go.Histogram(
            x=pot_m2, xbins=dict(start=bins_comunes[0], end=bins_comunes[-1], size=bins_comunes[1] - bins_comunes[0]),
            histnorm='probability', name='Densidad de Potencia',
            marker=dict(color='rgba(255, 138, 101, 0.55)'),
            hovertemplate="Densidad Potencia: %{x:.1f} MWe<br>Probabilidad: %{y:.3f}<extra></extra>"
        ))
        fig_overlay.update_layout(
            barmode='overlay',
            xaxis_title="Potencia Eléctrica Estimada (MWe)",
            yaxis_title="Probabilidad",
            template="plotly_white",
            height=440,
            margin=dict(l=60, r=20, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        estilo_figura(fig_overlay)
        st.plotly_chart(fig_overlay, use_container_width=True, theme=None)

    # =========================================================================
    # COMPARACIÓN CON ANTECEDENTES (opcional y editable; ver ANTECEDENTES_POR_DEFECTO)
    # =========================================================================
    if mostrar_antecedentes and antecedentes:
        st.subheader(f"📚 Comparación con Estimaciones Previas ({nombre_sitio})")
        filas_ant = [
            {
                'Estudio': a.get('estudio', ''),
                'Potencia Eléctrica (MWe)': a.get('mwe', '—'),
                'Calor Almacenado (EJ)': a.get('ej', '—'),
            }
            for a in antecedentes
        ]
        filas_ant.append({
            'Estudio': f'Esta simulación — P50 (mediana, n={n_iter_actual:,})',
            'Potencia Eléctrica (MWe)': f'{p50:.2f}',
            'Calor Almacenado (EJ)': f'{p50_ej:.3f}',
        })
        st.dataframe(pd.DataFrame(filas_ant), hide_index=True, use_container_width=True)
        st.caption(
            "Comparación referencial. Para que las cifras sean comparables, cada estudio debe "
            "compartir con esta simulación la convención de temperatura de referencia, el método "
            "de cálculo y los supuestos de entrada; de lo contrario, la diferencia puede "
            "deberse al método y no al recurso."
        )
