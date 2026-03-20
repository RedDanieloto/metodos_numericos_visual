import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# ============================
# MÉTODOS NUMÉRICOS
# ============================

def euler_mejorado_con_tabla(f, x0, y0, h, n):
    rows = []
    x = x0
    y = y0

    rows.append([0, x, y, np.nan, np.nan, np.nan, np.nan, 0.0])

    for i in range(1, n + 1):
        k1 = f(x, y)
        y_pred = y + h * k1
        k2 = f(x + h, y_pred)
        y_next = y + (h/2) * (k1 + k2)
        err = abs(y_next - y)

        rows.append([i, x, y, k1, y_pred, k2, y_next, err])

        x += h
        y = y_next

    return pd.DataFrame(
        rows,
        columns=["i", "x", "y", "k1", "y_pred", "k2", "y_next", "error"]
    )


def rk4_con_tabla(f, x0, y0, h, n):
    rows = []
    x = x0
    y = y0

    rows.append([0, x, y, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0])

    for i in range(1, n + 1):
        k1 = f(x, y)
        k2 = f(x + h/2, y + (h/2)*k1)
        k3 = f(x + h/2, y + (h/2)*k2)
        k4 = f(x + h, y + h*k3)

        y_next = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        err = abs(y_next - y)

        rows.append([i, x, y, k1, k2, k3, k4, y_next, err])

        x += h
        y = y_next

    return pd.DataFrame(
        rows,
        columns=["i", "x", "y", "k1", "k2", "k3", "k4", "y_next", "error"]
    )


def derivada_numerica(f, x, eps=None):
    if eps is None:
        eps = 1e-6 * max(1.0, abs(x))
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def newton_raphson_auto_df(f, x0, decimales=4, max_iter=100):
    rows = []
    x = x0
    tol = 0.5 * 10**(-decimales)

    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = derivada_numerica(f, x)

        if dfx == 0 or not np.isfinite(dfx):
            rows.append([i, x, fx, dfx, np.nan, "Derivada inválida"])
            break

        x_new = x - fx/dfx
        error = abs(x_new - x)

        rows.append([i, x, fx, dfx, x_new, error])

        x = x_new
        if error < tol:
            break

    df_table = pd.DataFrame(
        rows,
        columns=["iter", "x_i", "f(x_i)", "f'(x_i)", "x_{i+1}", "error"]
    )

    return round(x, decimales), df_table


# ============================
# UTILIDADES
# ============================

SAFE = {
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "pi": np.pi
}

def make_f_xy(expr):
    return lambda x, y: eval(expr, {"__builtins__": {}}, {**SAFE, "x": x, "y": y})

def make_f_x(expr):
    return lambda x: eval(expr, {"__builtins__": {}}, {**SAFE, "x": x})

def steps_from_xfinal(x0, x_final, h):
    if h <= 0 or x_final <= x0:
        return 0
    return int(np.floor((x_final - x0) / h))


def mostrar_tabla_centrada(df):
    html = df.to_html(index=False, na_rep="None")
    height = 40 + (len(df) + 1) * 38
    full_html = f"""
    <style>
        body {{ margin: 0; background-color: transparent; }}
        table {{ width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px; }}
        th {{ text-align: center; padding: 8px 12px; background-color: #6366f1; color: #ffffff; border: 1px solid #4f46e5; font-weight: bold; }}
        td {{ text-align: center; padding: 6px 12px; color: #1f2937; border: 1px solid #e5e7eb; background-color: #f9fafb; }}
        tr:nth-child(even) td {{ background-color: #f3f4f6; }}
        tr:hover td {{ background-color: #e0e7ff; }}
    </style>
    {html}
    """
    components.html(full_html, height=height, scrolling=False)


# ============================
# INTERFAZ
# ============================

st.set_page_config(page_title="Métodos Numéricos", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
    color: #1f2937;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Contenedor principal con padding */
.main {
    padding: 20px;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Estilos para las pestañas */
[data-baseweb="tab-list"] {
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    border: 2px solid rgba(99, 102, 241, 0.2);
}

[data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background-color: #6366f1 !important;
    color: white !important;
}

/* Contenido de pestañas */
[role="tabpanel"] {
    padding: 30px 0 !important;
}

/* Secciones de inputs */
.input-section {
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #6366f1;
}

/* Secciones de gráficos */
.chart-section {
    background-color: rgba(255, 255, 255, 0.98) !important;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
}

/* Secciones de tablas */
.table-section {
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
}

/* Grid para inputs - 4 columnas responsivo */
.input-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-bottom: 15px;
}

/* Estilos para inputs y botones */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: #f9fafb !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 10px !important;
    color: #1f2937 !important;
    font-size: 14px !important;
    padding: 12px !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border: 2px solid #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

.stButton > button {
    background-color: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: bold !important;
    font-size: 16px !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background-color: #4f46e5 !important;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
    transform: translateY(-2px);
}

/* Etiquetas de inputs */
.stTextInput > label,
.stNumberInput > label {
    font-weight: 600 !important;
    color: #374151 !important;
    font-size: 14px !important;
}

/* Título */
h1 {
    color: #ffffff !important;
    text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
    font-size: 3rem !important;
    margin-bottom: 30px !important;
    text-align: center;
}

h2, h3 {
    color: #1f2937;
    font-weight: 700;
}

/* Mensajes de error */
.stAlert {
    background-color: #fee2e2 !important;
    border-left: 5px solid #ef4444 !important;
    border-radius: 10px !important;
    padding: 15px !important;
}

/* Mensajes de información */
.stMarkdown {
    background-color: transparent;
}

/* Gráficos con mejor apariencia */
.stPlotlyChart, .stPyplotChart {
    background-color: #f9fafb !important;
    border-radius: 10px;
    padding: 15px;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.custom-header {
    position: fixed;
    top: 15px;
    right: 80px;
    font-size: 14px;
    font-weight: 500;
    color: #ffffff;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    z-index: 1000;
    background-color: rgba(0, 0, 0, 0.2);
    padding: 8px 15px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}
</style>
<div class="custom-header">
Angel Heriberto Rivera Saucedo
            </div>
""", unsafe_allow_html=True)

st.title("Métodos Numéricos")

tab_euler, tab_rk4, tab_newton = st.tabs(
    ["Euler Mejorado", "Runge-Kutta 4 (RK4)", "Newton-Raphson"]
)

# ============================
# TAB EULER
# ============================

with tab_euler:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Parámetros")
    
    # Crear columnas para los inputs
    col_expr, col_x0, col_y0 = st.columns(3)
    with col_expr:
        expr = st.text_input("f(x,y) =", value="x + y")
    with col_x0:
        x0 = st.number_input("x₀", value=0.0)
    with col_y0:
        y0 = st.number_input("y₀", value=1.0)
    
    col_h, col_xf = st.columns(2)
    with col_h:
        h = st.number_input("h (paso)", value=0.1, min_value=1e-6)
    with col_xf:
        x_final = st.number_input("x_final", value=1.0)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        resolver = st.button("🚀 Resolver Euler", use_container_width=True)
    
    f = make_f_xy(expr)

    if resolver:
        n = steps_from_xfinal(x0, x_final, h)
        if n <= 0:
            st.error("❌ x_final debe ser mayor que x0 y h positivo.")
        else:
            df = euler_mejorado_con_tabla(f, x0, y0, h, n)

            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df["x"], df["y_next"], marker="o", linewidth=2, markersize=6, color="#6366f1")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("x", fontsize=12, fontweight="bold")
            ax.set_ylabel("y", fontsize=12, fontweight="bold")
            ax.set_title("Método de Euler Mejorado", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="table-section">', unsafe_allow_html=True)
            st.markdown("### 📊 Tabla de Resultados")
            mostrar_tabla_centrada(df)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================
# TAB RK4
# ============================

with tab_rk4:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Parámetros")
    
    col_expr, col_x0, col_y0 = st.columns(3)
    with col_expr:
        expr = st.text_input("f(x,y) =", value="x + y", key="rk4_expr")
    with col_x0:
        x0 = st.number_input("x₀", value=0.0, key="rk4_x0")
    with col_y0:
        y0 = st.number_input("y₀", value=1.0, key="rk4_y0")
    
    col_h, col_xf = st.columns(2)
    with col_h:
        h = st.number_input("h (paso)", value=0.1, min_value=1e-6, key="rk4_h")
    with col_xf:
        x_final = st.number_input("x_final", value=1.0, key="rk4_xf")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        resolver = st.button("🚀 Resolver RK4", use_container_width=True)

    f = make_f_xy(expr)

    if resolver:
        n = steps_from_xfinal(x0, x_final, h)
        if n <= 0:
            st.error("❌ x_final debe ser mayor que x0 y h positivo.")
        else:
            df = rk4_con_tabla(f, x0, y0, h, n)

            st.markdown('<div class="chart-section">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df["x"], df["y_next"], marker="s", linewidth=2, markersize=6, color="#8b5cf6")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("x", fontsize=12, fontweight="bold")
            ax.set_ylabel("y", fontsize=12, fontweight="bold")
            ax.set_title("Método Runge-Kutta 4 (RK4)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="table-section">', unsafe_allow_html=True)
            st.markdown("### 📊 Tabla de Resultados")
            mostrar_tabla_centrada(df)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================
# TAB NEWTON
# ============================

with tab_newton:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Parámetros")
    
    col_expr, col_x0, col_dec = st.columns(3)
    with col_expr:
        expr_fx = st.text_input("f(x) =", value="x**3 - x - 2")
    with col_x0:
        x_init = st.number_input("x₀", value=1.0)
    with col_dec:
        decimales = st.number_input("Cifras decimales", value=4, min_value=1, max_value=12)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_btn, col_empty = st.columns([1, 3])
    with col_btn:
        resolver = st.button("🚀 Resolver Newton", use_container_width=True)

    f = make_f_x(expr_fx)

    if resolver:
        root, tabla = newton_raphson_auto_df(f, x_init, decimales=int(decimales))

        xs = np.linspace(x_init - 5, x_init + 5, 400)
        ys = [f(x) for x in xs]

        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(xs, ys, linewidth=2.5, color="#ec4899", label="f(x)")
        ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--", alpha=0.5)
        ax.plot(root, 0, "ro", markersize=10, label=f"Raíz: {root}")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x", fontsize=12, fontweight="bold")
        ax.set_ylabel("f(x)", fontsize=12, fontweight="bold")
        ax.set_title("Método de Newton-Raphson", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="input-section" style="text-align: center; padding: 20px;">', unsafe_allow_html=True)
            st.markdown(f"### ✓ Raíz Aproximada")
            st.markdown(f"# {root}", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="table-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Tabla de Iteraciones")
        mostrar_tabla_centrada(tabla)
        st.markdown('</div>', unsafe_allow_html=True)