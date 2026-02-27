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
        th {{ text-align: center; padding: 8px 12px; background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #444; }}
        td {{ text-align: center; padding: 6px 12px; color: #cdd6f4; border: 1px solid #333; background-color: #181825; }}
        tr:nth-child(even) td {{ background-color: #1e1e2e; }}
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
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
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
    z-index: 1000;
}
</style>
<div class="custom-header">
Nestor Daniel Cabrera Garcia
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
    col1, col2 = st.columns(2)

    with col1:
        expr = st.text_input("f(x,y) =", value="x + y")
        x0 = st.number_input("x0", value=0.0)
        y0 = st.number_input("y0", value=1.0)
        h = st.number_input("h (paso)", value=0.1, min_value=1e-6)
        x_final = st.number_input("x_final", value=1.0)

    f = make_f_xy(expr)

    if st.button("Resolver Euler"):
        n = steps_from_xfinal(x0, x_final, h)
        if n <= 0:
            st.error("x_final debe ser mayor que x0 y h positivo.")
        else:
            df = euler_mejorado_con_tabla(f, x0, y0, h, n)

            fig, ax = plt.subplots()
            ax.plot(df["x"], df["y_next"], marker="o")
            ax.grid(True)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            with col2:
                st.pyplot(fig)

            mostrar_tabla_centrada(df)

# ============================
# TAB RK4
# ============================

with tab_rk4:
    col1, col2 = st.columns(2)

    with col1:
        expr = st.text_input("f(x,y) =", value="x + y", key="rk4")
        x0 = st.number_input("x0", value=0.0, key="rk4x")
        y0 = st.number_input("y0", value=1.0, key="rk4y")
        h = st.number_input("h (paso)", value=0.1, min_value=1e-6, key="rk4h")
        x_final = st.number_input("x_final", value=1.0, key="rk4xf")

    f = make_f_xy(expr)

    if st.button("Resolver RK4"):
        n = steps_from_xfinal(x0, x_final, h)
        if n <= 0:
            st.error("x_final debe ser mayor que x0 y h positivo.")
        else:
            df = rk4_con_tabla(f, x0, y0, h, n)

            fig, ax = plt.subplots()
            ax.plot(df["x"], df["y_next"], marker="s")
            ax.grid(True)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            with col2:
                st.pyplot(fig)

            mostrar_tabla_centrada(df)

# ============================
# TAB NEWTON
# ============================

with tab_newton:
    col1, col2 = st.columns(2)

    with col1:
        expr_fx = st.text_input("f(x) =", value="x**3 - x - 2")
        x_init = st.number_input("x0", value=1.0)
        decimales = st.number_input("Cifras decimales", value=4, min_value=1, max_value=12)

    f = make_f_x(expr_fx)

    if st.button("Resolver Newton"):
        root, tabla = newton_raphson_auto_df(f, x_init, decimales=int(decimales))

        xs = np.linspace(x_init - 5, x_init + 5, 400)
        ys = [f(x) for x in xs]

        fig, ax = plt.subplots()
        ax.plot(xs, ys)
        ax.axhline(0)
        ax.grid(True)

        with col2:
            st.pyplot(fig)

        st.write("Raíz aproximada:", root)
        mostrar_tabla_centrada(tabla)