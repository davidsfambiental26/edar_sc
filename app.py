import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import tempfile
import os

# -----------------------------
# Modelo sintético de DBO5
# -----------------------------

def simulate_DBO5(DO, MLSS, Q, Q_max):
    DO_norm = np.clip(DO / 4.0, 0, 1)
    MLSS_norm = np.clip((MLSS - 2000) / 4000, 0, 1)
    carga_norm = np.clip(Q / (Q_max + 1e-6), 0, 1)

    removal_eff = 0.2 + 0.5 * DO_norm + 0.3 * MLSS_norm
    removal_eff = np.clip(removal_eff, 0, 0.95)

    overload_penalty = 1 + 1.5 * carga_norm

    DBO5 = 60 * (1 - removal_eff) * overload_penalty
    DBO5 += np.random.normal(0, 0.8)

    return max(0, DBO5)

# -----------------------------
# Simulación dinámica 4 horas
# -----------------------------

def simulate_4h(Q_base, DO_base, MLSS_base, scenario_factor, scenario_type, blower_intensity=1.0):
    # Valores pensados para una EDAR media en Tenerife
    Q_max = 500
    minutes = 240

    times = []
    Q_list = []
    DO_list = []
    DBO5_list = []

    Q = Q_base
    DO = DO_base
    MLSS = MLSS_base

    saturation_time = None

    for minute in range(minutes):
        t_h = minute / 60.0
        times.append(t_h)

        if scenario_type == "Vertido industrial":
            DBO5_in = 35 * (1 + scenario_factor / 100)  # algo más agresivo
            Q = Q_base
            DO = max(0.5, DO - 0.015 * (scenario_factor / 100))
        else:  # Lluvia torrencial
            Q = Q_base * (1 + scenario_factor / 100)
            DBO5_in = 28  # dilución típica
            DO = max(0.5, DO - 0.02 * (scenario_factor / 100))

        DO += 0.035 * (blower_intensity - 1.0)

        DBO5_eff = simulate_DBO5(DO, MLSS, Q, Q_max)

        carga_hidraulica = Q / Q_max

        # Umbrales de riesgo ajustados:
        # - Carga hidráulica > 0.9 → riesgo alto
        # - DBO5 efluente > 35 mg/L → riesgo alto
        if saturation_time is None and (carga_hidraulica > 0.9 or DBO5_eff > 35):
            saturation_time = t_h

        Q_list.append(Q)
        DO_list.append(DO)
        DBO5_list.append(DBO5_eff)

    df = pd.DataFrame({
        "t_h": times,
        "Q": Q_list,
        "DO": DO_list,
        "DBO5": DBO5_list
    })

    return df, saturation_time

# -----------------------------
# Modelo ML sintético
# -----------------------------

@st.cache_data
def train_ml_model(n_samples=2000):
    X = []
    y = []

    for _ in range(n_samples):
        scenario_type = np.random.choice([0, 1])  # 0: vertido, 1: lluvia
        factor = np.random.uniform(0, 300)
        Q_base = np.random.uniform(80, 220)
        DO_base = np.random.uniform(1.5, 3.5)
        MLSS_base = np.random.uniform(2600, 4200)
        blower = np.random.choice([0.5, 1.0, 1.5])

        scen_str = "Vertido industrial" if scenario_type == 0 else "Lluvia torrencial"
        df_sim, sat_time = simulate_4h(Q_base, DO_base, MLSS_base, factor, scen_str, blower)

        if sat_time is None:
            sat_time = 4.0

        X.append([scenario_type, factor, Q_base, DO_base, MLSS_base, blower])
        y.append(sat_time)

    model = RandomForestRegressor(n_estimators=80, random_state=42)
    model.fit(np.array(X), np.array(y))
    return model

def ml_predict_saturation(model, scenario_type_str, factor, Q_base, DO_base, MLSS_base, blower):
    scenario_type = 0 if scenario_type_str == "Vertido industrial" else 1
    X = np.array([[scenario_type, factor, Q_base, DO_base, MLSS_base, blower]])
    return float(model.predict(X)[0])

# -----------------------------
# Recomendaciones operativas
# -----------------------------

def generate_recommendations(sat_time, scenario, factor, blower):
    recs = []

    if sat_time is None or sat_time >= 4.0:
        recs.append("El sistema se mantiene estable en las próximas 4 horas.")
        if factor > 150:
            recs.append("Se recomienda vigilancia reforzada de DO y DBO5 cada 30 minutos durante el evento.")
        return recs

    if sat_time < 1.0:
        recs.append("Riesgo muy alto: saturación en menos de 1 hora.")
    elif sat_time < 2.0:
        recs.append("Riesgo alto: saturación en menos de 2 horas.")
    else:
        recs.append("Riesgo moderado: saturación antes de 4 horas.")

    if scenario == "Vertido industrial":
        recs.append("Valorar derivar parte del caudal a tanque de tormentas o línea de retención previa.")
        recs.append("Incrementar aireación temporalmente (blower ≥ 1.5) mientras dure el vertido.")
    else:
        recs.append("Optimizar reparto hidráulico entre líneas y tanques de laminación.")
        recs.append("Mantener DO > 2 mg/L durante el episodio de lluvia para proteger el proceso biológico.")

    if blower < 1.0:
        recs.append("La aireación actual es baja; se recomienda subir al menos a 1.0 durante el evento.")
    elif blower > 1.0:
        recs.append("La aireación ya está elevada; revisar impacto energético y reducir tras el pico si es seguro.")

    return recs

# -----------------------------
# Mapa de calor de riesgo
# -----------------------------

def build_risk_heatmap(Q_base, DO_base, MLSS_base, blower):
    factors = np.linspace(0, 300, 16)
    scenarios = ["Vertido industrial", "Lluvia torrencial"]

    risk_matrix = np.zeros((len(scenarios), len(factors)))

    for i, scen in enumerate(scenarios):
        for j, f in enumerate(factors):
            df_sim, sat_time = simulate_4h(Q_base, DO_base, MLSS_base, f, scen, blower)
            if sat_time is None:
                risk = 0
            elif sat_time < 1:
                risk = 3
            elif sat_time < 2:
                risk = 2
            else:
                risk = 1
            risk_matrix[i, j] = risk

    return factors, scenarios, risk_matrix

# -----------------------------
# Generación de informe PDF
# -----------------------------

def generate_pdf_report(df_sim, scenario, factor, Q_base, DO_base, MLSS_base, blower, sat_time, ml_sat_time, recs, fig_time, fig_hm):
    tmpdir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmpdir, "informe_SGA_EDAR_Barrio_Buenos_Aires.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Portada
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, height - 2.5*cm, "Informe de simulación What-if – SGA EDAR Barrio Buenos Aires")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, height - 3.5*cm, f"Escenario: {scenario}")
    c.drawString(2*cm, height - 4.1*cm, f"Intensidad del evento: {factor:.1f} %")
    c.drawString(2*cm, height - 4.7*cm, f"Condiciones base: Q={Q_base:.1f} m³/h, DO={DO_base:.2f} mg/L, MLSS={MLSS_base:.0f} mg/L")
    c.drawString(2*cm, height - 5.3*cm, f"Intensidad de soplante: {blower}")
    if sat_time is None:
        c.drawString(2*cm, height - 6.1*cm, "Resultado simulación: No se prevé saturación en las próximas 4 horas.")
    else:
        c.drawString(2*cm, height - 6.1*cm, f"Resultado simulación: Saturación estimada en {sat_time:.2f} horas.")
    if ml_sat_time >= 4.0:
        c.drawString(2*cm, height - 6.7*cm, "Modelo ML: No se prevé saturación en las próximas 4 horas.")
    else:
        c.drawString(2*cm, height - 6.7*cm, f"Modelo ML: Saturación estimada en {ml_sat_time:.2f} horas.")
    c.showPage()

    # Página 2: gráficas temporales
    time_fig_path = os.path.join(tmpdir, "time_series.png")
    fig_time.savefig(time_fig_path, dpi=150, bbox_inches="tight")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 2.5*cm, "Evolución temporal de Q, DO y DBO5 (4 horas)")
    c.drawImage(time_fig_path, 2*cm, height - 18*cm, width=16*cm, preserveAspectRatio=True, mask='auto')
    c.showPage()

    # Página 3: mapa de calor de riesgo
    hm_fig_path = os.path.join(tmpdir, "heatmap_riesgo.png")
    fig_hm.savefig(hm_fig_path, dpi=150, bbox_inches="tight")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 2.5*cm, "Mapa de calor de riesgo de saturación")
    c.drawImage(hm_fig_path, 2*cm, height - 18*cm, width=16*cm, preserveAspectRatio=True, mask='auto')
    c.showPage()

    # Página 4: recomendaciones
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 2.5*cm, "Recomendaciones operativas para el SGA")
    c.setFont("Helvetica", 11)
    y = height - 4*cm
    for r in recs:
        c.drawString(2*cm, y, f"- {r}")
        y -= 0.8*cm
        if y < 3*cm:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 3*cm

    c.showPage()
    c.save()
    return pdf_path

# -----------------------------
# Interfaz Streamlit
# -----------------------------

st.title("Motor de Simulación What‑If – Gemelo Digital EDAR Barrio Buenos Aires (Santa Cruz de Tenerife)")

st.sidebar.header("Parámetros base de operación (madrugada)")
Q_base = st.sidebar.slider("Caudal base Q (m³/h)", 80, 250, 130)
DO_base = st.sidebar.slider("DO base (mg/L)", 1.0, 4.0, 2.7, 0.1)
MLSS_base = st.sidebar.slider("MLSS base (mg/L)", 2600, 4500, 3300)
blower = st.sidebar.select_slider("Intensidad de soplante", options=[0.5, 1.0, 1.5], value=1.0)

st.header("Escenario What‑If")

scenario = st.selectbox("Selecciona escenario:", ["Vertido industrial", "Lluvia torrencial"])
factor = st.slider("Intensidad del evento (%)", 0, 300, 0)

st.write("Simulando 4 horas a partir de las condiciones actuales...")

df_sim, sat_time = simulate_4h(Q_base, DO_base, MLSS_base, factor, scenario, blower)

if sat_time is None:
    st.success("El reactor NO se satura en las próximas 4 horas.")
else:
    st.error(f"Si esto ocurre, dentro de {sat_time:.2f} horas el reactor se saturará.")

# Gráficas temporales
st.subheader("Evolución temporal de variables de proceso (4 horas)")

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(df_sim["t_h"], df_sim["Q"], color="tab:blue")
axes[0].set_ylabel("Q (m³/h)")
axes[0].set_title("Caudal")

axes[1].plot(df_sim["t_h"], df_sim["DO"], color="tab:green")
axes[1].axhline(2.0, color="red", linestyle="--", label="DO mínimo 2 mg/L")
axes[1].set_ylabel("DO (mg/L)")
axes[1].legend()

axes[2].plot(df_sim["t_h"], df_sim["DBO5"], color="tab:red")
axes[2].axhline(25, color="black", linestyle="--", label="Límite DBO5 25 mg/L")
axes[2].set_ylabel("DBO5 (mg/L)")
axes[2].set_xlabel("Tiempo (h)")
axes[2].legend()

plt.tight_layout()
st.pyplot(fig)

# Mapa de calor de riesgo
st.subheader("Mapa de calor de riesgo de saturación")

factors, scenarios_list, risk_matrix = build_risk_heatmap(Q_base, DO_base, MLSS_base, blower)

fig_hm, ax_hm = plt.subplots(figsize=(10, 3))
sns.heatmap(
    risk_matrix,
    annot=True,
    fmt=".0f",
    cmap="Reds",
    xticklabels=[f"{int(f)}%" for f in factors],
    yticklabels=scenarios_list,
    cbar_kws={"label": "Nivel de riesgo (0=sin, 3=alto)"}
)
ax_hm.set_xlabel("Intensidad del evento (%)")
ax_hm.set_ylabel("Escenario")
st.pyplot(fig_hm)

# Modelo ML y predicción
st.subheader("Predicción ML del tiempo hasta saturación")

with st.spinner("Entrenando modelo ML sintético (una sola vez)..."):
    model = train_ml_model()

ml_sat_time = ml_predict_saturation(model, scenario, factor, Q_base, DO_base, MLSS_base, blower)

if ml_sat_time >= 4.0:
    st.info("Modelo ML: no se espera saturación en las próximas 4 horas.")
else:
    st.warning(f"Modelo ML: se estima saturación en {ml_sat_time:.2f} horas.")

# Recomendaciones
st.subheader("Recomendaciones operativas")

recs = generate_recommendations(sat_time, scenario, factor, blower)
for r in recs:
    st.write("- ", r)

# Modo informe SGA
st.subheader("Informe SGA en PDF")

if st.button("Generar informe PDF para auditoría"):
    pdf_path = generate_pdf_report(
        df_sim, scenario, factor, Q_base, DO_base, MLSS_base,
        blower, sat_time, ml_sat_time, recs, fig, fig_hm
    )
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Descargar informe PDF",
            data=f,
            file_name="informe_SGA_EDAR_Barrio_Buenos_Aires.pdf",
            mime="application/pdf"
        )

