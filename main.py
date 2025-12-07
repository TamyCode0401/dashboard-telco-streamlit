import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from io import BytesIO
import joblib
import pickle


sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Dashboard Telco Churn", layout="wide")
st.title("📊 Dashboard Telco Churn + Machine Learning (Streamlit)")

# ======================================================================
# LOAD CSV
# ======================================================================
archivo = st.sidebar.file_uploader("📥 Subir CSV (delimitado por ;)", type=["csv"])
if archivo is None:
    st.stop()

df = pd.read_csv(archivo, delimiter=";")
columnas_num = df.select_dtypes(include=["number"]).columns.tolist()
columnas_cat = df.select_dtypes(exclude=["number"]).columns.tolist()

st.success("✔ CSV cargado correctamente")

# ======================================================================
# FILTROS
# ======================================================================
st.sidebar.header("🔎 Filtros")

filtro_cat = st.sidebar.selectbox("Filtrar por:", columnas_cat)
valores = df[filtro_cat].unique().tolist()
sel_valores = st.sidebar.multiselect("Valores:", valores, default=valores)

df_filtrado = df[df[filtro_cat].isin(sel_valores)]

# ======================================================================
# TABS
# ======================================================================
tabs = st.tabs(["📘 Datos", "📊 Visualizaciones (6)", "🤖 Modelo"])

# ======================================================================
# TAB 1 — DATOS
# ======================================================================
with tabs[0]:
    st.subheader("📘 Vista previa del dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("📘 Vista previa filtrada")
    st.dataframe(df_filtrado, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas Totales", len(df))
    col2.metric("Filtradas", len(df_filtrado))
    col3.metric("Columnas", len(df.columns))

# ======================================================================
# TAB 2 — VISUALIZACIONES (6)
# ======================================================================
with tabs[1]:

    st.subheader("📊 Selección de columnas")

    colA, colB, colC = st.columns(3)

    cat1 = colA.selectbox("Gráfico Categórico 1 (Countplot):", columnas_cat)
    cat2 = colB.selectbox("Gráfico Categórico 2 (Pie Chart):", columnas_cat)
    num1 = colC.selectbox("Gráfico Numérico 1 (Boxplot):", columnas_num)

    colD, colE, colF = st.columns(3)

    num2 = colD.selectbox("Gráfico Numérico 2 (Histograma):", columnas_num)
    scat_x = colE.selectbox("Scatter X:", columnas_num)
    scat_y = colF.selectbox("Scatter Y:", columnas_num)

    st.markdown("---")

    # GRID 3×2
    g1, g2, g3 = st.columns(3)
    g4, g5, g6 = st.columns(3)

    # ---------------------------------------------------
    # 1 — COUNTPLOT
    with g1:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df_filtrado, x=cat1, ax=ax, palette="Blues")
        plt.xticks(rotation=45)
        ax.set_title(f"Countplot — {cat1}")
        st.pyplot(fig)

    # ---------------------------------------------------
    # 2 — PIE CHART
    with g2:
        fig, ax = plt.subplots(figsize=(4, 3))
        df_filtrado[cat2].value_counts().plot(
            kind="pie", autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        ax.set_title(f"Pie Chart — {cat2}")
        st.pyplot(fig)

    # ---------------------------------------------------
    # 3 — BOXPLOT
    with g3:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(data=df_filtrado, y=num1, ax=ax, palette="coolwarm")
        ax.set_title(f"Boxplot — {num1}")
        st.pyplot(fig)

    # ---------------------------------------------------
    # 4 — HISTOGRAMA
    with g4:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df_filtrado[num2], kde=True, ax=ax)
        ax.set_title(f"Histograma — {num2}")
        st.pyplot(fig)

    # ---------------------------------------------------
    # 5 — SCATTERPLOT
    with g5:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.scatterplot(data=df_filtrado, x=scat_x, y=scat_y, ax=ax)
        ax.set_title(f"Scatter — {scat_x} vs {scat_y}")
        st.pyplot(fig)

    # ---------------------------------------------------
    # 6 — HEATMAP
    with g6:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(df_filtrado[columnas_num].corr(),
                    annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap de Correlaciones")
        st.pyplot(fig)

# ======================================================================
# TAB 3 — MODELO .PICKLE
# ======================================================================
with tabs[2]:
    st.subheader("🤖 Cargar modelo entrenado (.pickle / .pkl)")

    modelo_file = st.file_uploader("📥 Subir modelo .pickle", type=["pickle", "pkl"])

    if modelo_file:
        modelo = pickle.load(modelo_file)
        st.success("✔ Modelo cargado correctamente")

        target = st.selectbox("Selecciona columna Target:", ["churn_value_target", "churn_target"])

        X = df_filtrado.drop(columns=[target])
        y = df_filtrado[target]

        # mismas columnas que el modelo
        X = pd.get_dummies(X, drop_first=True)
        X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)

        pred = modelo.predict(X)

        st.subheader("📌 Predicciones")
        pred_df = pd.DataFrame({"Real": y, "Predicho": pred})
        st.dataframe(pred_df, use_container_width=True)

        # Matriz de confusión
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Matriz de Confusión")
        st.pyplot(fig)

        # Reporte
        st.text(classification_report(y, pred))

        # Descargar CSV
        buff = BytesIO()
        pred_df.to_csv(buff, index=False)
        buff.seek(0)

        st.download_button(
            "⬇ Descargar predicciones",
            data=buff,
            file_name="predicciones_churn.csv",
            mime="text/csv"
        )
