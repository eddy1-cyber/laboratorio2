import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="K-Means Clientes", layout="wide")

st.title(" Segmentación de Clientes (Automática)")

# 🔹 Función de limpieza
def limpiar_datos(df, country_sel):

    columnas_necesarias = ['CustomerID', 'Quantity', 'UnitPrice', 'InvoiceNo', 'Country']

    if all(col in df.columns for col in columnas_necesarias):

        st.info(" Dataset crudo detectado → limpiando automáticamente...")

        # Filtrar país
        df = df[df['Country'] == country_sel]

        # Eliminar nulos
        df = df.dropna(subset=['CustomerID'])

        # Eliminar devoluciones
        df = df[df['Quantity'] > 0]

        # Crear TotalSum
        df['TotalSum'] = df['Quantity'] * df['UnitPrice']

        # Agrupar por cliente
        df_clientes = df.groupby('CustomerID').agg({
            'TotalSum': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()

        df_clientes.columns = ['CustomerID', 'GastoTotal', 'Frecuencia']

        return df_clientes

    else:
        st.success("✅ Dataset ya limpio detectado")
        return df


# 🔹 Cargar archivo
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:

    df_raw = pd.read_csv(archivo)

    # 🔹 Detectar si tiene Country
    if 'Country' in df_raw.columns:

        paises = df_raw['Country'].dropna().unique()
        country_sel = st.sidebar.selectbox(" Selecciona un país", sorted(paises))

    else:
        country_sel = None

    # 🔹 Limpiar datos
    df = limpiar_datos(df_raw, country_sel)

    st.subheader(" Datos procesados")
    st.write(df.head())

    # 🔹 Validar columnas finales
    if 'GastoTotal' in df.columns and 'Frecuencia' in df.columns:

        # 🔹 Slider K
        k = st.sidebar.slider("Número de clusters (K)", 2, 10, 4)

        # 🔹 Preparar datos
        X = df[['GastoTotal', 'Frecuencia']]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 🔹 Modelo
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        centroides = scaler.inverse_transform(kmeans.cluster_centers_)

        # 🔹 Gráfico
        st.subheader(" Clusters")

        fig = px.scatter(
            df,
            x='GastoTotal',
            y='Frecuencia',
            color=df['Cluster'].astype(str),
            opacity=0.7
        )

        fig.add_trace(
            go.Scatter(
                x=centroides[:, 0],
                y=centroides[:, 1],
                mode='markers',
                marker=dict(size=15, symbol='x'),
                name='Centroides'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # 🔹 Selector cluster
        st.subheader(" Análisis por Cluster")

        cluster_sel = st.selectbox(
            "Selecciona un cluster",
            sorted(df['Cluster'].unique())
        )

        df_cluster = df[df['Cluster'] == cluster_sel]

        # 🔹 Métricas
        varianza = df_cluster['GastoTotal'].var()
        desviacion = df_cluster['GastoTotal'].std()

        col1, col2 = st.columns(2)
        col1.metric("Varianza", f"{varianza:,.2f}")
        col2.metric("Desviación Estándar", f"{desviacion:,.2f}")

        st.write(f" Clientes en el cluster: {len(df_cluster)}")

    else:
        st.error(" No se pudieron generar las columnas necesarias")

else:
    st.info(" Sube un archivo CSV para comenzar")