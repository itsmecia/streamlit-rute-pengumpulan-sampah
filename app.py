import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from itertools import cycle

st.set_page_config(page_title="Analisis Big Data - Rute TPS–TPA", layout="wide")

# Warna dan gaya global
st.markdown("""
<style>
/* WARNA DASAR*/
html, body, .main, .block-container {
    background-color: #E1F5E1 !important;
    font-family: 'Poppins', sans-serif;
    color: #000000 !important;
}

/* SIDEBAR*/
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom right, #c8e6c9, #DFFDDF);
    padding: 20px 10px 60px 10px;
    border-right: 2px solid #a5d6a7;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #1b4d3e !important;
    text-align: center;
}

/* NAVIGASI CARD  */
.menu-card {
    background-color: #FFFFFF !important;
    border: 1px solid #a5d6a7;
    border-radius: 12px;
    padding: 14px;
    margin: 10px 0;
    text-align: center;
    transition: all 0.3s ease;
    font-weight: 500;
    color: #2e7d32;
    cursor: pointer;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.menu-card:hover {
    background-color: #c8e6c9 !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    transform: translateY(-3px);
}

.menu-card.active {
    background-color: #81c784 !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 3px 8px rgba(0,0,0,0.3);
}

a.menu-link {
    text-decoration: none;
}

/*  HEADER  */
h1, h2, h3 {
    color: #000000 !important;
}

/*  METRIK  */
[data-testid="stMetricValue"] {
    color: #2e7d32 !important;
}

/*  INFO CARD  */
.info-card {
    background-color: #FFFFFF !important;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 14px;
    color: #1b4d3e;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}


/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #81c784;
    border-radius: 10px;
}


</style>
""", unsafe_allow_html=True)

#header
st.title("🌍 Sistem Analisis Rute & Pengumpulan Sampah")
st.markdown("Analitik dan optimasi rute pengangkutan sampah berbasis **Big Data**.")
st.markdown("---")

# data
def safe_read_csv(path, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        st.warning(f"Gagal memuat {path}: {e}")
        return pd.DataFrame()

tps_df = safe_read_csv("tps.csv")
tpa_df = safe_read_csv("tpa.csv")
histori_df = safe_read_csv("histori_rute.csv", parse_dates=["tanggal"])
routes_df = safe_read_csv("routes.csv")
vehicle_df = safe_read_csv("vehicle_routing_matrix.csv")

if tps_df.empty:
    st.info("Dataset TPS kosong / gagal dimuat. Beberapa fitur akan dinonaktifkan.")
if tpa_df.empty:
    st.info("Dataset TPA kosong / gagal dimuat. Beberapa fitur akan dinonaktifkan.")

if "keterisian_%" in tps_df.columns and "keterisian_%" not in tps_df.columns:
    tps_df = tps_df.rename(columns={"keterisian_%": "keterisian_%"})

# Helper: tambahkan marker TPS
def add_tps_marker(m, row, style="trash", popup_extra=None, tooltip=None):
    lat = row.get("latitude")
    lon = row.get("longitude")
    if pd.isna(lat) or pd.isna(lon):
        return
    label = tooltip or f"TPS {row.get('id_tps','-')}"
    keterisian = row.get("keterisian_%", 0) or 0
    popup_html = (
        f"<b>TPS {row.get('id_tps','-')}</b><br>"
        f"Kapasitas: {row.get('kapasitas','N/A')}<br>"
        f"Volume: {row.get('volume_saat_ini','N/A')}<br>"
        f"Keterisian: {keterisian:.1f}%"
    )
    if popup_extra:
        popup_html += f"<br>{popup_extra}"

    try:
        if style == "trash":
            folium.Marker(
                [lat, lon],
                popup=popup_html,
                tooltip=label,
                icon=folium.Icon(color="green", icon="trash", prefix="fa")
            ).add_to(m)
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px; color:green; font-weight:bold;">{row.get("id_tps","-")}</div>'
                ),
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.8,
                popup=popup_html,
                tooltip=label
            ).add_to(m)
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px; color:green;">{row.get("id_tps","-")}</div>'
                ),
            ).add_to(m)
    except Exception:
        pass
    
#sidebar
st.sidebar.markdown("<h2 style='text-align:center;'>📊 Navigasi Sistem</h2>", unsafe_allow_html=True)

# menu
menu_items = {
    "Dashboard Data": "📍 Dashboard Data",
    "Jadwal & Rute Pengangkutan": "🚛 Jadwal & Rute",
    "Prediksi Volume Sampah": "📈 Prediksi Volume"
}

# inisialisasi menu aktif
if "active_menu" not in st.session_state:
    st.session_state.active_menu = "Dashboard Data"  
    
def set_active(menu_name):
    st.session_state.active_menu = menu_name
    st.rerun()

for key, label in menu_items.items():
    is_active = st.session_state.active_menu == key

    # gaya tombol
    button_style = f"""
        display: flex;
        align-items: center;
        gap: 10px;
        background-color: {'#81c784' if is_active else '#FFFFFF'};
        color: {'white' if is_active else '#2e7d32'};
        border: 1px solid #a5d6a7;
        border-radius: 12px;
        font-weight: {'600' if is_active else '500'};
        padding: 10px 16px;
        margin-top: 8px;
        width: 100%;
        text-align: left;
        box-shadow: {'0 3px 8px rgba(0,0,0,0.3)' if is_active else '0 2px 4px rgba(0,0,0,0.1)'};
        transition: all 0.25s ease-in-out;
        cursor: pointer;
    """

    hover_style = """
        this.style.backgroundColor='#a5d6a7';
        this.style.color='white';
        this.style.transform='translateY(-2px)';
    """

    leave_style = f"""
        this.style.backgroundColor='{'#81c784' if is_active else '#FFFFFF'}';
        this.style.color='{'white' if is_active else '#2e7d32'}';
        this.style.transform='none';
    """

    # tombol sidebar 
    clicked = st.sidebar.button(
        label,
        key=f"btn_{key}",
        use_container_width=True,
    )
    st.sidebar.markdown(
        f"""
        <style>
        div[data-testid="stSidebar"] button[kind="secondary"][key="btn_{key}"] {{
            {button_style}
        }}
        div[data-testid="stSidebar"] button[kind="secondary"][key="btn_{key}"]:hover {{
            background-color: #a5d6a7 !important;
            color: white !important;
            transform: translateY(-2px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    if clicked:
        set_active(key)

mode = st.session_state.active_menu


# dayaset
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align:center;'>📂 Info Dataset</h3>", unsafe_allow_html=True)

st.sidebar.markdown(f'''
<div style="background-color:#fff; border-radius:10px; padding:12px; margin-top:12px;
font-size:14px; color:#1b4d3e; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
<b>tps.csv</b> – {len(tps_df)} baris<br>
<b>tpa.csv</b> – {len(tpa_df)} baris<br>
<b>histori_rute.csv</b> – {len(histori_df)} baris<br>
<b>routes.csv</b> – {len(routes_df)} baris<br>
<b>vehicle_routing_matrix.csv</b> – {len(vehicle_df)} baris
</div>
''', unsafe_allow_html=True)


st.sidebar.markdown("""
<div style='text-align:center; font-size:12px; margin-top:15px; opacity:0.7'>
Sistem ini menggunakan dataset internal untuk pemantauan & optimasi rute pengangkutan sampah di Delhi, India.
</div>
""", unsafe_allow_html=True)

# MODE: Dashboard Data 
if mode == "Dashboard Data":

    # METRIK UTAMA
    col1, col2, col3 = st.columns(3)
    col1.metric("Total TPS", len(tps_df))
    col2.metric("Total TPA", len(tpa_df))
    col3.metric("Total Histori Rute", len(histori_df))
    
    # FUNGSI UTILITAS
    def compute_keterisian(df):
        if "volume_saat_ini" in df.columns and "kapasitas" in df.columns:
            df["keterisian_%"] = (df["volume_saat_ini"] / df["kapasitas"]) * 100
            df["keterisian_%"] = df["keterisian_%"].fillna(0)
        else:
            if "keterisian_%" in df.columns:
                df["keterisian_%"] = df["keterisian_%"].fillna(0)
            else:
                df["keterisian_%"] = 0
        return df

    tps_df = compute_keterisian(tps_df)
    histori_df["tanggal"] = pd.to_datetime(histori_df["tanggal"], errors="coerce")
    histori_df["bulan"] = histori_df["tanggal"].dt.to_period("M").astype(str)

    st.markdown("---")

    #  PETA SEBARAN TPS & TPA
    st.subheader("Peta Sebaran Lokasi TPS dan TPA")
    
    # Filter TPS
    tps_options_map = sorted(tps_df["id_tps"].astype(str).unique().tolist())
    selected_tps_map = st.multiselect(
        "Pilih TPS:",
        tps_options_map,
        key="filter_tps_map"
    )
    
    if st.button("Reset Filter Peta", key="reset_peta"):
        selected_tps_map = []

    if selected_tps_map:
        filtered_tps_map = tps_df[tps_df["id_tps"].astype(str).isin(selected_tps_map)].copy()
    else:
        filtered_tps_map = tps_df.copy()
    
    # koordinat numerik
    for df in [filtered_tps_map, tpa_df]:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    
    filtered_tps_map = filtered_tps_map.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    tpa_valid = tpa_df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    
    # Tentukan pusat peta
    if not pd.concat([filtered_tps_map, tpa_valid]).empty:
        center_lat = pd.concat([filtered_tps_map, tpa_valid])["latitude"].mean()
        center_lon = pd.concat([filtered_tps_map, tpa_valid])["longitude"].mean()
    else:
        center_lat, center_lon = -7.8, 110.4  
    
    # Buat peta utama
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, control_scale=True)
    
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr=" "  
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
        name="Stamen Terrain",
        attr=" "
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        name="CartoDB Positron",
        attr=" "
    ).add_to(m)
    
      # Marker TPA
    for _, row in tpa_valid.iterrows():
        lat, lon = row["latitude"], row["longitude"]
    
        popup_html = f"""
        {row.get('nama', '-')}<br>
        <b>Koordinat:</b> {lat:.5f}, {lon:.5f}
        """
    
        # Marker utama
        folium.Marker(
            [lat, lon],
            popup=popup_html,
            tooltip=f"{row['nama']}",
            icon=folium.Icon(color="red", icon="recycle", prefix="fa"),
        ).add_to(m)
    
        # Label 
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 11px;
                    color: red;
                    font-weight: bold;
                    text-shadow: 1px 1px 2px #fff;
                    white-space: nowrap;
                    transform: translate(15px, -10px);
                ">
                    {row['nama']}
                </div>
                """
            )
        ).add_to(m)
    
    
    # Marker TPS
    for _, row in filtered_tps_map.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        keterisian = row.get("keterisian_%", 0)
    
        popup_html = f"""
        {row.get('id_tps','-')}<br>
        <b>Kapasitas:</b> {row.get('kapasitas','N/A')}<br>
        <b>Volume:</b> {row.get('volume_saat_ini','N/A')}<br>
        <b>Keterisian:</b> {keterisian:.1f}%
        """
    
        # Marker utama
        folium.Marker(
            [lat, lon],
            popup=popup_html,
            tooltip=f"{row['id_tps']}",
            icon=folium.Icon(color="green", icon="trash", prefix="fa"),
        ).add_to(m)

        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 11px;
                    color: green;
                    font-weight: bold; 
                    text-shadow: 1px 1px 2px #fff;
                    white-space: nowrap;
                    transform: translate(15px, -10px);
                ">
                    {row['id_tps']}
                </div>
                """
            )
        ).add_to(m)

    # Fit bounds semua titik
    all_points = pd.concat([filtered_tps_map[["latitude", "longitude"]], tpa_valid[["latitude", "longitude"]]])
    if not all_points.empty:
        m.fit_bounds([
            [all_points["latitude"].min(), all_points["longitude"].min()],
            [all_points["latitude"].max(), all_points["longitude"].max()],
        ])


    # Layer control
    folium.LayerControl().add_to(m)
    
    hide_attr_css = """
    <style>
    .leaflet-control-attribution {
        display: none !important;
    }
    </style>
    """
    st.markdown(hide_attr_css, unsafe_allow_html=True)
    
    legend_html = """
    <div style="
         position: absolute; 
         bottom: 3px; left: 130px;  
         z-index: 9999;
         background-color: rgba(255, 255, 255, 0.95);
         border: 1px solid #555;
         border-radius: 10px;
         padding: 10px 14px;
         font-size: 14px;
         line-height: 1.8;
         box-shadow: 0 3px 8px rgba(0,0,0,0.25);
         font-family: Arial, sans-serif;
         color: #222;
    ">
    <i class="fa fa-trash" style="color:green;"></i>
    <span style="font-weight:600; margin-left:6px;">TPS</span><br>
    <i class="fa fa-recycle" style="color:red;"></i>
    <span style="font-weight:600; margin-left:6px;">TPA</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    
    # Tampilkan peta
    st_folium(m, width=1000, height=550)
    st.markdown("---")

    # SCATTER: Kapasitas vs Volume
    st.subheader("Hubungan Kapasitas vs Volume per TPS")
    
    tps_options_scatter = sorted(tps_df["id_tps"].astype(str).unique().tolist())
    selected_tps_scatter = st.multiselect(
        "Pilih TPS:",
        tps_options_scatter,
        key="filter_tps_scatter"
    )
    
    if st.button("Reset Filter Scatter", key="reset_scatter"):
        selected_tps_scatter = []
    
    if selected_tps_scatter:
        tps_filtered_scatter = tps_df[tps_df["id_tps"].isin(selected_tps_scatter)]
    else:
        tps_filtered_scatter = tps_df.copy()
    
    if not tps_filtered_scatter.empty:
        # Ambang dinamis
        threshold = st.slider(
            "Atur ambang keterisian (%) untuk peringatan penuh:",
            50, 100, 85, step=1, key="slider_threshold_scatter"
        )
    
        # Tambahkan kolom status warna berdasarkan ambang
        def kategori_warna(x):
            if x >= threshold:
                return "Penuh"
            elif x >= threshold - 10:
                return "Hampir Penuh"
            else:
                return "Aman"
    
        tps_filtered_scatter["Status"] = tps_filtered_scatter["keterisian_%"].apply(kategori_warna)
    
        # Scatter plot dengan warna kategori
        fig_scatter = px.scatter(
            tps_filtered_scatter,
            x="kapasitas",
            y="volume_saat_ini",
            color="Status",
            size="keterisian_%",
            hover_name="id_tps",
            color_discrete_map={
                "Penuh": "red",
                "Hampir Penuh": "orange",
                "Aman": "green"
            },
            title=f"Kapasitas vs Volume Aktual TPS (Ambang {threshold}%)"
        )
    
        # Garis referensi Volume = Kapasitas
        max_val = max(
            tps_filtered_scatter["kapasitas"].max(),
            tps_filtered_scatter["volume_saat_ini"].max()
        )
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="gray", dash="dash")
        )
        fig_scatter.add_annotation(
            x=max_val*0.7, y=max_val*0.9,
            text="Volume = Kapasitas", showarrow=False
        )
    
        st.plotly_chart(fig_scatter, use_container_width=True)
    
        # Tambahkan legenda warna kustom
        legend_html = f"""
        <div style='text-align:center; margin-top:-10px;'>
            <span style='color:green;'>🟢 Aman (&lt; {threshold-10}%)</span> &nbsp;&nbsp;
            <span style='color:orange;'>🟠 Hampir Penuh ({threshold-10}–{threshold}%)</span> &nbsp;&nbsp;
            <span style='color:red;'>🔴 Penuh (&gt;= {threshold}%)</span>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        
    else:
        st.info("Tidak ada data untuk Scatter (TPS tidak dipilih).")
        
    st.markdown("### Insight")
    if not tps_filtered_scatter.empty:
        avg_fill = tps_filtered_scatter["keterisian_%"].mean()
        penuh = tps_filtered_scatter[tps_filtered_scatter["Status"] == "Penuh"]
        hampir = tps_filtered_scatter[tps_filtered_scatter["Status"] == "Hampir Penuh"]
    
        # Tampilkan insight utama
        st.write(f"- Rata-rata keterisian TPS (terfilter): **{avg_fill:.1f}%**")
    
        if not penuh.empty:
            st.warning(
                f" {len(penuh)} TPS melebihi ambang {threshold}%: "
                f"{', '.join(penuh['id_tps'].astype(str))}"
            )
        elif not hampir.empty:
            st.info(
                f"{len(hampir)} TPS mendekati ambang ({threshold-10}–{threshold}%): "
                f"{', '.join(hampir['id_tps'].astype(str))}"
            )
        else:
            st.success(f"Semua TPS masih di bawah {threshold-10}% keterisian.")
    
        avg_fill_all = tps_df["keterisian_%"].mean()
        corr = tps_df["kapasitas"].corr(tps_df["volume_saat_ini"])
        st.write(f"- Rata-rata keterisian TPS (keseluruhan): **{avg_fill_all:.1f}%**")
        st.write(f"- Korelasi kapasitas vs volume: **{corr:.2f}**")
    
    else:
        st.info("Tidak ada data TPS terfilter untuk dianalisis.")

    st.markdown("---")
    

    # TOP 5 TPS
    st.subheader("Top 5 TPS Berdasarkan Volume dan Persentase Keterisian")

    # FILTER INPUT (TPS SAJA)
    tps_options_top5 = sorted(histori_df["id_tps"].astype(str).unique().tolist())
    selected_tps_top5 = st.multiselect("Pilih TPS:", tps_options_top5, key="filter_tps_top5")

    if st.button("Reset Filter Top 5", key="reset_top5"):
        selected_tps_top5 = []

    # FILTER HISTORI BERDASARKAN TPS 
    hist_filtered_top5 = histori_df.copy()

    if selected_tps_top5:
        hist_filtered_top5 = hist_filtered_top5[hist_filtered_top5["id_tps"].astype(str).isin(selected_tps_top5)]

    # AGREGASI HISTORI DAN GABUNG DENGAN DATA TPS
    if not hist_filtered_top5.empty:
        hist_grouped = (
            hist_filtered_top5.groupby("id_tps", as_index=False)["Volume_kg"].sum()
        )
        merged_top5 = pd.merge(tps_df, hist_grouped, on="id_tps", how="right")
        merged_top5 = compute_keterisian(merged_top5)
    else:
        merged_top5 = pd.DataFrame(columns=tps_df.columns.tolist() + ["Volume_kg"])

    # PILIH KRITERIA 
    pilihan_kriteria = st.selectbox(
        "Pilih Kriteria Peringkat:",
        ["Volume Sampah Saat Ini", "Total Volume (Histori)", "Persentase Keterisian (%)"],
        key="kriteria_top5"
    )

    # TENTUKAN SORTING 
    if pilihan_kriteria == "Volume Sampah Saat Ini":
        kolom_sort = "volume_saat_ini"
        judul_grafik = "TPS dengan Volume Sampah Tertinggi (Aktual)"
    elif pilihan_kriteria == "Total Volume (Histori)":
        kolom_sort = "Volume_kg"
        judul_grafik = "TPS dengan Total Volume Sampah Tertinggi (Histori)"
    else:
        kolom_sort = "keterisian_%"
        judul_grafik = "TPS dengan Persentase Keterisian Tertinggi"

    # TAMPILKAN TOP 5 
    if not merged_top5.empty and kolom_sort in merged_top5.columns:
        top5 = merged_top5.sort_values(kolom_sort, ascending=False).head(5)
        fig_top5 = px.bar(
            top5, x="id_tps", y=kolom_sort, text=kolom_sort,
            color=kolom_sort, color_continuous_scale="Blues", title=judul_grafik
        )
        fig_top5.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        st.plotly_chart(fig_top5, use_container_width=True)

        st.markdown("### Insight")
        st.write(f"- Rata-rata {pilihan_kriteria.lower()} dari 5 TPS teratas: **{top5[kolom_sort].mean():.1f}**")
        st.write(f"- TPS teratas: **{top5.iloc[0]['id_tps']}**")
    else:
        st.info("Tidak ada data yang cocok dengan filter TPS yang dipilih.")
    st.markdown("---")    

# TREN VOLUME SAMPAH 
    st.subheader("Tren Volume Sampah Bulanan (Kota)")
    
    # Pilihan filter
    tps_options_tren = sorted(histori_df["id_tps"].unique().tolist())
    selected_tps_tren = st.multiselect("Pilih TPS:", tps_options_tren, key="filter_tps_tren")
    
    if st.button("Reset Filter Tren", key="reset_tren"):
        selected_tps_tren = []
    
    # Filter data
    hist_filtered_tren = histori_df.copy()
    if selected_tps_tren:
        hist_filtered_tren = hist_filtered_tren[hist_filtered_tren["id_tps"].isin(selected_tps_tren)]

    if not hist_filtered_tren.empty and "bulan" in hist_filtered_tren.columns:
        # Agregasi berdasarkan bulan
        monthly_trend = (
            hist_filtered_tren.groupby("bulan")["Volume_kg"].sum()
            .reset_index()
            .sort_values("bulan")
        )
    
        # Plot tren bulanan
        fig_trend = px.line(
            monthly_trend,
            x="bulan",
            y="Volume_kg",
            markers=True,
            title="Total Volume Sampah Bulanan",
            labels={"bulan": "Bulan", "Volume_kg": "Total Volume (kg)"}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
        # Insight tren
        st.markdown("### Insight")
        recent_avg = monthly_trend.tail(3)["Volume_kg"].mean() if len(monthly_trend) >= 3 else monthly_trend["Volume_kg"].mean()
        overall_avg = monthly_trend["Volume_kg"].mean()
        st.write(f"- Rata-rata 3 bulan terakhir: **{recent_avg:,.1f} kg/bulan**")
        st.write(f"- Rata-rata keseluruhan: **{overall_avg:,.1f} kg/bulan**")
        trend_note = "naik" if recent_avg > overall_avg else "turun/flat"
        st.write(f"- Tren recent vs keseluruhan: **{trend_note}**")
    
    else:
        st.info("Tidak ada data histori untuk periode / filter yang dipilih.")

    st.markdown("---")

    # Rata rata keterisian TPA
    if "nearest_tpa" in tps_df.columns:
        avg_per_tpa = tps_df.groupby("nearest_tpa")["keterisian_%"].mean().reset_index()
        st.subheader("Rata-rata keterisian per TPA")
        st.dataframe(
            avg_per_tpa.rename(columns={"nearest_tpa": "TPA", "keterisian_%": "Rata-rata (%)"}).round(2),
            use_container_width=True
        )

# MODE: Rute & jadwal
elif mode == "Jadwal & Rute Pengangkutan":

    # Fungsi Haversine
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    # Validasi Dataset
    if tps_df.empty or "nearest_tpa" not in tps_df.columns or "volume_saat_ini" not in tps_df.columns:
        st.warning("Pastikan file TPS memiliki kolom 'nearest_tpa' dan 'volume_saat_ini'.")
        st.stop()
    if "keterisian_%" not in tps_df.columns:
        tps_df["keterisian_%"] = (tps_df["volume_saat_ini"] / tps_df["kapasitas"]) * 100

    # Daftar Truk & Wilayah
    tpa_list = sorted(tpa_df["nama"].unique())
    all_trucks = [f"TR{str(i).zfill(2)}" for i in range(1, 11)]
    tpa_truck_map = {}
    split = [3, 3, 4]
    idx = 0
    for tpa, count in zip(tpa_list, split):
        tpa_truck_map[tpa] = all_trucks[idx:idx + count]
        idx += count

    st.subheader("Daftar Truk & Pembagian Wilayah")
    daftar_truk = []
    for tpa, trucks in tpa_truck_map.items():
        for truk in trucks:
            daftar_truk.append({"Truk": truk, "Wilayah (TPA)": tpa})
    st.dataframe(pd.DataFrame(daftar_truk), use_container_width=True)

    # Jadwal TPS
    tps_df = tps_df.copy()
    tps_df["prioritas_rank"] = tps_df.groupby("nearest_tpa")["keterisian_%"].rank(method="first", ascending=False)
    tps_df = tps_df.sort_values(["nearest_tpa", "prioritas_rank"])

    assigned_list = []
    for tpa in tps_df["nearest_tpa"].unique():
        subset = tps_df[tps_df["nearest_tpa"] == tpa].copy()
        trucks = tpa_truck_map.get(tpa, ["Cadangan"])
        subset["Truk"] = [trucks[i % len(trucks)] for i in range(len(subset))]
        assigned_list.append(subset)
    jadwal_final = pd.concat(assigned_list)

    jadwal_df = jadwal_final[[
        "id_tps", "nama", "nearest_tpa", "keterisian_%", "kapasitas",
        "volume_saat_ini", "Truk"
    ]].rename(columns={
        "id_tps": "ID TPS",
        "nama": "Nama TPS",
        "nearest_tpa": "Wilayah (TPA)",
        "keterisian_%": "Keterisian (%)",
        "kapasitas": "Kapasitas (m³)",
        "volume_saat_ini": "Volume Saat Ini (m³)"
    })

    st.subheader("Jadwal Pengangkutan")
    col1, col2 = st.columns(2)
    with col1:
        selected_truck = st.selectbox("Pilih Truk:", ["Semua"] + all_trucks, key="filter_truk")
    with col2:
        top_filter = st.selectbox("Tampilkan:", ["Semua TPS", "Top 5 Prioritas", "Top 10 Prioritas"], key="filter_top")

    filtered_df = jadwal_df.copy()
    if selected_truck != "Semua":
        filtered_df = filtered_df[filtered_df["Truk"] == selected_truck]
    if top_filter == "Top 5 Prioritas":
        filtered_df = filtered_df.sort_values("Keterisian (%)", ascending=False).head(5)
    elif top_filter == "Top 10 Prioritas":
        filtered_df = filtered_df.sort_values("Keterisian (%)", ascending=False).head(10)
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)


    st.markdown("---")


    # Rute Pengangkutan
    st.subheader("Rute Pengangkutan")
    
    tps_options = tps_df["id_tps"].astype(str).unique().tolist()
    selected_tps = st.multiselect("Pilih TPS", tps_options)
    
    # Titik tengah peta
    center_lat = float(tps_df["latitude"].mean())
    center_lon = float(tps_df["longitude"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # SEBELUM RUTE DICARI 
    if not selected_tps:
        # TPS icon 
        for _, row in tps_df.iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=f"{row['id_tps']}",
                icon=folium.Icon(color="green", icon="trash", prefix="fa")
            ).add_to(m)
            folium.map.Marker(
                [row["latitude"], row["longitude"]],
                icon=folium.DivIcon(html=f"<div style='font-size:12px; font-weight:bold; color:green; text-shadow:1px 1px 2px #fff;'>{row['id_tps']}</div>")
            ).add_to(m)
    
        # TPA icon 
        for _, row in tpa_df.iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=f"{row['nama']}",
                icon=folium.Icon(color="red", icon="recycle", prefix="fa")
            ).add_to(m)
            folium.map.Marker(
                [row["latitude"], row["longitude"]],
                icon=folium.DivIcon(html=f"<div style='font-size:12px; font-weight:bold; color:red; text-shadow:1px 1px 2px #fff;'>{row['nama']}</div>")
            ).add_to(m)
    
        # Legenda 
        legend_html = """
        <div style="
            position: fixed; 
            bottom: 40px; left: 40px; 
            width: 160px; 
            background-color: rgba(255,255,255,0.9); 
            border: 2px solid grey; 
            z-index: 9999; 
            font-size: 14px; 
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
            border-radius: 8px; 
            padding: 10px; 
            color: black;">
            <div style="margin-bottom:4px;"><i class="fa fa-trash" style="color:green"></i> TPS</div>
            <div><i class="fa fa-recycle" style="color:red"></i> TPA</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, width=1000, height=550)
    
    else:
        # SESUDAH RUTE DICARI 
        selected_tps_df = tps_df[tps_df["id_tps"].astype(str).isin(selected_tps)].copy()
    
        # RUTE GREEDY
        remaining = selected_tps_df.reset_index(drop=True).copy()
        current = remaining.iloc[0]
        route_order = [current]
        remaining = remaining.drop(index=0).reset_index(drop=True)
    
        while not remaining.empty:
            remaining["jarak"] = np.sqrt(
                (remaining["latitude"] - current["latitude"])**2 +
                (remaining["longitude"] - current["longitude"])**2
            ) * 111
            nearest_idx = remaining["jarak"].idxmin()
            nearest = remaining.loc[nearest_idx]
            route_order.append(nearest)
            current = nearest
            remaining = remaining.drop(nearest_idx).reset_index(drop=True)
    
        route = route_order
        last = route[-1]
        nearest_tpa_df = tpa_df.copy()
        nearest_tpa_df["jarak_km"] = np.sqrt(
            (nearest_tpa_df["latitude"] - last["latitude"])**2 +
            (nearest_tpa_df["longitude"] - last["longitude"])**2
        ) * 111
        nearest_tpa = nearest_tpa_df.sort_values("jarak_km").iloc[0]
        truk_ditangani = tpa_truck_map.get(nearest_tpa["nama"], ["Tidak Diketahui"])[0]
    
        # Marker rute (start - finish)
        for i, point in enumerate(route):
            color = "green" if i == 0 else "blue"
            icon_type = "truck" if i == 0 else "trash"
            folium.Marker(
                [point["latitude"], point["longitude"]],
                popup=f"{i+1}. {point['id_tps']}",
                icon=folium.Icon(color=color, icon=icon_type, prefix="fa")
            ).add_to(m)
        
            # Label titik rute
            folium.map.Marker(
                [point["latitude"], point["longitude"]],
                icon=folium.DivIcon(html=f"""
                    <div style='
                        font-size:11px;
                        font-weight:bold;
                        color:green;
                        text-shadow:1px 1px 2px #fff;
                        transform: translate(-50%, 14px);
                    '>
                        {point['id_tps']}
                    </div>
                """)
            ).add_to(m)
        
            # Garis antar TPS
            if i < len(route) - 1:
                next_point = route[i+1]
                folium.PolyLine(
                    [[point["latitude"], point["longitude"]],
                     [next_point["latitude"], next_point["longitude"]]],
                    color="blue", weight=4, opacity=0.8
                ).add_to(m)
        
        # Garis ke TPA terakhir
        last = route[-1]
        folium.PolyLine(
            [[last["latitude"], last["longitude"]],
             [nearest_tpa["latitude"], nearest_tpa["longitude"]]],
            color="red", weight=5,
            tooltip=f"TPS terakhir ➜ {nearest_tpa['nama']}"
        ).add_to(m)
        
        # Marker TPA tujuan + label
        folium.Marker(
            [nearest_tpa["latitude"], nearest_tpa["longitude"]],
            popup=f"{nearest_tpa['nama']}",
            icon=folium.Icon(color="red", icon="flag", prefix="fa")
        ).add_to(m)
        
        folium.map.Marker(
            [nearest_tpa["latitude"], nearest_tpa["longitude"]],
            icon=folium.DivIcon(html=f"""
                <div style='
                    font-size:11px;
                    font-weight:bold;
                    color:red;
                    text-shadow:1px 1px 2px #fff;
                    transform: translate(-50%, 14px);
                '>
                    {nearest_tpa['nama']}
                </div>
            """)
        ).add_to(m)

        # Legenda 
        legend_html = """
        <div style="
            position: fixed; 
            bottom: 40px; left: 40px; 
            width: 180px; 
            background-color: rgba(255,255,255,0.9); 
            border: 2px solid grey; 
            z-index: 9999; 
            font-size: 14px; 
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
            border-radius: 8px; 
            padding: 10px; 
            color: black;">
            <div style="margin-bottom:4px;"><i class="fa fa-truck" style="color:green"></i> Start TPS</div>
            <div><i class="fa fa-flag" style="color:red"></i> Finish TPA</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    
        st_folium(m, width=1000, height=550)

    
        #  INSIGHT  
        segmen_jarak = []
        for i in range(len(route)-1):
            dist = haversine(route[i]["latitude"], route[i]["longitude"],
                             route[i+1]["latitude"], route[i+1]["longitude"])
            segmen_jarak.append({"Dari": route[i]["id_tps"], "Ke": route[i+1]["id_tps"], "Jarak (km)": round(dist, 2)})
    
        dist_to_tpa = haversine(route[-1]["latitude"], route[-1]["longitude"],
                                nearest_tpa["latitude"], nearest_tpa["longitude"])
        segmen_jarak.append({"Dari": route[-1]["id_tps"], "Ke": nearest_tpa["nama"], "Jarak (km)": round(dist_to_tpa, 2)})
    
        total_distance = sum(s["Jarak (km)"] for s in segmen_jarak)
        avg_distance = total_distance / len(segmen_jarak)
        urutan_tps = " ➜ ".join([str(r["id_tps"]) for r in route])
    
        st.markdown("### Insight Rute")
        if len(selected_tps) == 1:
            # RUTE TUNGGAL
            st.write(f"- **Rute direkomendasikan:** {urutan_tps} ➜ {nearest_tpa['nama']}")
            st.write(f"- **Truk menangani:** {truk_ditangani}")
            st.write(f"- **Total jarak tempuh:** {total_distance:.2f} km")
            st.write(f"- **TPA tujuan akhir:** {nearest_tpa['nama']} ({dist_to_tpa:.2f} km dari TPS terakhir)")
        
        else:
            # MULTI RUTE
            st.write(f"- **Rute direkomendasikan:** {urutan_tps} ➜ {nearest_tpa['nama']}")
            st.write(f"- **Truk menangani:** {truk_ditangani}")
            st.write(f"- **Total jarak tempuh:** {total_distance:.2f} km")
            st.write(f"- **Rata-rata jarak antar segmen:** {avg_distance:.2f} km")
            st.write(f"- **TPA tujuan akhir:** {nearest_tpa['nama']} ({dist_to_tpa:.2f} km dari TPS terakhir)")
            
            st.markdown("#### Jarak Antar Segmen Rute")
            st.dataframe(pd.DataFrame(segmen_jarak).style.format({"Jarak (km)": "{:.2f}"}))        
                
        
# MODE: Prediksi Volume Sampah
elif mode == "Prediksi Volume Sampah":
    st.markdown("#### Prediksi Volume Sampah per TPS")

    df = histori_df.copy()
    required_cols = {"tanggal", "id_tps", "latitude", "longitude", "Volume_kg", "kapasitas", "keterisian_%"}
    if df.empty:
        st.error("Dataset histori_rute.csv kosong atau gagal dimuat.")
    elif not required_cols.issubset(set(df.columns)):
        st.error(f"Dataset histori_rute.csv harus memiliki kolom: {', '.join(required_cols)}")
    else:
        # Format tanggal dan sort
        df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
        df = df.dropna(subset=["tanggal", "Volume_kg", "id_tps"]).sort_values("tanggal").reset_index(drop=True)

        # Fitur waktu 
        df["tahun"] = df["tanggal"].dt.year
        df["bulan"] = df["tanggal"].dt.month
        df["bulan_ke"] = (df["tahun"] - df["tahun"].min()) * 12 + (df["bulan"] - df["bulan"].min())
        df["sin_bulan"] = np.sin(2 * np.pi * df["bulan"] / 12)
        df["cos_bulan"] = np.cos(2 * np.pi * df["bulan"] / 12)

        # Statistik TPS
        stats = df.groupby("id_tps")["Volume_kg"].agg(["mean", "std"]).rename(columns={"mean": "tps_mean", "std": "tps_std"})
        df = df.merge(stats, left_on="id_tps", right_index=True, how="left").fillna({"tps_std": 0})

        # Rolling mean dan slope
        df["vol_3m_mean"] = df.groupby("id_tps")["Volume_kg"].transform(lambda x: x.rolling(3, min_periods=1).mean())

        def calc_slope(arr):
            if len(arr) < 2:
                return 0.0
            xs = np.arange(len(arr))
            m, _ = np.polyfit(xs, arr, 1)
            return float(m)

        df["vol_slope"] = df.groupby("id_tps")["Volume_kg"].transform(
            lambda x: x.rolling(3, min_periods=2).apply(calc_slope, raw=True)
        )

        # Encode TPS
        tps_mapping = {tps: i for i, tps in enumerate(df["id_tps"].unique())}
        df["TPS_id"] = df["id_tps"].map(tps_mapping)

        # Fitur model 
        feature_cols = [
            "TPS_id", "kapasitas", "keterisian_%", "latitude", "longitude",
            "tahun", "bulan", "bulan_ke", "sin_bulan", "cos_bulan",
            "tps_mean", "tps_std", "vol_3m_mean", "vol_slope"
        ]

        X = df[feature_cols].fillna(0)
        y = np.log1p(df["Volume_kg"].astype(float))

        # Split train/test
        split_idx = int(len(df) * 0.8)
        if split_idx < 10:
            st.error("Data histori terlalu sedikit untuk pelatihan model.")
        else:
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], df["Volume_kg"].iloc[split_idx:]

            # Model RandomForest 
            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            # Prediksi & Evaluasi
            y_pred = np.expm1(model.predict(X_test))
            y_pred = np.maximum(y_pred, 0.1)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = (abs((y_test - y_pred) / y_test).mean()) * 100

            # Scatter Plot Aktual vs Prediksi
            compare_df = pd.DataFrame({
                "Tanggal": df["tanggal"].iloc[split_idx:].values,
                "id_tps": df["id_tps"].iloc[split_idx:].values,
                "Aktual": y_test.values,
                "Prediksi": y_pred
            })

            fig_comp = px.scatter(compare_df, x="Aktual", y="Prediksi", color="id_tps",
                                  title="Perbandingan Volume Aktual vs Prediksi per TPS",
                                  hover_data=["Tanggal", "id_tps"])
            max_val = max(compare_df["Aktual"].max(), compare_df["Prediksi"].max())
            fig_comp.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
            st.plotly_chart(fig_comp, use_container_width=True)

            # Insight (Grafik 1)
            selisih = abs(compare_df["Aktual"] - compare_df["Prediksi"])
            threshold = selisih.mean() + 2 * selisih.std()
            outlier_mask = selisih > threshold
            outlier_count = outlier_mask.sum()

            # Identifikasi TPS dan tanggal yang outlier
            outlier_df = compare_df[outlier_mask][["Tanggal", "id_tps", "Aktual", "Prediksi"]]

            kualitas = (
                "sangat baik" if r2 >= 0.9 else
                ("baik" if r2 >= 0.75 else
                ("cukup" if r2 >= 0.5 else "perlu perbaikan"))
            )

            st.markdown("### Insight")
            st.write(f"""
            - **Akurasi Model:** R² = {r2:.3f} ({kualitas})
            - **MAE:** {mae:.2f} kg | **MAPE:** {mape:.2f}%
            - **Jumlah Outlier:** {outlier_count}
            - Titik yang sejajar dengan garis diagonal menunjukkan prediksi mendekati aktual.
            Model menunjukkan performa {kualitas}, dengan prediksi volume per TPS relatif akurat dan konsisten.
            """)

            #  daftar titik outlier 
            if outlier_count > 0:
                st.write("**Daftar Titik Outlier (Prediksi jauh dari aktual):**")
                st.dataframe(
                    outlier_df.sort_values("Tanggal").reset_index(drop=True).round(2),
                    use_container_width=True
                )
            else:
                st.success("Tidak ditemukan titik outlier yang signifikan")

            st.markdown("---")

            # Prediksi Volume Sampah Beberapa Bulan ke Depan
            st.subheader("Prediksi Volume Sampah Beberapa Bulan ke Depan")
            
            # Pilih berapa bulan ke depan mau diprediksi
            col_pred1, col_pred2 = st.columns(2)
            n_months = col_pred1.selectbox(
                "Pilih Jumlah Bulan Prediksi",
                [3, 6, 12],
                index=1,
                help="Pilih berapa bulan ke depan yang ingin diprediksi."
            )
            
            # Tentukan periode dari tanggal terakhir data
            last_date = df["tanggal"].max()
            future_months = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=n_months, freq="MS")
            
            start_label = future_months[0].strftime("%b %Y")
            end_label = future_months[-1].strftime("%b %Y")
            
            st.caption(f"Periode prediksi otomatis: **{start_label} – {end_label}**")
            
            #  Data Prediksi
            pred_rows = []
            for tps in df["id_tps"].unique():
                last_data = df[df["id_tps"] == tps].iloc[-3:]
                slope = calc_slope(last_data["Volume_kg"].values)
                last_row = df[df["id_tps"] == tps].iloc[-1]
            
                for d in future_months:
                    bulan_ke = (d.year - df["tahun"].min()) * 12 + (d.month - df["bulan"].min())
                    row = {
                        "TPS_id": tps_mapping[tps],
                        "kapasitas": last_row["kapasitas"],
                        "keterisian_%": last_row["keterisian_%"],
                        "latitude": last_row["latitude"],
                        "longitude": last_row["longitude"],
                        "tahun": d.year,
                        "bulan": d.month,
                        "bulan_ke": bulan_ke,
                        "sin_bulan": np.sin(2 * np.pi * d.month / 12),
                        "cos_bulan": np.cos(2 * np.pi * d.month / 12),
                        "tps_mean": last_row["tps_mean"],
                        "tps_std": last_row["tps_std"],
                        "vol_3m_mean": last_row["vol_3m_mean"],
                        "vol_slope": slope,
                        "tanggal": d,
                        "id_tps": tps
                    }
                    pred_rows.append(row)
            
            future_df = pd.DataFrame(pred_rows)
            X_future = future_df[feature_cols].fillna(0)
            y_future = np.expm1(model.predict(X_future))
            future_df["Prediksi_Volume_kg"] = np.maximum(y_future, 0.1)
            
            # Gabungkan Aktual dan Prediksi
            combined_df = pd.concat([
                df[["tanggal", "id_tps", "Volume_kg"]].rename(columns={"Volume_kg": "Nilai"}),
                future_df[["tanggal", "id_tps", "Prediksi_Volume_kg"]].rename(columns={"Prediksi_Volume_kg": "Nilai"})
            ])
            combined_df["Tipe"] = ["Aktual"] * len(df) + ["Prediksi"] * len(future_df)
            
            # Filter Pilihan TPS & Tipe
            col1, col2 = st.columns(2)
            tps_list = sorted(combined_df["id_tps"].unique())
            selected_tps = col1.selectbox("Pilih TPS", ["Semua"] + tps_list, index=0)
            selected_tipe = col2.selectbox("Tampilkan Data", ["Aktual + Prediksi", "Hanya Prediksi"], index=0)
            
            plot_df = combined_df.copy()
            if selected_tps != "Semua":
                plot_df = plot_df[plot_df["id_tps"] == selected_tps]
            if selected_tipe == "Hanya Prediksi":
                plot_df = plot_df[plot_df["Tipe"] == "Prediksi"]
            
            # Visualisasi Tren
            if selected_tps == "Semua":
                avg_df = plot_df.groupby(["tanggal", "Tipe"])["Nilai"].mean().reset_index()
                fig_future = px.line(
                    avg_df,
                    x="tanggal",
                    y="Nilai",
                    color="Tipe",
                    markers=True,
                    title=f"Tren Rata-rata Volume Sampah (Aktual vs Prediksi {start_label} – {end_label})"
                )
            else:
                fig_future = px.line(
                    plot_df,
                    x="tanggal",
                    y="Nilai",
                    color="Tipe",
                    markers=True,
                    title=f"Tren Volume Sampah {selected_tps} (Aktual vs Prediksi {start_label} – {end_label})"
                )
            
            fig_future.update_traces(line=dict(width=3))
            fig_future.update_layout(
                yaxis_title="Volume Sampah (kg)",
                xaxis_title="Tanggal",
                template="plotly_dark",
                legend_title="Tipe Data"
            )
            
            st.plotly_chart(fig_future, use_container_width=True)
            
            # Ringkasan
            st.write("### Statistik Prediksi")
            st.write(future_df["Prediksi_Volume_kg"].describe())
            
           
            st.markdown("#### Insight")
            
            # Pastikan kolom tanggal sudah bertipe datetime
            histori_df["tanggal"] = pd.to_datetime(histori_df["tanggal"])
            future_df["tanggal"] = pd.to_datetime(future_df["tanggal"])
            
            # Tentukan periode prediksi
            pred_start = future_df["tanggal"].min()
            pred_end = future_df["tanggal"].max()
            periode_pred_awal = pred_start.strftime("%b %Y")
            periode_pred_akhir = pred_end.strftime("%b %Y")
            
            # Hitung jumlah bulan prediksi
            months_diff = (pred_end.year - pred_start.year) * 12 + (pred_end.month - pred_start.month) + 1
            
            # Tentukan periode historis dengan panjang waktu yang sama
            hist_end = pred_start - pd.offsets.MonthEnd(1)
            hist_start = hist_end - pd.DateOffset(months=months_diff - 1)
            
            # Ambil data aktual dari periode historis yang sama panjang
            actual_df = histori_df[(histori_df["tanggal"] >= hist_start) & (histori_df["tanggal"] <= hist_end)].copy()
            pred_df = future_df.copy()
            
            # Update label periode aktual
            periode_hist_awal = hist_start.strftime("%b %Y")
            periode_hist_akhir = hist_end.strftime("%b %Y")
            
            # Hitung rata-rata volume aktual dan prediksi
            avg_actual = actual_df["Volume_kg"].mean() if not actual_df.empty else None
            avg_pred = pred_df["Prediksi_Volume_kg"].mean() if not pred_df.empty else None

            
            # Hitung tren total prediksi
            if not pred_df.empty and not actual_df.empty:
                # Hitung rata-rata harian
                mean_pred = pred_df["Prediksi_Volume_kg"].mean()
                mean_actual = actual_df["Volume_kg"].mean()
            
                # Normalisasi berdasarkan jumlah hari yang sama
                days_pred = (pred_end - pred_start).days + 1
                total_pred_norm = mean_pred * days_pred
                total_actual_norm = mean_actual * days_pred  # pakai jumlah hari yang sama
            
                diff = total_pred_norm - total_actual_norm
                trend_status = "meningkat" if diff > 0 else "menurun"
                mean_diff = mean_pred - mean_actual
            
                # Tampilkan hasil
                st.write(
                    f"Selama periode **{months_diff} bulan ({periode_pred_awal} – {periode_pred_akhir})**, "
                    f"total volume sampah kota diprediksi **{trend_status} sebesar {abs(diff):,.2f} kg** "
                    f"(setelah dinormalisasi berdasarkan jumlah hari yang sama) "
                    f"dibandingkan total volume periode sebelumnya "
                    f"(**{periode_hist_awal} – {periode_hist_akhir}**)."
                )
            
                st.write(
                    f"- Rata-rata volume aktual ({periode_hist_awal} – {periode_hist_akhir}): **{mean_actual:.2f} kg/hari**"
                )
                st.write(
                    f"- Rata-rata volume prediksi ({periode_pred_awal} – {periode_pred_akhir}): **{mean_pred:.2f} kg/hari**"
                )
            
                if mean_diff > 0:
                    st.write(
                        f"- Volume prediksi rata-rata **{mean_diff:.2f} kg lebih tinggi** dibandingkan periode sebelumnya."
                    )
                elif mean_diff < 0:
                    st.write(
                        f"- Volume prediksi rata-rata **{abs(mean_diff):.2f} kg lebih rendah** dibandingkan periode sebelumnya."
                    )
                else:
                    st.write("- Volume prediksi rata-rata sama dengan periode sebelumnya.")

            st.markdown("---")
            #  Ringkasan Prediksi Bulanan
            if not pred_df.empty:
                pred_df["tanggal"] = pd.to_datetime(pred_df["tanggal"])
            
                monthly_summary = (
                    pred_df.groupby(pred_df["tanggal"].dt.to_period("M"))["Prediksi_Volume_kg"]
                    .agg(total_volume="sum", avg_daily_volume="mean")
                    .reset_index()
                )
            
                monthly_summary["bulan"] = monthly_summary["tanggal"].dt.to_timestamp()
                monthly_summary["selisih"] = monthly_summary["total_volume"].diff()
            
                st.markdown("#### Tabel Ringkasan Prediksi per Bulan")
                st.dataframe(
                    monthly_summary[["bulan", "total_volume", "avg_daily_volume", "selisih"]].round(2),
                    use_container_width=True
                )
            
                # Hitung bulan dengan kenaikan dan penurunan terbesar
                max_increase_idx = monthly_summary["selisih"].idxmax()
                max_decrease_idx = monthly_summary["selisih"].idxmin()
                
                bulan_max_inc = monthly_summary.loc[max_increase_idx, "bulan"].strftime("%B %Y")
                nilai_max_inc = monthly_summary.loc[max_increase_idx, "selisih"]
                
                bulan_max_dec = monthly_summary.loc[max_decrease_idx, "bulan"].strftime("%B %Y")
                nilai_max_dec = monthly_summary.loc[max_decrease_idx, "selisih"]
                
                # Tampilkan insight singkat
                st.markdown("#### Insight ")
                
                if nilai_max_inc > 0:
                    st.write(
                        f"Kenaikan terbesar diproyeksikan terjadi pada **{bulan_max_inc}**, "
                        f"naik sebesar **{nilai_max_inc:,.2f} kg** dibanding bulan sebelumnya."
                    )
                
                if nilai_max_dec < 0:
                    st.write(
                        f"Penurunan terbesar diperkirakan pada **{bulan_max_dec}**, "
                        f"turun sekitar **{abs(nilai_max_dec):,.2f} kg** dibanding bulan sebelumnya."
                    )
                 st.markdown("---")   
                 #  Top 5 TPS Berdasarkan Prediksi
                if not future_df.empty:
                    high_pred = (
                        future_df.groupby("id_tps")["Prediksi_Volume_kg"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(5)
                    )
                
                    next_month = future_df["tanggal"].min()
                    high_pred_next = (
                        future_df[future_df["tanggal"] == next_month]
                        .groupby("id_tps")["Prediksi_Volume_kg"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(5)
                    )
                
                    periode_awal = future_df["tanggal"].min().strftime("%b %Y")
                    periode_akhir = future_df["tanggal"].max().strftime("%b %Y")
                
                    st.markdown("### Top TPS Berdasarkan Prediksi Volume")
                    colA, colB = st.columns(2)
                
                    with colA:
                        st.write(f"**Top 5 TPS dengan Rata-rata Prediksi Tertinggi ({periode_awal} – {periode_akhir}):**")
                        st.dataframe(
                            high_pred.reset_index().rename(
                                columns={"id_tps": "TPS", "Prediksi_Volume_kg": "Rata-rata Prediksi (kg)"}
                            ).round(2),
                            use_container_width=True
                        )
                        st.caption(f"Periode rata-rata mencakup seluruh prediksi: {periode_awal} – {periode_akhir}")
                
                    with colB:
                        st.write(f"**Top 5 TPS Bulan {next_month.strftime('%B %Y')}:**")
                        st.dataframe(
                            high_pred_next.reset_index().rename(
                                columns={"id_tps": "TPS", "Prediksi_Volume_kg": "Prediksi Bulan Depan (kg)"}
                            ).round(2),
                            use_container_width=True
                        )
                        st.caption(f"Data ini menunjukkan prediksi untuk bulan terdekat: {next_month.strftime('%B %Y')}")
                
                    # Tambahan ringkasan TPS tertinggi
                    top_tps_pred = (
                        future_df.groupby("id_tps")["Prediksi_Volume_kg"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(5)
                    )
                     
                    st.markdown("#### Insight ")
                    st.write(f"- TPS dengan volume prediksi tertinggi: **{', '.join(top_tps_pred.index)}**.")



  
                

                
            
            
    










































































































































































































