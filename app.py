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

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Analisis Big Data - Rute TPS–TPA", layout="wide")
st.title("Sistem Analisis Rute & Pengumpulan Sampah Kota Delhi")
st.markdown("Analitik dan optimasi rute pengangkutan sampah berbasis **Big Data**.")

# MUAT DATA 
def safe_read_csv(path, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat {path}: {e}")
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
    
# NAVIGASI
st.sidebar.title("Navigasi")
mode = st.sidebar.radio(
    "Pilih Menu:",
    [
        "Dashboard Data",
        "Rute Pengangkutan",
        "Jadwal Pengangkutan",
        "Prediksi Volume Sampah"
    ],
    index=0
)
st.sidebar.markdown("---")

# MODE: Dashboard Data 
if mode == "Dashboard Data":
    st.header("Dashboard Pemantauan TPS & TPA")

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

    #  PETA SEBARAN TPS & TPA
    st.subheader("Peta Sebaran Lokasi TPS dan TPA")
    
    # Filter TPS
    tps_options_map = sorted(tps_df["id_tps"].astype(str).unique().tolist())
    selected_tps_map = st.multiselect(
        "Filter TPS:",
        tps_options_map,
        key="filter_tps_map"
    )
    
    if st.button("Reset Filter Peta", key="reset_peta"):
        selected_tps_map = []

    if selected_tps_map:
        filtered_tps_map = tps_df[tps_df["id_tps"].astype(str).isin(selected_tps_map)].copy()
    else:
        filtered_tps_map = tps_df.copy()
    
    # Pastikan koordinat numerik
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
    
        # Label di samping kanan marker
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
    
    # Tampilkan peta
    st_folium(m, width=1000, height=550)
    st.markdown("---")

    # SCATTER: Kapasitas vs Volume
    st.subheader("Hubungan Kapasitas vs Volume per TPS")

    tps_options_scatter = sorted(tps_df["id_tps"].astype(str).unique().tolist())
    selected_tps_scatter = st.multiselect(
        "Filter TPS:",
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
        fig_scatter = px.scatter(
            tps_filtered_scatter,
            x="kapasitas",
            y="volume_saat_ini",
            color="keterisian_%",
            size="keterisian_%",
            hover_name="id_tps",
            color_continuous_scale="RdYlGn_r",
            title="Kapasitas vs Volume Aktual TPS"
        )
        max_val = max(
            tps_filtered_scatter["kapasitas"].max(),
            tps_filtered_scatter["volume_saat_ini"].max()
        )
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                              line=dict(color="gray", dash="dash"))
        fig_scatter.add_annotation(x=max_val*0.7, y=max_val*0.9,
                                   text="Volume = Kapasitas", showarrow=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Tidak ada data untuk Scatter (TPS tidak dipilih).")

    st.markdown("### Insight Hubungan Kapasitas vs Volume per TPS")
    
    if not tps_filtered_scatter.empty:
        # Ambang dinamis
        threshold = st.slider(
            "Atur ambang keterisian (%) untuk peringatan penuh:",
            50, 100, 85, step=1, key="slider_threshold_scatter"
        )
    
        # Hitung nilai dan kelompokkan
        avg_fill = tps_filtered_scatter["keterisian_%"].mean()
        penuh = tps_filtered_scatter[tps_filtered_scatter["keterisian_%"] >= threshold]
        hampir = tps_filtered_scatter[
            (tps_filtered_scatter["keterisian_%"] >= threshold - 10) &
            (tps_filtered_scatter["keterisian_%"] < threshold)
        ]
    
        # Tampilkan insight utama
        st.write(f"- Rata-rata keterisian TPS (terfilter): **{avg_fill:.1f}%**")
    
        if not penuh.empty:
            st.warning(
                f"🚨 {len(penuh)} TPS melebihi ambang {threshold}%:"
                f" {', '.join(penuh['id_tps'].astype(str))}"
            )
        elif not hampir.empty:
            st.info(
                f"⚠️ {len(hampir)} TPS mendekati ambang ({threshold-10}–{threshold}%):"
                f" {', '.join(hampir['id_tps'].astype(str))}"
            )
        else:
            st.success(f"✅ Semua TPS masih di bawah {threshold-10}% keterisian.")
    
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
    selected_tps_top5 = st.multiselect("Filter TPS:", tps_options_top5, key="filter_tps_top5")

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
        "Filter Kriteria Peringkat:",
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
    selected_tps_tren = st.multiselect("Filter TPS:", tps_options_tren, key="filter_tps_tren")
    
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

# MODE: Simulasi Rute
elif mode == "Rute Pengangkutan":
    st.header("Rute Pengangkutan")
    st.write("Pilih satu atau beberapa TPS untuk mensimulasikan rute otomatis menuju TPA terdekat.")

    if "id_tps" not in tps_df.columns or tps_df.empty:
        st.error("Kolom 'id_tps' tidak ditemukan di data TPS atau data TPS kosong.")
    else:
        tps_options = tps_df["id_tps"].astype(str).unique().tolist()
        selected_tps = st.multiselect("Pilih TPS", tps_options)

        if not selected_tps:
            st.info("Silakan pilih minimal satu TPS untuk simulasi rute.")
        else:
         # RUTE TUNGGAL
            if len(selected_tps) == 1:
                tps_point = tps_df[tps_df["id_tps"].astype(str) == selected_tps[0]].iloc[0]
            
                if tpa_df.empty or "latitude" not in tpa_df.columns:
                    st.error("Data TPA tidak tersedia untuk menghitung rute.")
                else:
                    # Hitung jarak ke semua TPA 
                    tpa_df = tpa_df.copy()
                    tpa_df["jarak_km"] = np.sqrt(
                        (tpa_df["latitude"] - tps_point["latitude"])**2 +
                        (tpa_df["longitude"] - tps_point["longitude"])**2
                    ) * 111
            
                    rekomendasi_tpa = tpa_df.sort_values("jarak_km").iloc[0]
                    nama_tpa_terdekat = rekomendasi_tpa.get("nama", "-")
                    jarak_terpendek = rekomendasi_tpa["jarak_km"]
                    waktu_menit = jarak_terpendek / 40 * 60  
            
                    st.markdown(f"#### TPA Terdekat: {nama_tpa_terdekat}")
                    st.metric("Jarak Terpendek (km)", f"{jarak_terpendek:.2f}")
                    st.metric("Estimasi Waktu Tempuh (menit)", f"{waktu_menit:.1f}")
            
                    # PETA
                    center_lat = float((tps_point["latitude"] + rekomendasi_tpa["latitude"]) / 2)
                    center_lon = float((tps_point["longitude"] + rekomendasi_tpa["longitude"]) / 2)
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
                     # Marker TPS
                    tps_lat, tps_lon = tps_point["latitude"], tps_point["longitude"]
                    folium.Marker(
                        [tps_lat, tps_lon],
                        popup=f"{tps_point.get('id_tps')}</b>",
                        icon=folium.Icon(color="green", icon="trash", prefix="fa")
                    ).add_to(m)
                
                    folium.map.Marker(
                        [tps_lat, tps_lon],
                        icon=folium.DivIcon(
                            html=f"""
                            <div style="
                                font-size: 14px;
                                font-weight: bold;
                                color: green;
                                text-shadow: 1px 1px 2px #fff;
                                text-align: center;
                                transform: translate(10px, -25px);
                            ">
                                {tps_point.get('id_tps')}
                            </div>
                            """
                        )
                    ).add_to(m)
                    
                    
                    # Marker TPA
                    for _, row in tpa_df.iterrows():
                        lat = row.get("latitude")
                        lon = row.get("longitude")
                        if pd.isna(lat) or pd.isna(lon):
                            continue
                    
                        folium.Marker(
                            [lat, lon],
                            popup=f"{row.get('nama','-')}</b>",
                            icon=folium.Icon(color="red", icon="recycle", prefix="fa")
                        ).add_to(m)
                    
                        folium.map.Marker(
                            [lat, lon],
                            icon=folium.DivIcon(
                                html=f"""
                                <div style="
                                    font-size: 14px;
                                    font-weight: bold;
                                    color: red;
                                    text-shadow: 1px 1px 2px #fff;
                                    text-align: center;
                                    transform: translate(10px, -25px);
                                ">
                                    {row.get('nama')}
                                </div>
                                """
                            )
                        ).add_to(m)

                    # Garis Jalur 
                    folium.PolyLine(
                        [[tps_point["latitude"], tps_point["longitude"]],
                         [rekomendasi_tpa["latitude"], rekomendasi_tpa["longitude"]]],
                        color="blue",
                        weight=5,
                        tooltip=f"Rute: TPS {tps_point.get('id_tps')} ➜ TPA {nama_tpa_terdekat}"
                    ).add_to(m)
            
                    # Tampilkan Peta 
                    st_folium(m, width=1000, height=550)
            
                    #  Insight
                    st.markdown("### Insight Rute")
                    st.write(f"- Jalur terpendek dari **{selected_tps[0]} ➜ {nama_tpa_terdekat}** sejauh **{jarak_terpendek:.2f} km**.")
                    st.write(f"- Estimasi waktu tempuh: **{waktu_menit:.1f} menit**.")

            # RUTE MULTI (GREEDY)
            else:
                st.subheader("Simulasi Multi-Rute")
                selected_tps_df = tps_df[tps_df["id_tps"].astype(str).isin(selected_tps)].copy()
            
                if selected_tps_df.empty:
                    st.error("Data TPS terpilih tidak tersedia.")
                else:
                    # Greedy route: mulai dari TPS pertama yang dipilih
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
            
                    # Hitung total jarak antar TPS
                    total_distance = 0.0
                    for i in range(len(route) - 1):
                        lat1, lon1 = route[i]["latitude"], route[i]["longitude"]
                        lat2, lon2 = route[i+1]["latitude"], route[i+1]["longitude"]
                        total_distance += np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111
            
                    # Hubungkan ke TPA terdekat dari TPS terakhir
                    last = route[-1]
                    if not tpa_df.empty:
                        tpa_df = tpa_df.copy()
                        tpa_df["jarak_km"] = np.sqrt(
                            (tpa_df["latitude"] - last["latitude"])**2 +
                            (tpa_df["longitude"] - last["longitude"])**2
                        ) * 111
                        nearest_tpa = tpa_df.sort_values("jarak_km").iloc[0]
                        total_distance += nearest_tpa["jarak_km"]
                    else:
                        nearest_tpa = {"nama": "-", "jarak_km": 0.0}
            
                    avg_cap = selected_tps_df["kapasitas"].mean() if "kapasitas" in selected_tps_df.columns else 0
            
                    # VISUALISASI PETA
                    center_lat = float(selected_tps_df["latitude"].mean()) if "latitude" in selected_tps_df.columns else 0
                    center_lon = float(selected_tps_df["longitude"].mean()) if "longitude" in selected_tps_df.columns else 0
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
                    # Tambahkan Marker dan Label 
                    for i, point in enumerate(route):
                        lat = point.get("latitude")
                        lon = point.get("longitude")
                        if pd.isna(lat) or pd.isna(lon):
                            continue
                    
                        # Icon
                        if i == 0:
                            icon_type = "truck"   
                            color = "green"
                        else:
                            icon_type = "trash"
                            color = "blue"
                    
                        folium.Marker(
                            [lat, lon],
                            popup=f"{i+1}.{point.get('id_tps','-')}",
                            icon=folium.Icon(color=color, icon=icon_type, prefix="fa")
                        ).add_to(m)
                    
                        # Tambahkan label 
                        folium.map.Marker(
                            [lat, lon],
                            icon=folium.DivIcon(
                                html=f"<div style='font-size:14px; font-weight:bold; color:#003366; text-shadow:1px 1px 2px #fff;'>{point.get('id_tps')}</div>"
                            )
                        ).add_to(m)
            
                        # Garis antar titik TPS
                        if i < len(route) - 1:
                            next_point = route[i+1]
                            folium.PolyLine(
                                [[point["latitude"], point["longitude"]],
                                 [next_point["latitude"], next_point["longitude"]]],
                                color="blue", weight=4, opacity=0.8,
                                tooltip=f"{point.get('id_tps')} ➜ {next_point.get('id_tps')}"
                            ).add_to(m)
            
                    # Hubungkan ke TPA terdekat
                    if not tpa_df.empty:
                        folium.PolyLine(
                            [[last["latitude"], last["longitude"]],
                             [nearest_tpa["latitude"], nearest_tpa["longitude"]]],
                            color="red", weight=5, tooltip=f"TPS terakhir ➜ {nearest_tpa.get('nama')}"
                        ).add_to(m)
                        folium.Marker(
                            [nearest_tpa["latitude"], nearest_tpa["longitude"]],
                            popup=f"{nearest_tpa.get('nama')}",
                            icon=folium.Icon(color="red", icon="flag", prefix="fa")
                        ).add_to(m)
                        folium.map.Marker(
                            [nearest_tpa["latitude"], nearest_tpa["longitude"]],
                            icon=folium.DivIcon(
                                html=f"<div style='font-size:14px; font-weight:bold; color:red; text-shadow:1px 1px 2px #fff;'>{nearest_tpa.get('nama')}</div>"
                            )
                        ).add_to(m)
            
                    #  Legenda
                    legend_html = """
                    <div style="
                        position: fixed; 
                        bottom: 40px; left: 40px; 
                        width: 120px; 
                        background-color: rgba(255,255,255,0.9); 
                        border:2px solid grey; 
                        z-index:9999; 
                        font-size:14px; 
                        box-shadow:2px 2px 6px rgba(0,0,0,0.3); 
                        border-radius:8px; 
                        padding:10px; 
                        color:black;">
                        <i class="fa fa-truck fa-lg" style="color:green"></i> Start TPS<br>
                        <i class="fa fa-flag fa-lg" style="color:red"></i> TPA Finish
                    </div>
                    """
                    
                    m.get_root().html.add_child(folium.Element(legend_html))


                    # Tampilkan peta
                    st_folium(m, width=1000, height=550)
            
              # -Fungsi menghitung jarak antar dua koordinat (Haversine) 
                    def haversine(lat1, lon1, lat2, lon2):
                        from math import radians, sin, cos, sqrt, atan2
                        R = 6371.0  # radius bumi dalam km
                        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                        dlon, dlat = lon2 - lon1, lat2 - lat1
                        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
                        c = 2 * atan2(sqrt(a), sqrt(1 - a))
                        return R * c
                    
                    # Pastikan nearest_tpa berbentuk dictionary aman
                    if nearest_tpa is not None:
                        if isinstance(nearest_tpa, pd.DataFrame):
                            if not nearest_tpa.empty:
                                nearest_tpa = nearest_tpa.iloc[0].to_dict()
                            else:
                                nearest_tpa = {"nama": "-", "latitude": 0.0, "longitude": 0.0, "jarak_km": 0.0}
                        elif hasattr(nearest_tpa, "to_dict"):
                            nearest_tpa = nearest_tpa.to_dict()
                        elif not isinstance(nearest_tpa, dict):
                            nearest_tpa = {"nama": "-", "latitude": 0.0, "longitude": 0.0, "jarak_km": 0.0}
                    else:
                        nearest_tpa = {"nama": "-", "latitude": 0.0, "longitude": 0.0, "jarak_km": 0.0}
                    
                    # Hitung jarak antar segmen 
                    segmen_jarak = []
                    for i in range(len(route) - 1):
                        tps_a = route[i]
                        tps_b = route[i + 1]
                        dist = haversine(tps_a["latitude"], tps_a["longitude"],
                                         tps_b["latitude"], tps_b["longitude"])
                        segmen_jarak.append({
                            "from": tps_a["id_tps"],
                            "to": tps_b["id_tps"],
                            "distance": dist
                        })
                    
                    # Tambahkan jarak terakhir ke TPA
                    last_tps = route[-1]
                    dist_to_tpa = haversine(last_tps["latitude"], last_tps["longitude"],
                                            nearest_tpa["latitude"], nearest_tpa["longitude"])
                    segmen_jarak.append({
                        "from": last_tps["id_tps"],
                        "to": nearest_tpa["nama"],
                        "distance": dist_to_tpa
                    })
                    
                    # Hitung total jarak secara konsisten
                    total_distance = sum(s["distance"] for s in segmen_jarak)
                    avg_distance = total_distance / len(segmen_jarak)  # rata-rata antar segmen
                    
                    # insight & rekomendasi 
                    urutan_tps = " ➜ ".join([str(r.get("id_tps")) for r in route])
                    st.markdown("### Insight & Rekomendasi Multi-Rute")
                    st.write(f"- **Rute terpendek yang direkomendasikan:** {urutan_tps} ➜ {nearest_tpa.get('nama','-')}")
                    st.write(f"- **Total jarak tempuh:** {total_distance:.2f} km untuk {len(selected_tps)} TPS.")
                    st.write(f"- **Rata-rata jarak antar segmen:** {avg_distance:.2f} km.")
                    st.write(f"- **TPA tujuan akhir:** {nearest_tpa.get('nama','-')} ({dist_to_tpa:.2f} km dari TPS terakhir).")
                    
                    #  Detail jarak antar segmen 
                    st.markdown("#### Jarak Antar Segmen Rute:")
                    for seg in segmen_jarak:
                        st.write(f"- {seg['from']} ➜ {seg['to']}: **{seg['distance']:.2f} km**")

# MODE: Jadwal Otomatis 
elif mode == "Jadwal Pengangkutan":
    st.header("Jadwal Otomatis Berdasarkan Aktivitas TPS")

    num_truck = st.number_input("Jumlah Truk Operasional", min_value=1, value=2)
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]

    if st.button("Buat Jadwal Otomatis"):
        data = []
        for i, h in enumerate(hari):
            n = min(num_truck, len(tps_df)) if not tps_df.empty else 0
            if n == 0:
                continue
            tps_hari = tps_df.sample(n)
            for j, t in enumerate(tps_hari.itertuples()):
                data.append({
                    "Hari": h,
                    "Truk": f"Truk {j+1}",
                    "ID_TPS": getattr(t, "id_tps", None),
                    "TPA": getattr(t, "nearest_tpa", None),
                    "Jarak (km)": getattr(t, "nearest_dist_km", None)
                })
        jadwal_df = pd.DataFrame(data)
        if jadwal_df.empty:
            st.info("Tidak ada jadwal yang dapat dibuat (data TPS kosong).")
        else:
            st.dataframe(jadwal_df, use_container_width=True)
            if "Jarak (km)" in jadwal_df.columns and not jadwal_df["Jarak (km)"].isna().all():
                fig = px.bar(jadwal_df, x="Hari", y="Jarak (km)", color="Truk",
                             barmode="group", title="Distribusi Jarak per Hari per Truk")
                st.plotly_chart(fig, use_container_width=True)

            # Insight Jadwal
            st.markdown("### Insight:")
            jarak_avg = jadwal_df["Jarak (km)"].mean() if "Jarak (km)" in jadwal_df.columns else 0
            st.info(f"• **Rata-rata jarak per rute:** {jarak_avg:.2f} km")
            try:
                truk_terjauh = jadwal_df.groupby("Truk")["Jarak (km)"].sum().idxmax()
                st.write(f"• **Truk dengan jarak tempuh tertinggi:** {truk_terjauh}")
            except Exception:
                st.write("• Tidak dapat menentukan truk terjauh.")
            st.write("• **Saran:** Rotasi truk agar beban kerja lebih merata tiap hari.")

# MODE: Prediksi Volume Sampah
elif mode == "Prediksi Volume Sampah":
    st.header("Prediksi Volume Sampah per TPS")

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
                ("cukup" if r2 >= 0.5 else "perlu perbaikan ⚠️"))
            )

            st.markdown("### Insight")
            st.write(f"""
            - **Akurasi Model:** R² = {r2:.3f} ({kualitas})
            - **MAE:** {mae:.2f} kg | **MAPE:** {mape:.2f}%
            - **Jumlah Outlier:** {outlier_count}
            - Titik yang sejajar dengan garis diagonal menunjukkan prediksi mendekati aktual.
            Model menunjukkan performa {kualitas}, dengan prediksi volume per TPS relatif akurat dan konsisten.
            """)

            # tampilkan daftar titik outlier 
            if outlier_count > 0:
                st.write("**📍 Daftar Titik Outlier (Prediksi jauh dari aktual):**")
                st.dataframe(
                    outlier_df.sort_values("Tanggal").reset_index(drop=True).round(2),
                    use_container_width=True
                )
            else:
                st.success("Tidak ditemukan titik outlier yang signifikan 👍")

            st.markdown("---")

            # Prediksi 6 Bulan ke Depan
            st.subheader("Prediksi Volume Sampah 6 Bulan ke Depan (Jan–Jun 2021)")

            future_months = pd.date_range("2021-01-01", periods=6, freq="MS")

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

            combined_df = pd.concat([
                df[["tanggal", "id_tps", "Volume_kg"]].rename(columns={"Volume_kg": "Nilai"}),
                future_df[["tanggal", "id_tps", "Prediksi_Volume_kg"]].rename(columns={"Prediksi_Volume_kg": "Nilai"})
            ])
            combined_df["Tipe"] = ["Aktual"] * len(df) + ["Prediksi"] * len(future_df)

            #  Filter 
            col1, col2 = st.columns(2)
            tps_list = sorted(combined_df["id_tps"].unique())
            selected_tps = col1.selectbox("Pilih TPS", ["Semua"] + tps_list, index=0)
            selected_tipe = col2.selectbox("Tampilkan Data", ["Aktual + Prediksi", "Hanya Prediksi"], index=0)

            plot_df = combined_df.copy()
            if selected_tps != "Semua":
                plot_df = plot_df[plot_df["id_tps"] == selected_tps]
            if selected_tipe == "Hanya Prediksi":
                plot_df = plot_df[plot_df["Tipe"] == "Prediksi"]

            #  Visualisasi Tren 
            if selected_tps == "Semua":
                avg_df = plot_df.groupby(["tanggal", "Tipe"])["Nilai"].mean().reset_index()
                fig_future = px.line(
                    avg_df,
                    x="tanggal",
                    y="Nilai",
                    color="Tipe",
                    markers=True,
                    title="Tren Rata-rata Volume Sampah (Aktual vs Prediksi Jan–Jun 2021)"
                )
            else:
                fig_future = px.line(
                    plot_df,
                    x="tanggal",
                    y="Nilai",
                    color="Tipe",
                    markers=True,
                    title=f"Tren Volume Sampah {selected_tps} (Aktual vs Prediksi Jan–Jun 2021)"
                )

            fig_future.update_traces(line=dict(width=3))
            st.plotly_chart(fig_future, use_container_width=True)
                 
            st.write("Ringkasan Prediksi:")
            st.dataframe(future_df.head(10))

            st.write("Statistik Prediksi:")
            st.write(future_df["Prediksi_Volume_kg"].describe())
            
            high_pred = future_df.groupby("id_tps")["Prediksi_Volume_kg"].mean().sort_values(ascending=False).head(10)
            st.write("**Top TPS dengan Prediksi Volume Tertinggi (Jan–Jun 2021):**")
            st.dataframe(high_pred.reset_index().rename(
                columns={"id_tps": "TPS", "Prediksi_Volume_kg": "Rata-rata Prediksi (kg)"}
            ).round(2), use_container_width=True)  

           #  Insight grafik kedua 
            st.markdown("#### Insight")

            # Pastikan kolom tanggal datetime
            histori_df ["tanggal"] = pd.to_datetime(histori_df["tanggal"])
            future_df["tanggal"] = pd.to_datetime(future_df["tanggal"])

            # Pisahkan data aktual Jan–Des 2020 dari HISTORI RUTE
            actual_df = histori_df[
                (histori_df["tanggal"].dt.year == 2020)
            ]

            # Pisahkan data prediksi Jan–Jun 2021 dari FUTURE_DF
            pred_df = future_df[
                (future_df["tanggal"].dt.year == 2021)
                & (future_df["tanggal"].dt.month <= 6)
            ]

            #  Hitung rata-rata 
            avg_actual = actual_df["Volume_kg"].mean() if not actual_df.empty else None
            avg_pred = (
                pred_df["Prediksi_Volume_kg"].mean()
                if "Prediksi_Volume_kg" in pred_df.columns and not pred_df.empty
                else None
            )

            #  Hitung tren total prediksi 
            if not pred_df.empty:
                trend_total = pred_df.groupby("tanggal")["Prediksi_Volume_kg"].sum().reset_index()
                diff = trend_total["Prediksi_Volume_kg"].iloc[-1] - trend_total["Prediksi_Volume_kg"].iloc[0]
                trend_status = "meningkat" if diff > 0 else "menurun"
            else:
                trend_status = "tidak tersedia"
                diff = 0  

            st.write(f"- Total volume sampah kota diprediksi **{trend_status}** hingga Juni 2021 (selisih {diff:.2f} kg).")

            if avg_actual is not None:
                st.write(f"- Rata-rata volume aktual (Jan–Des 2020): **{avg_actual:.2f} kg/hari**")
            else:
                st.write("- Data volume aktual (Jan–Des 2020) tidak tersedia.")

            if avg_pred is not None:
                st.write(f"- Rata-rata volume prediksi (Jan–Jun 2021): **{avg_pred:.2f} kg/hari**")

            if avg_actual is not None and avg_pred is not None:
                selisih = avg_pred - avg_actual
                arah = "lebih tinggi" if selisih > 0 else "lebih rendah"
                st.write(f"- Volume prediksi rata-rata **{abs(selisih):.2f} kg {arah}** dibandingkan volume aktual tahun sebelumnya.")
            high_pred = future_df.groupby("id_tps")["Prediksi_Volume_kg"].mean().sort_values(ascending=False).head(3)
            st.write(f"- TPS dengan volume tertinggi diprediksi: **{', '.join(high_pred.index)}**.")
            
            st.markdown("---")
            st.subheader("Rekomendasi")
            top3 = high_pred.head(3).index.tolist()
            st.write(f"- **Prioritaskan pengangkutan** di TPS: **{', '.join(top3)}** bulan depan.")
            st.write("- **Atur kapasitas TPA** untuk lonjakan volume dari TPS tersebut.")
            st.write("- **Pantau tren kenaikan vol_slope** sebagai indikasi peningkatan aktivitas.")
            
            
    











































































