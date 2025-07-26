# streamlit_app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
import io
import matplotlib.pyplot as plt
import numpy as np

# Streamlit configuration - must be first
st.set_page_config(layout="wide", page_title="Comprehensive Pollution Data Analyzer")

# Only import the analyzer when needed (lazy loading)
@st.cache_resource
def get_analyzer():
    try:
        from pollution_analyzer_logic import PollutionDataAnalyzer
        return PollutionDataAnalyzer()
    except ImportError:
        st.error("pollution_analyzer_logic module not found. Using mock analyzer.")
        return None

# Cache sample geographic data generation
@st.cache_data
def generate_sample_geo_data(n_points=50):
    """Generate cached sample geographic data"""
    np.random.seed(42)  # For reproducible results
    center_lat, center_lon = 40.7128, -74.0060
    
    lats = np.random.normal(center_lat, 0.05, n_points)
    lons = np.random.normal(center_lon, 0.05, n_points)
    pollutions = np.random.gamma(2, 15, n_points)
    
    return pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'pollution': pollutions
    })

st.title("🌬️ Comprehensive Pollution Data Analysis Platform")
st.markdown("**Time Series Analysis with Romberg Method + Geographic Visualization**")

# Create tabs for different analysis types
tab1, tab2 = st.tabs(["📈 Time Series Analysis", "🗺️ Geographic Visualization"])

# =====================================================
# TAB 1: TIME SERIES ANALYSIS (Optimized)
# =====================================================
with tab1:
    # Only load analyzer when tab1 is accessed
    analyzer = get_analyzer()
    
    if analyzer is None:
        st.error("Time series analysis not available - missing analyzer module")
        st.stop()
    
    # --- Data Loading Section ---
    st.header("1. Load Time Series Pollution Data")

    data_source = st.radio(
        "Choose data source:",
        ("Generate Sample Data", "Upload CSV File"),
        key="data_source_timeseries"
    )

    # Use session state to avoid regenerating data unnecessarily
    if 'sample_data_generated' not in st.session_state:
        st.session_state.sample_data_generated = False

    if data_source == "Generate Sample Data":
        if not st.session_state.sample_data_generated:
            with st.spinner("Generating sample data..."):
                analyzer.generate_sample_data(days=30)
                st.session_state.sample_data_generated = True
        
        st.success("Sample data ready!")
        if analyzer.data_points:
            with st.expander("View Sample Data", expanded=False):
                df_display = pd.DataFrame({
                    'Hour': analyzer.time_points[:50],  # Limit display
                    'Concentration (µg/m³)': analyzer.data_points[:50]
                })
                st.dataframe(df_display.set_index('Hour'))

    elif data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload your time series CSV file", type="csv", key="timeseries_upload")
        if uploaded_file is not None:
            try:
                with st.spinner("Loading CSV data..."):
                    df_uploaded = pd.read_csv(uploaded_file)
                    if 'Concentration_ugm3' in df_uploaded.columns:
                        analyzer.data_points = df_uploaded['Concentration_ugm3'].tolist()
                        analyzer.time_points = list(range(len(analyzer.data_points)))
                        st.success(f"Successfully loaded {len(analyzer.data_points)} data points from CSV.")
                        
                        with st.expander("View Uploaded Data", expanded=False):
                            st.dataframe(df_uploaded.head())
                    else:
                        st.error("CSV must contain a column named 'Concentration_ugm3'.")
                        st.info("Falling back to sample data.")
                        analyzer.generate_sample_data(days=30)
                        st.success("Sample data generated instead.")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                analyzer.generate_sample_data(days=30)
                st.success("Sample data generated instead.")
        else:
            # Only generate default data if none exists
            if not hasattr(analyzer, 'data_points') or not analyzer.data_points:
                analyzer.generate_sample_data(days=30)
                st.info("Default sample data loaded.")

    # --- Data Visualization (Optimized) ---
    if hasattr(analyzer, 'data_points') and analyzer.data_points:
        st.header("2. Time Series Data Visualization")
        
        # Use session state to cache the plot
        if 'time_series_plot' not in st.session_state or st.button("Refresh Plot"):
            with st.spinner("Generating plot..."):
                fig, ax = plt.subplots(figsize=(12, 6))
                # Downsample if too many points for better performance
                if len(analyzer.data_points) > 1000:
                    step = len(analyzer.data_points) // 1000
                    time_sample = analyzer.time_points[::step]
                    data_sample = analyzer.data_points[::step]
                else:
                    time_sample = analyzer.time_points
                    data_sample = analyzer.data_points
                    
                ax.plot(time_sample, data_sample, label='Pollution Concentration', color='skyblue', linewidth=1)
                ax.set_title('Pollution Concentration Over Time')
                ax.set_xlabel('Time (Hours)')
                ax.set_ylabel('Concentration (µg/m³)')
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                st.session_state.time_series_plot = fig
        
        st.pyplot(st.session_state.time_series_plot)
    else:
        st.warning("No data available for visualization. Load or generate data first.")

    # --- Analysis Section (Lazy Loading) ---
    if hasattr(analyzer, 'data_points') and analyzer.data_points:
        st.header("3. Pollution Analysis Reports")

        # Use expanders to avoid loading everything at once
        with st.expander("3.1. Comprehensive Analysis Report"):
            if st.button("Generate Full Report"):
                with st.spinner("Generating comprehensive report..."):
                    report_text = analyzer.generate_report()
                    st.text_area("Full Analysis Report", report_text, height=400)

        with st.expander("3.2. Custom Time Period Exposure", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.number_input(
                    "Start Hour:",
                    min_value=float(analyzer.time_points[0]),
                    max_value=float(analyzer.time_points[-1]),
                    value=float(analyzer.time_points[0]),
                    step=1.0,
                    key="start_time"
                )
            with col2:
                end_time = st.number_input(
                    "End Hour:",
                    min_value=float(analyzer.time_points[0]),
                    max_value=float(analyzer.time_points[-1]),
                    value=min(float(analyzer.time_points[0]) + 24, float(analyzer.time_points[-1])),  # Default to 24h
                    step=1.0,
                    key="end_time"
                )

            if st.button("Calculate Exposure"):
                if start_time >= end_time:
                    st.error("End time must be greater than start time.")
                else:
                    with st.spinner("Calculating exposure..."):
                        exposure_result = analyzer.calculate_total_exposure(start_time, end_time)
                        if "error" not in exposure_result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Exposure", f"{exposure_result['total_exposure']:.2f} µg·h/m³")
                            with col2:
                                st.metric("Average Concentration", f"{exposure_result['average_concentration']:.2f} µg/m³")
                            with col3:
                                st.metric("Integration Iterations", exposure_result['integration_iterations'])
                        else:
                            st.error(exposure_result["error"])

        with st.expander("3.3. Peak Exposure Analysis"):
            custom_threshold = st.slider(
                "Select Peak Threshold (µg/m³):",
                min_value=0.0,
                max_value=float(max(analyzer.data_points) if analyzer.data_points else 100.0),
                value=25.0,
                step=1.0,
                key="custom_threshold"
            )

            if st.button("Analyze Peak Exposure"):
                with st.spinner("Analyzing peak exposure..."):
                    peak_result = analyzer.analyze_peak_exposure(custom_threshold)
                    if "error" not in peak_result:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Excess Exposure", f"{peak_result['total_excess_exposure']:.2f} µg·h/m³")
                            st.metric("Hours Above Threshold", peak_result['hours_above_threshold'])
                        with col2:
                            st.metric("Percentage Above Threshold", f"{peak_result['percentage_above_threshold']:.1f}%")
                            st.metric("Integration Iterations", peak_result['integration_iterations'])
                    else:
                        st.error(peak_result["error"])

        with st.expander("3.4. Daily Average Concentrations"):
            if st.button("Calculate Daily Averages"):
                with st.spinner("Calculating daily averages..."):
                    daily_averages = analyzer.calculate_daily_averages()
                    if daily_averages:
                        df_daily = pd.DataFrame(daily_averages)
                        df_daily.columns = ['Day', 'Average Concentration (µg/m³)', 'Integration Iterations']
                        st.dataframe(df_daily.set_index('Day'))
                    else:
                        st.info("No daily averages to display.")

        with st.expander("3.5. Download Sample CSV"):
            if st.button("Generate Sample CSV"):
                try:
                    from pollution_analyzer_logic import create_sample_csv
                    csv_buffer = io.StringIO()
                    create_sample_csv(filename=csv_buffer, days=30)
                    csv_buffer.seek(0)

                    st.download_button(
                        label="Download pollution_data.csv",
                        data=csv_buffer.getvalue(),
                        file_name="pollution_data.csv",
                        mime="text/csv",
                    )
                    st.success("Sample CSV ready for download!")
                except ImportError:
                    st.error("Sample CSV generation not available - missing create_sample_csv function")

# =====================================================
# TAB 2: GEOGRAPHIC VISUALIZATION (Optimized)
# =====================================================
with tab2:
    st.header("🗺️ Geographical Pollution Data Visualization")
    st.markdown("Upload a CSV file with geographic coordinates and pollution data to visualize pollution hotspots.")
    
    # Create columns for options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        use_sample_geo = st.checkbox("Use Sample Geographic Data", key="use_sample_geo")
    
    with col2:
        if use_sample_geo:
            sample_size = st.selectbox("Sample Size:", [25, 50, 100, 200], index=1, key="sample_size")
    
    df_geo = None
    
    if use_sample_geo:
        # Use cached sample data generation
        with st.spinner("Loading sample data..."):
            df_geo = generate_sample_geo_data(sample_size if use_sample_geo else 50)
        
        st.success("✅ Sample geographic data loaded!")
        with st.expander("View Sample Data", expanded=False):
            st.dataframe(df_geo.head(10))
    
    else:
        # File upload for geographic data
        uploaded_geo_file = st.file_uploader(
            "Upload CSV with latitude, longitude, and pollution columns", 
            type="csv", 
            key="geographic_upload"
        )
        
        if uploaded_geo_file:
            try:
                with st.spinner("Loading CSV..."):
                    df_geo = pd.read_csv(uploaded_geo_file)
                st.success("CSV loaded successfully!")
                with st.expander("View Uploaded Data", expanded=False):
                    st.dataframe(df_geo.head(10))
            except Exception as e:
                st.error(f"Error loading geographic CSV: {e}")
                df_geo = None
    
    # Process the data if available
    if df_geo is not None and len(df_geo) > 0:
        # Check for required columns
        required_columns = {'latitude', 'longitude', 'pollution'}
        available_columns = set(df_geo.columns)
        
        if required_columns.issubset(available_columns):
            # Clean data efficiently
            initial_count = len(df_geo)
            df_geo = df_geo.dropna(subset=['latitude', 'longitude', 'pollution'])
            df_geo = df_geo[
                (df_geo['latitude'].between(-90, 90)) & 
                (df_geo['longitude'].between(-180, 180)) &
                (df_geo['pollution'] >= 0)
            ]
            
            if len(df_geo) == 0:
                st.error("No valid data points after cleaning. Please check your coordinates.")
            else:
                if len(df_geo) < initial_count:
                    st.warning(f"Removed {initial_count - len(df_geo)} invalid data points during cleaning.")
                
                # Display data statistics efficiently
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Data Points", len(df_geo))
                with col2:
                    st.metric("Avg Pollution", f"{df_geo['pollution'].mean():.1f}")
                with col3:
                    st.metric("Max Pollution", f"{df_geo['pollution'].max():.1f}")
                with col4:
                    st.metric("Min Pollution", f"{df_geo['pollution'].min():.1f}")
                
                # Basic map visualization (fast)
                st.subheader("📍 Basic Map View")
                try:
                    # Limit points for performance
                    display_df = df_geo.sample(min(500, len(df_geo))) if len(df_geo) > 500 else df_geo
                    st.map(display_df[['latitude', 'longitude']])
                    if len(df_geo) > 500:
                        st.info(f"Showing {len(display_df)} random points out of {len(df_geo)} for performance.")
                except Exception as e:
                    st.warning(f"Basic map failed: {e}")
                
                # Advanced visualization (on-demand)
                st.subheader("🎯 Interactive Pollution Visualization")
                
                # Only show advanced viz if user wants it
                show_advanced = st.checkbox("Show Advanced Visualization", key="show_advanced")
                
                if show_advanced:
                    try:
                        # Limit data for performance
                        viz_df = df_geo.sample(min(300, len(df_geo))) if len(df_geo) > 300 else df_geo
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            point_radius = st.slider("Point Radius:", 50, 500, 150, key="point_radius")
                        with col2:
                            zoom_level = st.slider("Zoom Level:", 5, 15, 10, key="zoom_level")
                        
                        # Calculate center efficiently
                        center_lat = float(viz_df['latitude'].mean())
                        center_lon = float(viz_df['longitude'].mean())
                        
                        # Normalize pollution values
                        max_pol = float(viz_df['pollution'].max())
                        min_pol = float(viz_df['pollution'].min())
                        
                        if max_pol != min_pol:
                            viz_df = viz_df.copy()
                            viz_df['color_intensity'] = ((viz_df['pollution'] - min_pol) / 
                                                       (max_pol - min_pol) * 200 + 55)
                        else:
                            viz_df = viz_df.copy()
                            viz_df['color_intensity'] = 128
                        
                        # Create simplified PyDeck chart
                        deck = pdk.Deck(
                            layers=[
                                pdk.Layer(
                                    'ScatterplotLayer',
                                    data=viz_df,
                                    get_position=['longitude', 'latitude'],
                                    get_color=['color_intensity', 50, 100, 160],
                                    get_radius=point_radius,
                                    pickable=True
                                )
                            ],
                            initial_view_state=pdk.ViewState(
                                latitude=center_lat,
                                longitude=center_lon,
                                zoom=zoom_level,
                                pitch=0,
                            ),
                            tooltip={'text': 'Pollution: {pollution:.1f}\nLat: {latitude:.3f}\nLon: {longitude:.3f}'}
                        )
                        
                        st.pydeck_chart(deck)
                        
                        if len(df_geo) > 300:
                            st.info(f"Showing {len(viz_df)} random points for performance. Full dataset has {len(df_geo)} points.")
                            
                    except Exception as e:
                        st.warning(f"Advanced visualization failed: {e}")
                        st.info("Showing matplotlib scatter plot instead:")
                        
                        # Lightweight matplotlib fallback
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_df = df_geo.sample(min(200, len(df_geo))) if len(df_geo) > 200 else df_geo
                        scatter = ax.scatter(plot_df['longitude'], plot_df['latitude'], 
                                           c=plot_df['pollution'], cmap='Reds', 
                                           alpha=0.7, s=30)
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        ax.set_title('Pollution Levels by Geographic Location')
                        plt.colorbar(scatter, label='Pollution Level')
                        st.pyplot(fig)
                
                # Geographic analysis (on-demand)
                with st.expander("📊 Geographic Analysis Summary"):
                    try:
                        # Efficient quadrant analysis
                        lat_median = df_geo['latitude'].median()
                        lon_median = df_geo['longitude'].median()
                        
                        df_geo_analysis = df_geo.copy()
                        df_geo_analysis['quadrant'] = np.where(
                            df_geo_analysis['latitude'] > lat_median,
                            np.where(df_geo_analysis['longitude'] > lon_median, 'North-East', 'North-West'),
                            np.where(df_geo_analysis['longitude'] > lon_median, 'South-East', 'South-West')
                        )
                        
                        quadrant_stats = df_geo_analysis.groupby('quadrant')['pollution'].agg(['mean', 'max', 'count']).round(2)
                        quadrant_stats.columns = ['Average Pollution', 'Max Pollution', 'Data Points']
                        st.dataframe(quadrant_stats)
                        
                        # Efficient hotspot identification
                        pollution_threshold = st.slider(
                            "Pollution Hotspot Threshold:", 
                            float(df_geo['pollution'].min()), 
                            float(df_geo['pollution'].max()), 
                            float(df_geo['pollution'].quantile(0.75)),
                            key="hotspot_threshold"
                        )
                        
                        hotspots = df_geo[df_geo['pollution'] > pollution_threshold]
                        st.write(f"**Hotspots (>{pollution_threshold:.1f}):** {len(hotspots)} locations ({len(hotspots)/len(df_geo)*100:.1f}%)")
                        
                        if len(hotspots) > 0:
                            st.dataframe(hotspots[['latitude', 'longitude', 'pollution']].head(10))
                            
                    except Exception as e:
                        st.warning(f"Geographic analysis failed: {e}")
                
        else:
            missing_cols = required_columns - available_columns
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("**Required columns:** latitude, longitude, pollution")
            st.info("**Available columns:** " + ", ".join(available_columns))
    
    # Show example when no data
    if df_geo is None or len(df_geo) == 0:
        if not use_sample_geo:
            st.info("👆 Upload a CSV file or use sample data to begin geographic analysis")
        
        with st.expander("📋 Expected CSV Format", expanded=False):
            st.markdown("""
            **Required columns:**
            - **latitude**: Geographic latitude (-90 to 90)
            - **longitude**: Geographic longitude (-180 to 180)  
            - **pollution**: Pollution measurement (positive numbers)
            """)
            
            sample_geo_data = pd.DataFrame({
                'latitude': [40.7128, 40.7589, 40.6892, 40.7505, 40.7614],
                'longitude': [-74.0060, -73.9851, -74.0445, -73.9934, -73.9776],
                'pollution': [25.5, 32.1, 18.7, 41.2, 28.9]
            })
            st.dataframe(sample_geo_data)

# =====================================================
# SIDEBAR: Application Info and Instructions
# =====================================================
with st.sidebar:
    st.header("ℹ️ Application Guide")
    
    st.markdown("""
    ### 📈 Time Series Analysis
    - **Generate Sample Data**: Creates realistic pollution time series
    - **Upload CSV**: Must contain 'Concentration_ugm3' column
    - **Romberg Integration**: High-precision numerical analysis
    - **Reports**: Comprehensive exposure analysis
    
    ### 🗺️ Geographic Visualization  
    - **Upload CSV**: Must contain 'latitude', 'longitude', 'pollution' columns
    - **Interactive Maps**: 2D and 3D visualizations
    - **Hotspot Analysis**: Identify pollution concentration areas
    - **Regional Statistics**: Quadrant-based analysis
    """)
    
    st.markdown("---")
    st.markdown("**🔬 Built with:**")
    st.markdown("- Streamlit for web interface")
    st.markdown("- Romberg method for integration")
    st.markdown("- PyDeck for 3D mapping")
    st.markdown("- Matplotlib for time series plots")