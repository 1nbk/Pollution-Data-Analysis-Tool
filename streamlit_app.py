# streamlit_app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
import io
import matplotlib.pyplot as plt
from pollution_analyzer_logic import PollutionDataAnalyzer, create_sample_csv

st.set_page_config(layout="wide", page_title="Comprehensive Pollution Data Analyzer")

st.title("🌬️ Comprehensive Pollution Data Analysis Platform")
st.markdown("**Time Series Analysis with Romberg Method + Geographic Visualization**")

# Initialize the analyzer (Streamlit's cache ensures it's not re-initialized unnecessarily)
@st.cache_resource
def get_analyzer():
    return PollutionDataAnalyzer()

analyzer = get_analyzer()

# Create tabs for different analysis types
tab1, tab2 = st.tabs(["📈 Time Series Analysis", "🗺️ Geographic Visualization"])

# =====================================================
# TAB 1: TIME SERIES ANALYSIS (Original functionality)
# =====================================================
with tab1:
    # --- Data Loading Section ---
    st.header("1. Load Time Series Pollution Data")

    data_source = st.radio(
        "Choose data source:",
        ("Generate Sample Data", "Upload CSV File"),
        key="data_source_timeseries"
    )

    if data_source == "Generate Sample Data":
        st.info("Generating realistic sample pollution data (30 days by default).")
        analyzer.generate_sample_data(days=30)
        st.success("Sample data generated!")
        if analyzer.data_points:
            st.write(f"Showing first 50 hours of generated data:")
            df_display = pd.DataFrame({
                'Hour': analyzer.time_points,
                'Concentration (µg/m³)': analyzer.data_points
            }).head(50)
            st.dataframe(df_display.set_index('Hour'))

    elif data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload your time series CSV file", type="csv", key="timeseries_upload")
        if uploaded_file is not None:
            try:
                # Read the uploaded file directly
                df_uploaded = pd.read_csv(uploaded_file)
                if 'Concentration_ugm3' in df_uploaded.columns:
                    analyzer.data_points = df_uploaded['Concentration_ugm3'].tolist()
                    analyzer.time_points = list(range(len(analyzer.data_points)))
                    st.success(f"Successfully loaded {len(analyzer.data_points)} data points from CSV.")
                    st.write("First 5 rows of uploaded data:")
                    st.dataframe(df_uploaded.head())
                else:
                    st.error("CSV must contain a column named 'Concentration_ugm3'.")
                    st.info("Falling back to sample data as CSV format is incorrect.")
                    analyzer.generate_sample_data(days=30) # Fallback
                    st.success("Sample data generated instead.")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                st.info("Falling back to sample data due to loading error.")
                analyzer.generate_sample_data(days=30) # Fallback
                st.success("Sample data generated instead.")
        else:
            st.info("Please upload a CSV file or select 'Generate Sample Data'.")
            # Ensure some data exists if nothing is uploaded yet
            if not analyzer.data_points:
                analyzer.generate_sample_data(days=30) # Initial load if no file uploaded yet
                st.success("Default sample data loaded for initial view.")

    # --- Data Visualization ---
    if analyzer.data_points:
        st.header("2. Time Series Data Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(analyzer.time_points, analyzer.data_points, label='Pollution Concentration', color='skyblue')
        ax.set_title('Pollution Concentration Over Time')
        ax.set_xlabel('Time (Hours)')
        ax.set_ylabel('Concentration (µg/m³)')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        st.pyplot(fig)
    else:
        st.warning("No data available for visualization. Load or generate data first.")

    # --- Analysis Section ---
    if analyzer.data_points:
        st.header("3. Pollution Analysis Reports")

        st.subheader("3.1. Comprehensive Analysis Report")
        report_text = analyzer.generate_report()
        st.text_area("Full Analysis Report", report_text, height=500)

        st.subheader("3.2. Custom Time Period Exposure")
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
                value=float(analyzer.time_points[-1]),
                step=1.0,
                key="end_time"
            )

        if start_time >= end_time:
            st.error("End time must be greater than start time.")
        else:
            exposure_result = analyzer.calculate_total_exposure(start_time, end_time)
            if "error" not in exposure_result:
                st.markdown(f"**Total Exposure ({start_time:.0f}-{end_time:.0f}h):** {exposure_result['total_exposure']:.2f} µg·h/m³")
                st.markdown(f"**Average Concentration ({start_time:.0f}-{end_time:.0f}h):** {exposure_result['average_concentration']:.2f} µg/m³")
                st.markdown(f"**Integration Iterations Used:** {exposure_result['integration_iterations']}")
            else:
                st.error(exposure_result["error"])

        st.subheader("3.3. Peak Exposure Analysis with Custom Threshold")
        custom_threshold = st.slider(
            "Select Peak Threshold (µg/m³):",
            min_value=0.0,
            max_value=float(max(analyzer.data_points) if analyzer.data_points else 100.0),
            value=25.0, # Default to 25 as per WHO
            step=1.0,
            key="custom_threshold"
        )

        peak_result = analyzer.analyze_peak_exposure(custom_threshold)
        if "error" not in peak_result:
            st.markdown(f"**Threshold:** {peak_result['threshold']:.1f} µg/m³")
            st.markdown(f"**Total Excess Exposure:** {peak_result['total_excess_exposure']:.2f} µg·h/m³")
            st.markdown(f"**Hours Above Threshold:** {peak_result['hours_above_threshold']}")
            st.markdown(f"**Percentage Above Threshold:** {peak_result['percentage_above_threshold']:.1f}%")
            st.markdown(f"**Integration Iterations Used:** {peak_result['integration_iterations']}")
        else:
            st.error(peak_result["error"])

        st.subheader("3.4. Daily Average Concentrations")
        daily_averages = analyzer.calculate_daily_averages()
        if daily_averages:
            df_daily = pd.DataFrame(daily_averages)
            df_daily.columns = ['Day', 'Average Concentration (µg/m³)', 'Integration Iterations']
            st.dataframe(df_daily.set_index('Day'))
        else:
            st.info("No daily averages to display.")

        st.subheader("3.5. Create Sample CSV")
        st.markdown("You can generate and download a sample CSV file for testing purposes.")
        if st.button("Generate & Download Sample CSV"):
            # Use io.StringIO to create an in-memory CSV file
            csv_buffer = io.StringIO()
            create_sample_csv(filename=csv_buffer, days=30)
            csv_buffer.seek(0) # Rewind the buffer to the beginning

            st.download_button(
                label="Download pollution_data.csv",
                data=csv_buffer.getvalue(),
                file_name="pollution_data.csv",
                mime="text/csv",
            )
            st.success("Sample CSV ready for download!")

# =====================================================
# TAB 2: GEOGRAPHIC VISUALIZATION (New functionality)
# =====================================================
with tab2:
    st.header("🗺️ Geographical Pollution Data Visualization")
    st.markdown("Upload a CSV file with geographic coordinates and pollution data to visualize pollution hotspots on an interactive map.")
    
    # Create sample data option
    use_sample_geo = st.checkbox("Use Sample Geographic Data", key="use_sample_geo")
    
    df_geo = None
    
    if use_sample_geo:
        # Create sample geographic data
        import numpy as np
        np.random.seed(42)  # For reproducible results
        
        # Generate sample data around a central location (e.g., New York City area)
        n_points = 50
        center_lat, center_lon = 40.7128, -74.0060
        
        # Generate random points within ~10km radius
        lats = np.random.normal(center_lat, 0.05, n_points)
        lons = np.random.normal(center_lon, 0.05, n_points)
        pollutions = np.random.gamma(2, 15, n_points)  # Gamma distribution for realistic pollution values
        
        df_geo = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'pollution': pollutions
        })
        
        st.success("✅ Sample geographic data generated!")
        st.write("**Sample Data Preview:**")
        st.dataframe(df_geo.head())
    
    else:
        # File upload for geographic data
        uploaded_geo_file = st.file_uploader(
            "Upload CSV with latitude, longitude, and pollution columns", 
            type="csv", 
            key="geographic_upload"
        )
        
        if uploaded_geo_file:
            try:
                df_geo = pd.read_csv(uploaded_geo_file)
                st.write("**Uploaded Data Preview:**")
                st.dataframe(df_geo.head())
            except Exception as e:
                st.error(f"Error loading geographic CSV: {e}")
                df_geo = None
    
    # Process the data if available
    if df_geo is not None:
        # Check for required columns
        required_columns = {'latitude', 'longitude', 'pollution'}
        available_columns = set(df_geo.columns)
        
        if required_columns.issubset(available_columns):
            st.success("✅ All required columns found: latitude, longitude, pollution")
            
            # Clean data - remove any invalid coordinates
            df_geo = df_geo.dropna(subset=['latitude', 'longitude', 'pollution'])
            df_geo = df_geo[(df_geo['latitude'].between(-90, 90)) & 
                           (df_geo['longitude'].between(-180, 180))]
            
            if len(df_geo) == 0:
                st.error("No valid data points after cleaning. Please check your coordinates.")
            else:
                # Display data statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Data Points", len(df_geo))
                with col2:
                    st.metric("Avg Pollution Level", f"{df_geo['pollution'].mean():.2f}")
                with col3:
                    st.metric("Max Pollution Level", f"{df_geo['pollution'].max():.2f}")
                
                # Basic map visualization using Streamlit's built-in map
                st.subheader("📍 Basic Map View")
                try:
                    st.map(df_geo[['latitude', 'longitude']])
                except Exception as e:
                    st.warning(f"Basic map failed to load: {e}")
                
                # Advanced visualization with PyDeck (with error handling)
                st.subheader("🎯 Interactive Pollution Visualization")
                
                try:
                    # Customize visualization settings
                    col1, col2 = st.columns(2)
                    with col1:
                        point_radius = st.slider("Point Radius:", 50, 500, 200, key="point_radius")
                    with col2:
                        zoom_level = st.slider("Zoom Level:", 5, 15, 10, key="zoom_level")
                    
                    # Calculate center point for map
                    center_lat = float(df_geo['latitude'].mean())
                    center_lon = float(df_geo['longitude'].mean())
                    
                    # Normalize pollution values for color mapping (0-255 range)
                    max_pollution = float(df_geo['pollution'].max())
                    min_pollution = float(df_geo['pollution'].min())
                    
                    # Avoid division by zero
                    if max_pollution == min_pollution:
                        df_geo['color_intensity'] = 128
                    else:
                        df_geo['color_intensity'] = ((df_geo['pollution'] - min_pollution) / 
                                                   (max_pollution - min_pollution) * 255)
                    
                    # Create the scatter plot layer
                    scatter_layer = pdk.Layer(
                        'ScatterplotLayer',
                        data=df_geo,
                        get_position=['longitude', 'latitude'],
                        get_color=[200, 'color_intensity', 100, 160],
                        get_radius=point_radius,
                        pickable=True
                    )
                    
                    # Create PyDeck chart with simpler configuration
                    deck = pdk.Deck(
                        layers=[scatter_layer],
                        initial_view_state=pdk.ViewState(
                            latitude=center_lat,
                            longitude=center_lon,
                            zoom=zoom_level,
                            pitch=0,
                        ),
                        tooltip={
                            'text': 'Pollution Level: {pollution}\nLat: {latitude}\nLon: {longitude}'
                        }
                    )
                    
                    st.pydeck_chart(deck)
                    
                except Exception as e:
                    st.warning(f"Advanced visualization failed: {e}")
                    st.info("Showing basic scatter plot instead:")
                    
                    # Fallback to matplotlib scatter plot
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(df_geo['longitude'], df_geo['latitude'], 
                                       c=df_geo['pollution'], cmap='Reds', 
                                       alpha=0.7, s=50)
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    ax.set_title('Pollution Levels by Geographic Location')
                    plt.colorbar(scatter, label='Pollution Level')
                    st.pyplot(fig)
                
                # Additional analysis for geographic data
                st.subheader("📊 Geographic Analysis Summary")
                
                try:
                    # Pollution statistics by geographic regions (simple quadrant analysis)
                    lat_median = df_geo['latitude'].median()
                    lon_median = df_geo['longitude'].median()
                    
                    # Create quadrants
                    df_geo['quadrant'] = df_geo.apply(lambda row: 
                        f"{'North' if row['latitude'] > lat_median else 'South'}-{'East' if row['longitude'] > lon_median else 'West'}", 
                        axis=1
                    )
                    
                    quadrant_stats = df_geo.groupby('quadrant')['pollution'].agg(['mean', 'max', 'count']).round(2)
                    quadrant_stats.columns = ['Average Pollution', 'Max Pollution', 'Data Points']
                    st.dataframe(quadrant_stats)
                    
                    # Hotspot identification
                    pollution_threshold = st.slider(
                        "Pollution Hotspot Threshold:", 
                        float(min_pollution), 
                        float(max_pollution), 
                        float(df_geo['pollution'].quantile(0.75)),
                        key="hotspot_threshold"
                    )
                    
                    hotspots = df_geo[df_geo['pollution'] > pollution_threshold]
                    st.write(f"**Hotspots (>{pollution_threshold:.2f}):** {len(hotspots)} locations")
                    
                    if len(hotspots) > 0:
                        st.dataframe(hotspots[['latitude', 'longitude', 'pollution']].head(10))
                        
                except Exception as e:
                    st.warning(f"Geographic analysis failed: {e}")
                
        else:
            missing_cols = required_columns - available_columns
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("**Required columns:** latitude, longitude, pollution")
            st.info("**Available columns:** " + ", ".join(available_columns))
            
            # Show sample format
            st.subheader("📋 Expected CSV Format:")
            sample_geo_data = pd.DataFrame({
                'latitude': [40.7128, 40.7589, 40.6892],
                'longitude': [-74.0060, -73.9851, -74.0445],
                'pollution': [25.5, 32.1, 18.7]
            })
            st.dataframe(sample_geo_data)
    
    else:
        if not use_sample_geo:
            st.info("👆 Upload a CSV file or use sample data to begin geographic analysis")
        
        # Show example format when no file is uploaded
        st.subheader("📋 Expected CSV Format:")
        st.markdown("""
        Your CSV file should contain these columns:
        - **latitude**: Geographic latitude (decimal degrees, -90 to 90)
        - **longitude**: Geographic longitude (decimal degrees, -180 to 180)  
        - **pollution**: Pollution measurement (any positive numeric value)
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