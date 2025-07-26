# streamlit_app.py
# Main Streamlit application for comprehensive pollution data analysis
# This app provides both time series analysis and geographic visualization of pollution data

# Import necessary libraries for the application
import streamlit as st          # Main web app framework
import pandas as pd             # Data manipulation and analysis
import pydeck as pdk           # 3D geographic visualization
import io                      # Input/output operations for file handling
import matplotlib.pyplot as plt # Plotting and visualization
import numpy as np             # Numerical computations

# Configure Streamlit page settings - MUST be the first Streamlit command
st.set_page_config(layout="wide", page_title="Comprehensive Pollution Data Analyzer")

# Define a cached function to load the pollution analyzer module
# @st.cache_resource ensures this only runs once and caches the result
@st.cache_resource
def get_analyzer():
    """
    Lazy loading function for the pollution analyzer module.
    Uses caching to avoid repeated imports and handles import errors gracefully.
    """
    try:
        # Try to import the custom pollution analyzer logic
        from pollution_analyzer_logic import PollutionDataAnalyzer
        return PollutionDataAnalyzer()  # Return instance of the analyzer
    except ImportError:
        # If the module is not found, show error and return None
        st.error("pollution_analyzer_logic module not found. Using mock analyzer.")
        return None

# Define a cached function to generate sample geographic data
# @st.cache_data caches the output data to improve performance
@st.cache_data
def generate_sample_geo_data(n_points=50):
    """
    Generate cached sample geographic data for demonstration purposes.
    Creates realistic pollution data points around New York City coordinates.
    """
    np.random.seed(42)  # Set random seed for reproducible results
    # Set center coordinates (NYC: latitude, longitude)
    center_lat, center_lon = 40.7128, -74.0060
    
    # Generate random coordinates around the center point
    # Normal distribution creates realistic clustering around center
    lats = np.random.normal(center_lat, 0.05, n_points)  # Latitude values
    lons = np.random.normal(center_lon, 0.05, n_points)  # Longitude values
    
    # Generate pollution values using gamma distribution (realistic for pollution data)
    pollutions = np.random.gamma(2, 15, n_points)
    
    # Return as pandas DataFrame with proper column names
    return pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'pollution': pollutions
    })

# Create the main application title and description
st.title("🌬️ Comprehensive Pollution Data Analysis Platform")
st.markdown("**Time Series Analysis with Romberg Method + Geographic Visualization**")

# Create tabs to organize different analysis types
# This separates time series analysis from geographic visualization
tab1, tab2 = st.tabs(["📈 Time Series Analysis", "🗺️ Geographic Visualization"])

# =====================================================
# TAB 1: TIME SERIES ANALYSIS (Optimized for performance)
# =====================================================
with tab1:
    # Load the analyzer only when this tab is accessed (lazy loading)
    analyzer = get_analyzer()
    
    # Check if analyzer loaded successfully, stop execution if not
    if analyzer is None:
        st.error("Time series analysis not available - missing analyzer module")
        st.stop()  # Stop execution of this tab
    
    # --- Data Loading Section ---
    st.header("1. Load Time Series Pollution Data")

    # Create radio buttons for data source selection
    data_source = st.radio(
        "Choose data source:",
        ("Generate Sample Data", "Upload CSV File"),
        key="data_source_timeseries"  # Unique key for this widget
    )

    # Use session state to track if sample data has been generated
    # This prevents regenerating data on every app refresh
    if 'sample_data_generated' not in st.session_state:
        st.session_state.sample_data_generated = False

    # Handle sample data generation
    if data_source == "Generate Sample Data":
        # Only generate if not already generated (performance optimization)
        if not st.session_state.sample_data_generated:
            # Show spinner while generating data
            with st.spinner("Generating sample data..."):
                analyzer.generate_sample_data(days=30)  # Generate 30 days of data
                st.session_state.sample_data_generated = True  # Mark as generated
        
        # Show success message
        st.success("Sample data ready!")
        
        # Display sample of the generated data in an expandable section
        if analyzer.data_points:
            with st.expander("View Sample Data", expanded=False):
                # Create DataFrame for display (limit to first 50 points for performance)
                df_display = pd.DataFrame({
                    'Hour': analyzer.time_points[:50],  # Time values
                    'Concentration (µg/m³)': analyzer.data_points[:50]  # Pollution values
                })
                # Display as a table with Hour as index
                st.dataframe(df_display.set_index('Hour'))

    # Handle CSV file upload
    elif data_source == "Upload CSV File":
        # Create file uploader widget
        uploaded_file = st.file_uploader("Upload your time series CSV file", type="csv", key="timeseries_upload")
        
        # Process uploaded file if one is provided
        if uploaded_file is not None:
            try:
                # Show loading spinner while processing CSV
                with st.spinner("Loading CSV data..."):
                    df_uploaded = pd.read_csv(uploaded_file)  # Read CSV into DataFrame
                    
                    # Check if required column exists
                    if 'Concentration_ugm3' in df_uploaded.columns:
                        # Extract data from the CSV
                        analyzer.data_points = df_uploaded['Concentration_ugm3'].tolist()
                        analyzer.time_points = list(range(len(analyzer.data_points)))  # Create time indices
                        st.success(f"Successfully loaded {len(analyzer.data_points)} data points from CSV.")
                        
                        # Display preview of uploaded data
                        with st.expander("View Uploaded Data", expanded=False):
                            st.dataframe(df_uploaded.head())  # Show first few rows
                    else:
                        # Handle missing required column
                        st.error("CSV must contain a column named 'Concentration_ugm3'.")
                        st.info("Falling back to sample data.")
                        analyzer.generate_sample_data(days=30)  # Generate fallback data
                        st.success("Sample data generated instead.")
            except Exception as e:
                # Handle any errors in CSV processing
                st.error(f"Error loading CSV: {e}")
                analyzer.generate_sample_data(days=30)  # Generate fallback data
                st.success("Sample data generated instead.")
        else:
            # Generate default data if no file uploaded and no data exists
            if not hasattr(analyzer, 'data_points') or not analyzer.data_points:
                analyzer.generate_sample_data(days=30)
                st.info("Default sample data loaded.")

    # --- Data Visualization Section (Performance Optimized) ---
    # Only show visualization if data is available
    if hasattr(analyzer, 'data_points') and analyzer.data_points:
        st.header("2. Time Series Data Visualization")
        
        # Use session state to cache the plot and avoid regenerating on every interaction
        if 'time_series_plot' not in st.session_state or st.button("Refresh Plot"):
            with st.spinner("Generating plot..."):
                # Create matplotlib figure and axis
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Downsample data if too many points (performance optimization)
                if len(analyzer.data_points) > 1000:
                    # Calculate step size to reduce to ~1000 points
                    step = len(analyzer.data_points) // 1000
                    time_sample = analyzer.time_points[::step]  # Sample time points
                    data_sample = analyzer.data_points[::step]  # Sample data points
                else:
                    # Use all data if reasonable size
                    time_sample = analyzer.time_points
                    data_sample = analyzer.data_points
                    
                # Create the plot
                ax.plot(time_sample, data_sample, label='Pollution Concentration', color='skyblue', linewidth=1)
                ax.set_title('Pollution Concentration Over Time')  # Set plot title
                ax.set_xlabel('Time (Hours)')  # X-axis label
                ax.set_ylabel('Concentration (µg/m³)')  # Y-axis label
                ax.grid(True, linestyle='--', alpha=0.7)  # Add grid for readability
                plt.legend()  # Show legend
                
                # Cache the plot in session state
                st.session_state.time_series_plot = fig
        
        # Display the cached plot
        st.pyplot(st.session_state.time_series_plot)
    else:
        # Show warning if no data is available
        st.warning("No data available for visualization. Load or generate data first.")

    # --- Analysis Section (Lazy Loading for Performance) ---
    # Only show analysis options if data is available
    if hasattr(analyzer, 'data_points') and analyzer.data_points:
        st.header("3. Pollution Analysis Reports")

        # Use expanders to organize analysis options and load them only when needed
        with st.expander("3.1. Comprehensive Analysis Report"):
            # Generate full report only when button is clicked
            if st.button("Generate Full Report"):
                with st.spinner("Generating comprehensive report..."):
                    report_text = analyzer.generate_report()  # Get full analysis report
                    # Display report in a text area (400px height for readability)
                    st.text_area("Full Analysis Report", report_text, height=400)

        # Custom time period exposure analysis
        with st.expander("3.2. Custom Time Period Exposure", expanded=True):
            # Create two columns for start and end time inputs
            col1, col2 = st.columns(2)
            
            with col1:
                # Input for start time with validation
                start_time = st.number_input(
                    "Start Hour:",
                    min_value=float(analyzer.time_points[0]),  # Minimum possible time
                    max_value=float(analyzer.time_points[-1]),  # Maximum possible time
                    value=float(analyzer.time_points[0]),  # Default to first time point
                    step=1.0,  # Step size for input
                    key="start_time"  # Unique key
                )
            
            with col2:
                # Input for end time with validation
                end_time = st.number_input(
                    "End Hour:",
                    min_value=float(analyzer.time_points[0]),
                    max_value=float(analyzer.time_points[-1]),
                    # Default to 24 hours from start or max time, whichever is smaller
                    value=min(float(analyzer.time_points[0]) + 24, float(analyzer.time_points[-1])),
                    step=1.0,
                    key="end_time"
                )

            # Calculate exposure when button is clicked
            if st.button("Calculate Exposure"):
                # Validate time range
                if start_time >= end_time:
                    st.error("End time must be greater than start time.")
                else:
                    with st.spinner("Calculating exposure..."):
                        # Calculate total exposure using the analyzer
                        exposure_result = analyzer.calculate_total_exposure(start_time, end_time)
                        
                        # Display results if calculation successful
                        if "error" not in exposure_result:
                            # Create three columns for metrics display
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Display total exposure metric
                                st.metric("Total Exposure", f"{exposure_result['total_exposure']:.2f} µg·h/m³")
                            with col2:
                                # Display average concentration metric
                                st.metric("Average Concentration", f"{exposure_result['average_concentration']:.2f} µg/m³")
                            with col3:
                                # Display number of integration iterations (technical detail)
                                st.metric("Integration Iterations", exposure_result['integration_iterations'])
                        else:
                            # Display error if calculation failed
                            st.error(exposure_result["error"])

        # Peak exposure analysis section
        with st.expander("3.3. Peak Exposure Analysis"):
            # Slider to set threshold for peak analysis
            custom_threshold = st.slider(
                "Select Peak Threshold (µg/m³):",
                min_value=0.0,
                max_value=float(max(analyzer.data_points) if analyzer.data_points else 100.0),
                value=25.0,  # Default threshold value
                step=1.0,
                key="custom_threshold"
            )

            # Analyze peaks when button is clicked
            if st.button("Analyze Peak Exposure"):
                with st.spinner("Analyzing peak exposure..."):
                    # Perform peak analysis using the analyzer
                    peak_result = analyzer.analyze_peak_exposure(custom_threshold)
                    
                    # Display results if analysis successful
                    if "error" not in peak_result:
                        # Create two columns for results display
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Display excess exposure above threshold
                            st.metric("Total Excess Exposure", f"{peak_result['total_excess_exposure']:.2f} µg·h/m³")
                            # Display hours above threshold
                            st.metric("Hours Above Threshold", peak_result['hours_above_threshold'])
                        
                        with col2:
                            # Display percentage of time above threshold
                            st.metric("Percentage Above Threshold", f"{peak_result['percentage_above_threshold']:.1f}%")
                            # Display integration iterations
                            st.metric("Integration Iterations", peak_result['integration_iterations'])
                    else:
                        # Display error if analysis failed
                        st.error(peak_result["error"])

        # Daily averages analysis section
        with st.expander("3.4. Daily Average Concentrations"):
            # Calculate daily averages when button is clicked
            if st.button("Calculate Daily Averages"):
                with st.spinner("Calculating daily averages..."):
                    # Get daily averages from analyzer
                    daily_averages = analyzer.calculate_daily_averages()
                    
                    # Display results if available
                    if daily_averages:
                        # Convert to DataFrame for better display
                        df_daily = pd.DataFrame(daily_averages)
                        df_daily.columns = ['Day', 'Average Concentration (µg/m³)', 'Integration Iterations']
                        # Display with Day as index
                        st.dataframe(df_daily.set_index('Day'))
                    else:
                        st.info("No daily averages to display.")

        # Sample CSV download section
        with st.expander("3.5. Download Sample CSV"):
            # Generate and download sample CSV when button is clicked
            if st.button("Generate Sample CSV"):
                try:
                    # Import the CSV creation function
                    from pollution_analyzer_logic import create_sample_csv
                    csv_buffer = io.StringIO()  # Create string buffer for CSV data
                    
                    # Generate CSV data into the buffer
                    create_sample_csv(filename=csv_buffer, days=30)
                    csv_buffer.seek(0)  # Reset buffer position to beginning

                    # Create download button
                    st.download_button(
                        label="Download pollution_data.csv",
                        data=csv_buffer.getvalue(),  # Get CSV data as string
                        file_name="pollution_data.csv",
                        mime="text/csv",  # Set proper MIME type
                    )
                    st.success("Sample CSV ready for download!")
                except ImportError:
                    # Handle case where CSV creation function is not available
                    st.error("Sample CSV generation not available - missing create_sample_csv function")

# =====================================================
# TAB 2: GEOGRAPHIC VISUALIZATION (Performance Optimized)
# =====================================================
with tab2:
    st.header("🗺️ Geographical Pollution Data Visualization")
    st.markdown("Upload a CSV file with geographic coordinates and pollution data to visualize pollution hotspots.")
    
    # Create columns for layout organization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Checkbox to use sample geographic data
        use_sample_geo = st.checkbox("Use Sample Geographic Data", key="use_sample_geo")
    
    with col2:
        # Sample size selector (only shown when using sample data)
        if use_sample_geo:
            sample_size = st.selectbox("Sample Size:", [25, 50, 100, 200], index=1, key="sample_size")
    
    # Initialize DataFrame variable
    df_geo = None
    
    # Handle sample data generation
    if use_sample_geo:
        # Use cached sample data generation function
        with st.spinner("Loading sample data..."):
            df_geo = generate_sample_geo_data(sample_size if use_sample_geo else 50)
        
        st.success("✅ Sample geographic data loaded!")
        # Show preview of sample data
        with st.expander("View Sample Data", expanded=False):
            st.dataframe(df_geo.head(10))  # Display first 10 rows
    
    else:
        # Handle CSV file upload for geographic data
        uploaded_geo_file = st.file_uploader(
            "Upload CSV with latitude, longitude, and pollution columns", 
            type="csv", 
            key="geographic_upload"
        )
        
        # Process uploaded file if provided
        if uploaded_geo_file:
            try:
                with st.spinner("Loading CSV..."):
                    df_geo = pd.read_csv(uploaded_geo_file)  # Read CSV into DataFrame
                st.success("CSV loaded successfully!")
                # Show preview of uploaded data
                with st.expander("View Uploaded Data", expanded=False):
                    st.dataframe(df_geo.head(10))
            except Exception as e:
                # Handle CSV loading errors
                st.error(f"Error loading geographic CSV: {e}")
                df_geo = None
    
    # Process the data if available
    if df_geo is not None and len(df_geo) > 0:
        # Define required columns for geographic analysis
        required_columns = {'latitude', 'longitude', 'pollution'}
        available_columns = set(df_geo.columns)  # Get available columns from data
        
        # Check if all required columns are present
        if required_columns.issubset(available_columns):
            # Clean data efficiently for better performance and accuracy
            initial_count = len(df_geo)  # Store initial count for comparison
            
            # Remove rows with missing values in required columns
            df_geo = df_geo.dropna(subset=['latitude', 'longitude', 'pollution'])
            
            # Filter data to valid ranges
            df_geo = df_geo[
                (df_geo['latitude'].between(-90, 90)) &      # Valid latitude range
                (df_geo['longitude'].between(-180, 180)) &   # Valid longitude range
                (df_geo['pollution'] >= 0)                   # Non-negative pollution values
            ]
            
            # Check if any valid data remains after cleaning
            if len(df_geo) == 0:
                st.error("No valid data points after cleaning. Please check your coordinates.")
            else:
                # Show warning if data was removed during cleaning
                if len(df_geo) < initial_count:
                    st.warning(f"Removed {initial_count - len(df_geo)} invalid data points during cleaning.")
                
                # Display data statistics efficiently using metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Data Points", len(df_geo))
                with col2:
                    st.metric("Avg Pollution", f"{df_geo['pollution'].mean():.1f}")
                with col3:
                    st.metric("Max Pollution", f"{df_geo['pollution'].max():.1f}")
                with col4:
                    st.metric("Min Pollution", f"{df_geo['pollution'].min():.1f}")
                
                # Basic map visualization (fast and simple)
                st.subheader("📍 Basic Map View")
                try:
                    # Limit points for performance (Streamlit map can be slow with many points)
                    display_df = df_geo.sample(min(500, len(df_geo))) if len(df_geo) > 500 else df_geo
                    # Display simple map using Streamlit's built-in map function
                    st.map(display_df[['latitude', 'longitude']])
                    
                    # Inform user if data was sampled for performance
                    if len(df_geo) > 500:
                        st.info(f"Showing {len(display_df)} random points out of {len(df_geo)} for performance.")
                except Exception as e:
                    st.warning(f"Basic map failed: {e}")
                
                # Advanced visualization section (on-demand loading)
                st.subheader("🎯 Interactive Pollution Visualization")
                
                # Only show advanced visualization if user explicitly requests it
                show_advanced = st.checkbox("Show Advanced Visualization", key="show_advanced")
                
                if show_advanced:
                    try:
                        # Limit data for performance (PyDeck can be resource-intensive)
                        viz_df = df_geo.sample(min(300, len(df_geo))) if len(df_geo) > 300 else df_geo
                        
                        # Create controls for visualization customization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Slider for point radius
                            point_radius = st.slider("Point Radius:", 50, 500, 150, key="point_radius")
                        with col2:
                            # Slider for map zoom level
                            zoom_level = st.slider("Zoom Level:", 5, 15, 10, key="zoom_level")
                        
                        # Calculate center point for map view (efficient calculation)
                        center_lat = float(viz_df['latitude'].mean())
                        center_lon = float(viz_df['longitude'].mean())
                        
                        # Normalize pollution values for color mapping
                        max_pol = float(viz_df['pollution'].max())
                        min_pol = float(viz_df['pollution'].min())
                        
                        # Only normalize if there's variation in pollution values
                        if max_pol != min_pol:
                            viz_df = viz_df.copy()  # Create copy to avoid modifying original
                            # Map pollution values to color intensity (55-255 range for visibility)
                            viz_df['color_intensity'] = ((viz_df['pollution'] - min_pol) / 
                                                       (max_pol - min_pol) * 200 + 55)
                        else:
                            # Use constant color if all pollution values are the same
                            viz_df = viz_df.copy()
                            viz_df['color_intensity'] = 128
                        
                        # Create PyDeck visualization
                        deck = pdk.Deck(
                            layers=[
                                pdk.Layer(
                                    'ScatterplotLayer',  # Use scatter plot layer for points
                                    data=viz_df,
                                    get_position=['longitude', 'latitude'],  # Set point positions
                                    get_color=['color_intensity', 50, 100, 160],  # Set point colors (RGBA)
                                    get_radius=point_radius,  # Set point radius
                                    pickable=True  # Allow clicking on points
                                )
                            ],
                            initial_view_state=pdk.ViewState(
                                latitude=center_lat,   # Center latitude
                                longitude=center_lon,  # Center longitude
                                zoom=zoom_level,       # Zoom level
                                pitch=0,               # Camera pitch (0 = top-down view)
                            ),
                            # Tooltip to show data when hovering over points
                            tooltip={'text': 'Pollution: {pollution:.1f}\nLat: {latitude:.3f}\nLon: {longitude:.3f}'}
                        )
                        
                        # Display the PyDeck chart
                        st.pydeck_chart(deck)
                        
                        # Inform user if data was sampled for performance
                        if len(df_geo) > 300:
                            st.info(f"Showing {len(viz_df)} random points for performance. Full dataset has {len(df_geo)} points.")
                            
                    except Exception as e:
                        # Fallback to matplotlib if PyDeck fails
                        st.warning(f"Advanced visualization failed: {e}")
                        st.info("Showing matplotlib scatter plot instead:")
                        
                        # Create lightweight matplotlib fallback visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Sample data for performance if needed
                        plot_df = df_geo.sample(min(200, len(df_geo))) if len(df_geo) > 200 else df_geo
                        
                        # Create scatter plot with color mapping
                        scatter = ax.scatter(plot_df['longitude'], plot_df['latitude'], 
                                           c=plot_df['pollution'], cmap='Reds',  # Red colormap for pollution
                                           alpha=0.7, s=30)  # Semi-transparent points
                        
                        # Set axis labels and title
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        ax.set_title('Pollution Levels by Geographic Location')
                        
                        # Add colorbar to show pollution scale
                        plt.colorbar(scatter, label='Pollution Level')
                        st.pyplot(fig)  # Display the plot
                
                # Geographic analysis section (on-demand)
                with st.expander("📊 Geographic Analysis Summary"):
                    try:
                        # Efficient quadrant analysis (divide map into 4 sections)
                        lat_median = df_geo['latitude'].median()   # Find median latitude
                        lon_median = df_geo['longitude'].median()  # Find median longitude
                        
                        # Create copy for analysis
                        df_geo_analysis = df_geo.copy()
                        
                        # Assign quadrant labels based on median values
                        df_geo_analysis['quadrant'] = np.where(
                            df_geo_analysis['latitude'] > lat_median,
                            np.where(df_geo_analysis['longitude'] > lon_median, 'North-East', 'North-West'),
                            np.where(df_geo_analysis['longitude'] > lon_median, 'South-East', 'South-West')
                        )
                        
                        # Calculate statistics for each quadrant
                        quadrant_stats = df_geo_analysis.groupby('quadrant')['pollution'].agg(['mean', 'max', 'count']).round(2)
                        quadrant_stats.columns = ['Average Pollution', 'Max Pollution', 'Data Points']
                        st.dataframe(quadrant_stats)  # Display quadrant statistics
                        
                        # Efficient hotspot identification
                        pollution_threshold = st.slider(
                            "Pollution Hotspot Threshold:", 
                            float(df_geo['pollution'].min()),      # Minimum possible value
                            float(df_geo['pollution'].max()),      # Maximum possible value
                            float(df_geo['pollution'].quantile(0.75)),  # Default to 75th percentile
                            key="hotspot_threshold"
                        )
                        
                        # Find locations above threshold
                        hotspots = df_geo[df_geo['pollution'] > pollution_threshold]
                        # Display hotspot summary
                        st.write(f"**Hotspots (>{pollution_threshold:.1f}):** {len(hotspots)} locations ({len(hotspots)/len(df_geo)*100:.1f}%)")
                        
                        # Show hotspot details if any found
                        if len(hotspots) > 0:
                            # Display first 10 hotspots
                            st.dataframe(hotspots[['latitude', 'longitude', 'pollution']].head(10))
                            
                    except Exception as e:
                        # Handle analysis errors gracefully
                        st.warning(f"Geographic analysis failed: {e}")
                
        else:
            # Handle missing required columns
            missing_cols = required_columns - available_columns
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("**Required columns:** latitude, longitude, pollution")
            st.info("**Available columns:** " + ", ".join(available_columns))
    
    # Show example format when no data is loaded
    if df_geo is None or len(df_geo) == 0:
        # Only show instruction if not using sample data
        if not use_sample_geo:
            st.info("👆 Upload a CSV file or use sample data to begin geographic analysis")
        
        # Show expected CSV format in expandable section
        with st.expander("📋 Expected CSV Format", expanded=False):
            st.markdown("""
            **Required columns:**
            - **latitude**: Geographic latitude (-90 to 90)
            - **longitude**: Geographic longitude (-180 to 180)  
            - **pollution**: Pollution measurement (positive numbers)
            """)
            
            # Create and display example data format
            sample_geo_data = pd.DataFrame({
                'latitude': [40.7128, 40.7589, 40.6892, 40.7505, 40.7614],
                'longitude': [-74.0060, -73.9851, -74.0445, -73.9934, -73.9776],
                'pollution': [25.5, 32.1, 18.7, 41.2, 28.9]
            })
            st.dataframe(sample_geo_data)

# =====================================================
# SIDEBAR: Application Information and User Guide
# =====================================================
with st.sidebar:
    st.header("ℹ️ Application Guide")
    
    # Provide comprehensive user instructions
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
    
    # Add separator line
    st.markdown("---")
    
    # Show technical information about the application
    st.markdown("**🔬 Built with:**")
    st.markdown("- Streamlit for web interface")       # Web framework
    st.markdown("- Romberg method for integration")    # Numerical integration method
    st.markdown("- PyDeck for 3D mapping")            # 3D visualization library
    st.markdown("- Matplotlib for time series plots") # Plotting library