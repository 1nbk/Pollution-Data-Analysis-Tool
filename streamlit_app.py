# streamlit_app.py
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from pollution_analyzer_logic import PollutionDataAnalyzer, create_sample_csv

st.set_page_config(layout="wide", page_title="Pollution Data Analyzer")

st.title("🌬️ Pollution Data Analysis with Romberg Method")

# Initialize the analyzer (Streamlit's cache ensures it's not re-initialized unnecessarily)
@st.cache_resource
def get_analyzer():
    return PollutionDataAnalyzer()

analyzer = get_analyzer()

# --- Data Loading Section ---
st.header("1. Load Pollution Data")

data_source = st.radio(
    "Choose data source:",
    ("Generate Sample Data", "Upload CSV File"),
    key="data_source"
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
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
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
    st.header("2. Data Visualization")
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