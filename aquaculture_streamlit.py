import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Set page config
st.set_page_config(page_title="Aquaculture Analytics", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('aquaculture.csv')

df = load_data()

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("best_xgb_turbidez_model.pkl")
    return model

model = load_model()

# Create sidebar with tabs
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select Tab", ["Data Analysis", "Prediction"])

if tab == "Data Analysis":
    st.title("Aquaculture Data Analysis")
    st.subheader("📘 Overview")
    st.markdown("""
    **🌊 Aquaculture Water Quality Monitoring — 2024**  
    This project analyzes key water quality parameters collected from aquaculture ponds in Montería, Colombia using IoT-based real-time monitoring.


    **Purpose of this App**:
    - Provide visual insights into water quality data.
    - Assist researchers in understanding water parameters.
    - Support sustainability in aquaculture with data-driven decisions.
    
    **Measured Parameters**:
    - 🟠 Temperature (°C)  
    - 🔵 Dissolved Oxygen (mg/L)  
    - 🟢 pH (Acidity/Alkalinity)  
    - ⚪ Turbidity (NTU – Water Clarity)  
    - 📅 Temporal features: hour, day, month  
    - 📐 Scaled values for ML modeling
    """)
    
    # Initial Inspection
    st.subheader("Data Preview")
    st.markdown("Use the preview above to get a sense of the data layout.")
    st.dataframe(df.head())
    
    
    # Distribution Plots with Tabbed Interface
    st.subheader("Parameter Distributions")

    # Findings dictionary with emoji icons
    dist_findings = {
        'temperatura': {
            'icon': '🌡️',
            'summary': "Stable range (26.9°C-27.3°C) | Normally distributed",
            'details': [
                "Normalized range: 0.0-1.0",
                "Min-Max scaled from raw sensor data",
                "No significant outliers detected"
            ]
        },
        'oxigeno': {
            'icon': '🫧',
            'summary': "Widest variation (7.63-8.19 mg/L) | Some outliers present",
            'details': [
                "Normalized range: -1.5 to 2.5",
                "Standardized using z-score normalization",
                "Afternoon peaks suggest photosynthetic activity"
            ]
        },
        'ph': {
            'icon': '🧪', 
            'summary': "Consistent values (0.91-1.00 normalized) | Tight clustering",
            'details': [
                "Normalized range: 0.0-1.0",
                "Min-Max scaled from raw pH readings",
                "Evening peaks indicate CO₂ consumption"
            ]
        },
        'turbidez': {
            'icon': '🌫️',
            'summary': "Most values near zero (-0.04 to +0.03) | Right-skewed distribution",
            'details': [
                "Normalized range: 0.0-1.0",
                "Negative values indicate sensor calibration baseline",
                "Morning/evening peaks correlate with feeding times"
            ]
        }
    }

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡️ Temperature", 
        "🫧 Dissolved Oxygen", 
        "🧪 pH", 
        "🌫️ Turbidity"
    ])

    with tab1:
        col = 'temperatura'
        fig = px.histogram(
            df, 
            x=col, 
            nbins=30, 
            marginal='box',
            title=f"{col.capitalize()} Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📊 Distribution Findings", expanded=True):
            st.markdown(f"**{dist_findings[col]['icon']} {col.capitalize()} Characteristics**")
            st.markdown(f"📌 {dist_findings[col]['summary']}")
            for detail in dist_findings[col]['details']:
                st.markdown(f"• {detail}")

    with tab2:
        col = 'oxigeno'
        fig = px.histogram(
            df, 
            x=col, 
            nbins=30, 
            marginal='box',
            title=f"{col.capitalize()} Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📊 Distribution Findings", expanded=True):
            st.markdown(f"**{dist_findings[col]['icon']} {col.capitalize()} Characteristics**")
            st.markdown(f"📌 {dist_findings[col]['summary']}")
            for detail in dist_findings[col]['details']:
                st.markdown(f"• {detail}")

    with tab3:
        col = 'ph'
        fig = px.histogram(
            df, 
            x=col, 
            nbins=30, 
            marginal='box',
            title=f"{col.capitalize()} Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📊 Distribution Findings", expanded=True):
            st.markdown(f"**{dist_findings[col]['icon']} {col.capitalize()} Characteristics**")
            st.markdown(f"📌 {dist_findings[col]['summary']}")
            for detail in dist_findings[col]['details']:
                st.markdown(f"• {detail}")

    with tab4:
        col = 'turbidez'
        fig = px.histogram(
            df, 
            x=col, 
            nbins=30, 
            marginal='box',
            title=f"{col.capitalize()} Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📊 Distribution Findings", expanded=True):
            st.markdown(f"**{dist_findings[col]['icon']} {col.capitalize()} Characteristics**")
            st.markdown(f"📌 {dist_findings[col]['summary']}")
            for detail in dist_findings[col]['details']:
                st.markdown(f"• {detail}")


    # Hourly Distribution with Findings
    st.subheader("Hourly Distribution of Parameters")
    st.markdown("""
    Understanding how water quality parameters vary across different hours helps identify **daily environmental cycles**, detect **anomalous behavior**, and optimize **aquaculture operations**.
    """)

    # Findings dictionary with emoji icons
    hourly_findings = {
        'temperatura': {
            'icon': '🌡️',
            'observation': "Remains stable (26.9-27.3°C) with minor fluctuations",
            'finding': "Indicates consistent thermal regulation; small variations follow daylight cycle"
        },
        'oxigeno': {
            'icon': '🫧', 
            'observation': "Rises from 6AM, peaks at 14-16PM (8.1-8.2 mg/L), declines by night",
            'finding': "Clear photosynthesis-driven cycle (daytime O₂ production vs nighttime respiration)"
        },
        'ph': {
            'icon': '🧪',
            'observation': "Gradual daytime rise (0.92-0.99), peaks at 18-20PM",
            'finding': "Algal photosynthesis consumes CO₂, increasing pH; stabilizes overnight"
        },
        'turbidez': {
            'icon': '🌫️',
            'observation': "Morning (9AM) and evening (21PM) peaks (-0.02 to +0.03)",
            'finding': "Correlates with feeding activity and fish movement disturbing sediments"
        }
    }

    # Create tabs for each parameter
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡️ Temperature", 
        "🫧 Dissolved Oxygen", 
        "🧪 pH", 
        "🌫️ Turbidity"
    ])

    with tab1:
        col = 'temperatura'
        fig = px.box(df, x='hour', y=col, points='outliers',
                    title=f"Hourly {col.capitalize()} Distribution")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("🔍 Key Findings", expanded=True):
            st.markdown(f"""
            **{hourly_findings[col]['icon']} Observation**:  
            {hourly_findings[col]['observation']}  
            
            **📌 Finding**:  
            {hourly_findings[col]['finding']}
            """)

    with tab2:
        col = 'oxigeno'
        fig = px.box(df, x='hour', y=col, points='outliers',
                    title=f"Hourly {col.capitalize()} Distribution")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("🔍 Key Findings", expanded=True):
            st.markdown(f"""
            **{hourly_findings[col]['icon']} Observation**:  
            {hourly_findings[col]['observation']}  
            
            **📌 Finding**:  
            {hourly_findings[col]['finding']}
            """)

    with tab3:
        col = 'ph'
        fig = px.box(df, x='hour', y=col, points='outliers',
                    title=f"Hourly {col.capitalize()} Distribution")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("🔍 Key Findings", expanded=True):
            st.markdown(f"""
            **{hourly_findings[col]['icon']} Observation**:  
            {hourly_findings[col]['observation']}  
            
            **📌 Finding**:  
            {hourly_findings[col]['finding']}
            """)

    with tab4:
        col = 'turbidez'
        fig = px.box(df, x='hour', y=col, points='outliers',
                    title=f"Hourly {col.capitalize()} Distribution")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("🔍 Key Findings", expanded=True):
            st.markdown(f"""
            **{hourly_findings[col]['icon']} Observation**:  
            {hourly_findings[col]['observation']}  
            
            **📌 Finding**:  
            {hourly_findings[col]['finding']}
            """)

    
    

    
    # Correlation Matrix
    st.subheader("Parameter Correlation Matrix")
    st.markdown("""
            Correlation analysis helps identify relationships between water quality parameters that may be influenced by **underlying environmental or biological processes**.  
            By understanding these interactions, we can detect potential **cause-effect patterns**, **reduce redundancy** in data, and inform **predictive modeling strategies** in aquaculture management.

""")
    corr_matrix = df[['temperatura', 'oxigeno', 'ph', 'turbidez']].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}"
        )
    )
    fig.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
##### 🌡️ Temperature vs Others
- **Observation:** Very weak or negligible correlation with oxygen (0.006), pH (-0.021), and turbidity (0.046).
- **Finding:** Temperature appears largely independent of other parameters in this dataset, suggesting stable thermal control or minimal biological influence.

---

##### 🫧 Oxygen vs Others
- **Observation:** Slight positive correlation with turbidity (0.126) and weak negative correlation with pH (-0.203).
- **Finding:** Dissolved oxygen may be loosely linked to water clarity and biological activity affecting pH, but the relationships are weak.

---

##### 🧪 pH vs Others
- **Observation:** Strong negative correlation with turbidity (-0.611); weak to no correlation with temperature or oxygen.
- **Finding:** Increased turbidity is strongly associated with decreased pH, likely due to organic matter or increased respiration lowering water alkalinity.

---

##### 🌫️ Turbidity vs Others
- **Observation:** Weak positive correlation with oxygen (0.126); strong negative correlation with pH (-0.611).
- **Finding:** High turbidity may indicate higher microbial/organic load affecting both oxygen usage and pH reduction.

---
                """)
    
    # pH vs Turbidity with Regression
    st.subheader("pH vs Turbidity Relationship")
    color_by = st.radio(
                        "",
                        options=['None', 'temperatura', 'oxigeno', 'hour'],
                        index=1,
                        horizontal=True,
                        label_visibility="collapsed"
                    )
    
    if color_by == 'None':
        fig = px.scatter(
            df, 
            x='ph', 
            y='turbidez', 
            trendline='ols',
            title="pH vs Turbidity with Regression Line"
        )
    else:
        fig = px.scatter(
            df, 
            x='ph', 
            y='turbidez', 
            color=color_by,
            color_continuous_scale='viridis',
            trendline='ols',
            title=f"pH vs Turbidity Colored by {color_by.capitalize()}"
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly pH vs Turbidity in sequential order
    st.subheader("Hourly pH vs Turbidity Relationship")
    st.markdown("This visualization explores the hourly relationship between pH and turbidity in aquaculture ponds to uncover potential time-based patterns or correlations. By plotting regression lines for each hour, we aim to identify when these two critical water quality parameters exhibit significant interactions. Understanding such temporal trends is essential for optimizing aquaculture management and preventing harmful water conditions.")
    # Ensure hours are treated as ordered categories
    df['hour'] = pd.Categorical(df['hour'], categories=sorted(df['hour'].unique()), ordered=True)

    hour_range = st.slider(
        "Select hour range:",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )

    filtered_df = df[(df['hour'] >= hour_range[0]) & (df['hour'] <= hour_range[1])].sort_values('hour')

    fig = px.scatter(
        filtered_df,
        x='ph',
        y='turbidez',
        facet_col='hour',
        facet_col_wrap=4,
        category_orders={'hour': list(range(hour_range[0], hour_range[1]+1))},
        trendline='ols',
        title=f"pH vs Turbidity from {hour_range[0]}:00 to {hour_range[1]}:00"
    )

    # Calculate R-squared values for each hour separately
    for i, hour in enumerate(range(hour_range[0], hour_range[1]+1)):
        hour_df = filtered_df[filtered_df['hour'] == hour]
        if len(hour_df) > 1:  # Need at least 2 points for regression
            try:
                # Create separate figure for this hour to get trendline results
                hour_fig = px.scatter(hour_df, x='ph', y='turbidez', trendline='ols')
                results = px.get_trendline_results(hour_fig)
                r_squared = results.iloc[0]["px_fit_results"].rsquared
                
                fig.add_annotation(
                    xref=f"x{i+1}",
                    yref=f"y{i+1}",
                    x=0.05,
                    y=0.95,
                    text=f"R²={r_squared:.2f}",
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=10)
                )
            except:
                pass  # Skip if regression fails

    # Adjust layout
    fig.update_layout(
        height=800 + 300*((hour_range[1]-hour_range[0])//6),
        margin=dict(t=100)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
                ##### Observation
- During mid-day hours (e.g., 13–17), data points are highly clustered and show **minimal variation** in pH and turbidity, indicating **stable conditions**, possibly due to environmental equilibrium or sensor limitations during that time.

##### Finding
- Several hours (e.g., 0, 2, 9, 15, 23) show a **noticeable negative correlation** between pH and turbidity, suggesting that as water becomes clearer (lower turbidity), pH may rise, or vice versa.

                """)
    st.markdown("""
                    
    **Citation**:  
    Baena-Navarro, Rubén; Carriazo-Regino, Yulieth; Torres-Hoyos, Francisco (2024), 
    "Environmental Parameters in Aquaculture: Temperature, pH, Oxygen, and Turbidity Measurements", 
    Mendeley Data, V1, [doi:10.17632/8s73jfvgr5.1](https://data.mendeley.com/datasets/8s73jfvgr5/1)
                """)

# ============== TAB: PREDICTION ==============
elif tab == "Prediction":
    st.title("Aquaculture Parameter Prediction – Turbidez")
    st.write("Enter environmental values below to predict turbidez (water clarity).")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            temperatura = st.number_input("Temperature (normalized)", value=1.0, step=0.01)
            hour = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=12)

        with col2:
            oxigeno = st.number_input("Dissolved Oxygen (normalized)", value=0.5, step=0.01)
            day = st.number_input("Day (1–31)", min_value=1, max_value=31, value=15)

        with col3:
            ph = st.number_input("pH (normalized)", value=0.95, step=0.01)
            month = st.number_input("Month (1–12)", min_value=1, max_value=12, value=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            'temperatura': temperatura,
            'oxigeno': oxigeno,
            'ph': ph,
            'hour': hour,
            'day': day,
            'month': month
        }])

        turbidez_pred = model.predict(input_data)[0]

        st.success(f"🌊 **Predicted Turbidez:** {turbidez_pred:.4f} NTU (normalized)")
        st.caption("This turbidity level is estimated based on your inputs using the XGBoost model.")