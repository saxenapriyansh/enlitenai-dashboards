"""
Seizure Analytics Dashboard
A comprehensive Streamlit application for analyzing 52-week seizure, medication, and assessment data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import io

# ========================================
# PAGE CONFIGURATION
# ========================================

st.set_page_config(
    page_title="Seizure Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS
# ========================================

def apply_custom_css():
    """Apply custom CSS styling for the dashboard."""
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2C3E50;
            padding: 1rem 0;
            border-bottom: 3px solid #3498DB;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #34495E;
            padding: 0.8rem;
            background: linear-gradient(90deg, #ECF0F1 0%, #FFFFFF 100%);
            border-left: 5px solid #3498DB;
            margin: 1.5rem 0 1rem 0;
            border-radius: 5px;
        }
        .subsection-header {
            font-size: 1.3rem;
            font-weight: 500;
            color: #5D6D7E;
            margin: 1rem 0 0.5rem 0;
        }
        .insight-box {
            background-color: #EBF5FB;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #3498DB;
            margin: 0.5rem 0;
            font-size: 0.95rem;
            color: #2C3E50;
        }
        .metric-container {
            background-color: #F8F9F9;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

# ========================================
# DATA PARSING FUNCTIONS
# ========================================

@st.cache_data
def load_csv_data(seizures_file, medications_file, assessments_file):
    """Load CSV files and return DataFrames."""
    try:
        seizures_df = pd.read_csv(seizures_file)
        medications_df = pd.read_csv(medications_file)
        assessments_df = pd.read_csv(assessments_file)
        return seizures_df, medications_df, assessments_df
    except Exception as e:
        st.error(f"Error loading CSV files: {str(e)}")
        return None, None, None

def prepare_medication_dataframe(medications_df):
    """Prepare medication DataFrame in the format expected by the app."""
    # The CSV already has the correct structure, just ensure it has a 'day' column
    if 'day' not in medications_df.columns:
        medications_df['day'] = range(1, len(medications_df) + 1)
    return medications_df

def prepare_assessment_dataframe(assessments_df, total_days=364):
    """Prepare assessment DataFrame with all days, filling missing days with NaN."""
    # Create a complete day range
    all_days = pd.DataFrame({'day': range(1, total_days + 1)})
    
    # Merge with assessments (sparse data)
    full_assessment_df = all_days.merge(
        assessments_df[['day', 'QoL', 'Anxiety', 'Depression', 'Behavioral']], 
        on='day', 
        how='left'
    )
    
    return full_assessment_df

def create_weekly_aggregates(daily_df, med_df, assessment_df):
    """Create weekly aggregated data."""
    # Aggregate seizures by week (using actual Week# from Excel)
    weekly_seizures = daily_df.groupby('week').agg({
        'daily_total': 'sum',
        'daily_severe': 'sum',
        'day': 'count'
    }).reset_index()
    weekly_seizures.columns = ['week', 'total_seizures', 'severe_seizures', 'days_in_week']
    
    # Add week numbers from daily_df to med_df and assessment_df
    # Create a mapping of day number to week number from the actual Excel data
    day_to_week = dict(zip(daily_df['day'], daily_df['week']))
    
    # Aggregate medications by week (using actual Week# mapping)
    med_df_with_week = med_df.copy()
    med_df_with_week['week'] = med_df_with_week['day'].map(day_to_week)
    
    weekly_meds = med_df_with_week.groupby('week')[['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']].mean().reset_index()
    
    # Aggregate assessments by week (using actual Week# mapping)
    assessment_df_with_week = assessment_df.copy()
    assessment_df_with_week['week'] = assessment_df_with_week['day'].map(day_to_week)
    
    weekly_assessments = assessment_df_with_week.groupby('week')[['QoL', 'Anxiety', 'Depression', 'Behavioral']].mean().reset_index()
    
    # Merge all weekly data
    weekly_df = weekly_seizures.merge(weekly_meds, on='week', how='left')
    weekly_df = weekly_df.merge(weekly_assessments, on='week', how='left')
    
    return weekly_df

# ========================================
# VISUALIZATION FUNCTIONS
# ========================================

def plot_daily_seizures(daily_df):
    """Create daily seizure timeline chart."""
    fig = go.Figure()
    
    # Total seizures
    fig.add_trace(go.Scatter(
        x=daily_df['day'],
        y=daily_df['daily_total'],
        mode='lines',
        name='Total Seizures',
        line=dict(color='#E74C3C', width=2),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))
    
    # Severe seizures
    fig.add_trace(go.Scatter(
        x=daily_df['day'],
        y=daily_df['daily_severe'],
        mode='lines',
        name='Severe Seizures',
        line=dict(color='#8E44AD', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(142, 68, 173, 0.1)'
    ))
    
    fig.update_layout(
        title='Daily Seizure Activity Timeline',
        xaxis_title='Day of Year',
        yaxis_title='Number of Seizures',
        hovermode='x unified',
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        xaxis=dict(gridcolor='#ECF0F1'),
        yaxis=dict(gridcolor='#ECF0F1')
    )
    
    return fig

def plot_weekly_seizures(weekly_df):
    """Create weekly seizure burden chart."""
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    fig.add_trace(
        go.Bar(
            x=weekly_df['week'],
            y=weekly_df['total_seizures'],
            name='Total Weekly Seizures',
            marker_color='#3498DB',
            opacity=0.7
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=weekly_df['week'],
            y=weekly_df['severe_seizures'],
            mode='lines+markers',
            name='Severe Seizures',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=8, symbol='diamond')
        )
    )
    
    fig.update_layout(
        title='Weekly Seizure Burden',
        xaxis_title='Week Number',
        yaxis_title='Number of Seizures',
        hovermode='x unified',
        template='plotly_white',
        height=450,
        showlegend=True,
        barmode='overlay',
        xaxis=dict(gridcolor='#ECF0F1'),
        yaxis=dict(gridcolor='#ECF0F1')
    )
    
    return fig

def plot_heatmap_calendar(daily_df):
    """Create 52-week × 7-day heatmap calendar."""
    # Create a matrix for the heatmap
    heatmap_data = np.zeros((52, 7))
    
    for _, row in daily_df.iterrows():
        week_idx = int(row['week']) - 1
        day_idx = int(row['day_of_week']) - 1
        if 0 <= week_idx < 52 and 0 <= day_idx < 7:
            heatmap_data[week_idx, day_idx] = row['daily_total']
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        y=[f'Week {i+1}' for i in range(52)],
        colorscale='YlOrRd',
        colorbar=dict(title='Seizures'),
        hovertemplate='%{y}, %{x}<br>Seizures: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='52-Week Seizure Calendar Heatmap',
        xaxis_title='Day of Week',
        yaxis_title='Week',
        height=800,
        template='plotly_white'
    )
    
    return fig

def plot_medications(med_df):
    """Plot medication dose trajectories."""
    medications = ['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    fig = go.Figure()
    
    for med, color in zip(medications, colors):
        # Filter out NaN values for cleaner visualization
        mask = ~np.isnan(med_df[med])
        fig.add_trace(go.Scatter(
            x=med_df.loc[mask, 'day'],
            y=med_df.loc[mask, med],
            mode='lines',
            name=med,
            line=dict(color=color, width=2.5),
            connectgaps=False
        ))
    
    fig.update_layout(
        title='Medication Dose Trajectories Over Time',
        xaxis_title='Day of Year',
        yaxis_title='Dose (mg)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.9)'),
        xaxis=dict(gridcolor='#ECF0F1'),
        yaxis=dict(gridcolor='#ECF0F1')
    )
    
    return fig

def detect_dose_changes(med_df, threshold=10):
    """Detect medication dose changes."""
    medications = ['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']
    changes = []
    
    for med in medications:
        doses = med_df[med].dropna()
        if len(doses) > 1:
            # Calculate differences
            diff = doses.diff()
            significant_changes = diff[abs(diff) > threshold]
            
            for day_idx, change in significant_changes.items():
                day = med_df.loc[day_idx, 'day']
                change_type = 'Up-titration' if change > 0 else 'Down-titration'
                changes.append({
                    'medication': med,
                    'day': day,
                    'change_type': change_type,
                    'change_amount': change
                })
    
    return pd.DataFrame(changes)

def plot_dose_changes(med_df, changes_df):
    """Plot medication trajectories with dose change markers."""
    medications = ['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    fig = go.Figure()
    
    for med, color in zip(medications, colors):
        mask = ~np.isnan(med_df[med])
        fig.add_trace(go.Scatter(
            x=med_df.loc[mask, 'day'],
            y=med_df.loc[mask, med],
            mode='lines',
            name=med,
            line=dict(color=color, width=2.5),
            connectgaps=False
        ))
    
    # Add vertical lines for dose changes
    for _, change in changes_df.iterrows():
        fig.add_vline(
            x=change['day'],
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"{change['medication'][:3]}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title='Medication Doses with Change Detection',
        xaxis_title='Day of Year',
        yaxis_title='Dose (mg)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        xaxis=dict(gridcolor='#ECF0F1'),
        yaxis=dict(gridcolor='#ECF0F1')
    )
    
    return fig

def plot_dose_response(weekly_df, medication):
    """Create scatter plot of medication dose vs weekly seizures."""
    # Filter out NaN values
    plot_data = weekly_df[['week', medication, 'total_seizures']].dropna()
    
    if len(plot_data) < 2:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=plot_data[medication],
        y=plot_data['total_seizures'],
        mode='markers',
        marker=dict(
            size=10,
            color=plot_data['week'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Week"),
            line=dict(width=1, color='white')
        ),
        text=[f"Week {w}" for w in plot_data['week']],
        hovertemplate='<b>Week %{text}</b><br>Dose: %{x:.1f} mg<br>Seizures: %{y}<extra></extra>'
    ))
    
    # Add trendline
    if len(plot_data) > 1:
        z = np.polyfit(plot_data[medication], plot_data['total_seizures'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(plot_data[medication].min(), plot_data[medication].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=f'{medication} Dose vs Weekly Seizure Frequency',
        xaxis_title=f'{medication} Average Weekly Dose (mg)',
        yaxis_title='Total Weekly Seizures',
        template='plotly_white',
        height=450,
        showlegend=True,
        xaxis=dict(gridcolor='#ECF0F1'),
        yaxis=dict(gridcolor='#ECF0F1')
    )
    
    return fig

def plot_medication_stacked_bar(med_df, granularity='Weekly'):
    """Create stacked bar chart showing medication load over time."""
    medications = ['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    if granularity == 'Weekly':
        # Aggregate by week
        med_df_copy = med_df.copy()
        med_df_copy['week'] = ((med_df_copy['day'] - 1) // 7) + 1
        grouped = med_df_copy.groupby('week')[medications].mean().reset_index()
        x_axis = grouped['week']
        x_label = 'Week Number'
        title = 'Weekly Medication Load (Stacked)'
    elif granularity == 'Monthly':
        # Aggregate by month
        med_df_copy = med_df.copy()
        med_df_copy['month'] = ((med_df_copy['day'] - 1) // 30) + 1
        med_df_copy['month'] = med_df_copy['month'].clip(upper=12)
        grouped = med_df_copy.groupby('month')[medications].mean().reset_index()
        x_axis = grouped['month']
        x_label = 'Month'
        title = 'Monthly Medication Load (Stacked)'
        month_names = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        x_axis = [month_names[int(m)-1] if int(m) <= 12 else 'Aug' for m in x_axis]
    else:  # Daily - sample every 7 days to avoid overcrowding
        sampled = med_df[::7].copy()
        x_axis = sampled['day']
        x_label = 'Day of Year'
        title = 'Daily Medication Load (Stacked, sampled)'
        grouped = sampled
    
    fig = go.Figure()
    
    # Add stacked bars for each medication
    for med, color in zip(medications, colors):
        fig.add_trace(go.Bar(
            x=x_axis,
            y=grouped[med] if granularity != 'Daily' else sampled[med],
            name=med,
            marker_color=color,
            hovertemplate='<b>%{fullData.name}</b><br>Dose: %{y:.1f} mg<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title='Total Medication Dose (mg)',
        barmode='stack',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified',
        xaxis=dict(gridcolor='#ECF0F1'),
        yaxis=dict(gridcolor='#ECF0F1')
    )
    
    return fig

def plot_assessments(assessment_df):
    """Plot assessment time series."""
    assessments = ['QoL', 'Anxiety', 'Depression', 'Behavioral']
    colors = ['#27AE60', '#E67E22', '#9B59B6', '#E74C3C']
    
    fig = go.Figure()
    
    for assessment, color in zip(assessments, colors):
        # Only plot non-NaN values
        mask = ~np.isnan(assessment_df[assessment])
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=assessment_df.loc[mask, 'day'],
                y=assessment_df.loc[mask, assessment],
                mode='lines+markers',
                name=assessment,
                line=dict(color=color, width=2.5),
                marker=dict(size=8),
                connectgaps=False
            ))
    
    fig.update_layout(
        title='Mental Health & Quality of Life Assessments',
        xaxis_title='Day of Year',
        yaxis_title='Assessment Score',
        hovermode='x unified',
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.9)'),
        xaxis=dict(gridcolor='#ECF0F1'),
        yaxis=dict(gridcolor='#ECF0F1')
    )
    
    return fig

def plot_seizure_qol_overlay(weekly_df):
    """Create dual-axis chart with seizures and QoL."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Weekly seizures on primary axis
    fig.add_trace(
        go.Bar(
            x=weekly_df['week'],
            y=weekly_df['total_seizures'],
            name='Weekly Seizures',
            marker_color='#E74C3C',
            opacity=0.6
        ),
        secondary_y=False
    )
    
    # Severe seizures line on primary axis
    fig.add_trace(
        go.Scatter(
            x=weekly_df['week'],
            y=weekly_df['severe_seizures'],
            name='Severe Seizures',
            line=dict(color='#8E44AD', width=3),
            marker=dict(size=8, symbol='diamond'),
            mode='lines+markers'
        ),
        secondary_y=False
    )
    
    # QoL on secondary axis
    qol_data = weekly_df[['week', 'QoL']].dropna()
    if len(qol_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=qol_data['week'],
                y=qol_data['QoL'],
                name='Quality of Life',
                line=dict(color='#27AE60', width=3),
                marker=dict(size=10, symbol='diamond'),
                mode='lines+markers'
            ),
            secondary_y=True
        )
    
    fig.update_xaxes(title_text="Week Number", gridcolor='#ECF0F1')
    fig.update_yaxes(title_text="Weekly Seizures", secondary_y=False, gridcolor='#ECF0F1')
    fig.update_yaxes(title_text="Quality of Life Score", secondary_y=True, gridcolor='#ECF0F1')
    
    fig.update_layout(
        title='Weekly Seizures vs Quality of Life',
        hovermode='x unified',
        template='plotly_white',
        height=450,
        showlegend=True,
        barmode='overlay'
    )
    
    return fig

def plot_correlation_heatmap(weekly_df):
    """Create correlation heatmap for clinical metrics."""
    # Select numeric columns for correlation
    corr_columns = ['total_seizures', 'severe_seizures', 'Lamictal', 'Clonazepam', 
                    'Vimpat', 'Zonisamide', 'Fycompa', 'QoL', 'Anxiety', 'Depression', 'Behavioral']
    
    # Filter to only include columns that exist and have data
    available_columns = [col for col in corr_columns if col in weekly_df.columns]
    corr_data = weekly_df[available_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_data.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title='Clinical Metrics Correlation Matrix',
        template='plotly_white',
        height=600,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig

def plot_integrated_dashboard(daily_df, weekly_df, med_df, assessment_df):
    """Create comprehensive integrated dashboard."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Daily Seizure Activity', 'Medication Trajectories',
                       'Weekly Seizure Burden', 'Quality of Life Trend',
                       'Severe Seizures Pattern', 'Anxiety & Depression'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Row 1, Col 1: Daily seizures
    fig.add_trace(
        go.Scatter(x=daily_df['day'], y=daily_df['daily_total'],
                  mode='lines', name='Daily Seizures',
                  line=dict(color='#E74C3C', width=1.5)),
        row=1, col=1
    )
    
    # Row 1, Col 2: Medications
    medications = ['Lamictal', 'Vimpat', 'Fycompa']
    colors = ['#E74C3C', '#2ECC71', '#9B59B6']
    for med, color in zip(medications, colors):
        mask = ~np.isnan(med_df[med])
        fig.add_trace(
            go.Scatter(x=med_df.loc[mask, 'day'], y=med_df.loc[mask, med],
                      mode='lines', name=med, line=dict(color=color, width=1.5)),
            row=1, col=2
        )
    
    # Row 2, Col 1: Weekly seizures
    fig.add_trace(
        go.Bar(x=weekly_df['week'], y=weekly_df['total_seizures'],
              name='Weekly Total', marker_color='#3498DB', opacity=0.7),
        row=2, col=1
    )
    
    # Row 2, Col 2: QoL
    qol_data = assessment_df[['day', 'QoL']].dropna()
    if len(qol_data) > 0:
        fig.add_trace(
            go.Scatter(x=qol_data['day'], y=qol_data['QoL'],
                      mode='lines+markers', name='QoL',
                      line=dict(color='#27AE60', width=2)),
            row=2, col=2
        )
    
    # Row 3, Col 1: Severe seizures
    fig.add_trace(
        go.Scatter(x=daily_df['day'], y=daily_df['daily_severe'],
                  mode='lines', name='Severe Seizures',
                  line=dict(color='#8E44AD', width=2), fill='tozeroy'),
        row=3, col=1
    )
    
    # Row 3, Col 2: Anxiety & Depression
    for assessment, color in [('Anxiety', '#E67E22'), ('Depression', '#9B59B6')]:
        mask = ~np.isnan(assessment_df[assessment])
        if mask.sum() > 0:
            fig.add_trace(
                go.Scatter(x=assessment_df.loc[mask, 'day'],
                          y=assessment_df.loc[mask, assessment],
                          mode='lines+markers', name=assessment,
                          line=dict(color=color, width=2)),
                row=3, col=2
            )
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        template='plotly_white',
        title_text="Integrated Clinical Dashboard"
    )
    
    fig.update_xaxes(gridcolor='#ECF0F1')
    fig.update_yaxes(gridcolor='#ECF0F1')
    
    return fig

def plot_tachygrid(daily_df):
    """
    Create a GitHub-style contribution map heatmap for seizure activity.
    
    Args:
        daily_df: DataFrame with columns 'week', 'day_of_week', 'daily_total'
    
    Returns:
        Plotly Figure object with interactive hover
    """
    # Create 7x52 grid (days of week x weeks)
    grid = np.zeros((7, 52))
    hover_text = []
    
    # Month labels - approximate week numbers for each month start
    month_positions = [0, 4, 9, 13, 17, 22, 26, 30, 35, 39, 43, 48]
    month_names = ['September', 'October', 'November', 'December', 'January', 'February', 
                   'March', 'April', 'May', 'June', 'July', 'August']
    
    # Fill the grid with seizure data using actual week and day_of_week from Excel
    for idx, row in daily_df.iterrows():
        week = int(row['week']) - 1  # Convert to 0-indexed
        day_of_week = int(row['day_of_week']) - 1  # Convert to 0-indexed (1-7 -> 0-6)
        count = row['daily_total']
        
        # Validate indices are in valid range
        if 0 <= week < 52 and 0 <= day_of_week < 7:
            grid[day_of_week, week] = count
    
    # Create hover text matrix
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create a mapping of (week, day_of_week) -> day number for accurate hover text
    day_mapping = {}
    for idx, row in daily_df.iterrows():
        week = int(row['week']) - 1
        day_of_week = int(row['day_of_week']) - 1
        if 0 <= week < 52 and 0 <= day_of_week < 7:
            day_mapping[(day_of_week, week)] = int(row['day'])
    
    for dow in range(7):
        row_hover = []
        for week in range(52):
            count = int(grid[dow, week])
            day_num = day_mapping.get((dow, week), week * 7 + dow + 1)
            
            # Create richer hover text
            if count == 0:
                hover_str = f"<b>{day_names[dow]}</b><br>Day {day_num}<br><span style='color:#27AE60'>✓ No seizures</span>"
            elif count == 1:
                hover_str = f"<b>{day_names[dow]}</b><br>Day {day_num}<br><span style='color:#E74C3C'>1 seizure</span>"
            else:
                hover_str = f"<b>{day_names[dow]}</b><br>Day {day_num}<br><span style='color:#C0392B'><b>{count} seizures</b></span>"
            row_hover.append(hover_str)
        hover_text.append(row_hover)
    
    # Calculate max for better color scaling
    max_val = np.max(grid)
    
    # Enhanced medical-themed colorscale (blue-to-red gradient for clinical data)
    # Light colors for low values, intense colors for high values
    colorscale = [
        [0, '#F8F9FA'],      # Very light gray for 0 (no seizures)
        [0.1, '#E3F2FD'],    # Very light blue
        [0.2, '#BBDEFB'],    # Light blue
        [0.35, '#90CAF9'],   # Sky blue
        [0.5, '#FFF9C4'],    # Light yellow (transition)
        [0.65, '#FFE082'],   # Yellow
        [0.75, '#FFAB91'],   # Light orange
        [0.85, '#FF7043'],   # Orange
        [0.92, '#E53935'],   # Red
        [1.0, '#B71C1C']     # Dark red (high seizure count)
    ]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=list(range(52)),
        y=['Mon', '', 'Wed', '', 'Fri', '', 'Sun'],
        colorscale=colorscale,
        showscale=True,
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        colorbar=dict(
            title=dict(
                text="<b>Seizures</b><br>per Day",
                side="right",
                font=dict(size=11, family='Arial, sans-serif', color='#2C3E50')
            ),
            thickness=18,
            len=0.6,
            x=1.02,
            tickmode='linear',
            tick0=0,
            dtick=max(1, max_val // 5) if max_val > 0 else 1,
            tickfont=dict(size=10, color='#34495E'),
            outlinewidth=1,
            outlinecolor='#BDC3C7',
            borderwidth=0
        ),
        xgap=2.5,
        ygap=2.5,
    ))
    
    # Add month labels at the top with better styling
    month_annotations = []
    for i, (pos, name) in enumerate(zip(month_positions, month_names)):
        if pos < 52:
            month_annotations.append(
                dict(
                    x=pos,
                    y=7.5,
                    text=f"<b>{name}</b>",
                    showarrow=False,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(size=11, color='#34495E', family='Arial, sans-serif')
                )
            )
    
    # Update layout with modern styling
    fig.update_layout(
        title=dict(
            text='<b>TachyGrid</b> — Seizure Activity Heatmap',
            font=dict(size=18, color='#2C3E50', family='Arial, sans-serif'),
            x=0,
            xanchor='left',
            y=0.98,
            yanchor='top'
        ),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            side='top',
            range=[-0.5, 51.5]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=11, color='#34495E', family='Arial, sans-serif'),
            autorange='reversed',
            fixedrange=True
        ),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        height=250,
        margin=dict(l=60, r=120, t=90, b=30),
        annotations=month_annotations,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial, sans-serif",
            bordercolor="#BDC3C7"
        )
    )
    
    # Remove axis lines for cleaner look
    fig.update_xaxes(showline=False, fixedrange=True)
    fig.update_yaxes(showline=False)
    
    return fig

def plot_activity_heatmap_by_granularity(daily_df, temporal_df, granularity, show_severe_only=False):
    """Create activity heatmap based on selected granularity."""
    
    seizure_col = 'daily_severe' if show_severe_only else 'daily_total'
    seizure_label = "Severe Seizures" if show_severe_only else "Seizures"
    
    if granularity == '24 Hours x Week':
        # Hour of day x Day of week heatmap
        if temporal_df is None or len(temporal_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No temporal data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
            return fig
        
        # Filter for severe if needed
        if show_severe_only:
            temporal_df_filtered = temporal_df[temporal_df['is_severe'] == True].copy()
        else:
            temporal_df_filtered = temporal_df.copy()
        
        # Create hour x day-of-week matrix
        heatmap_data = temporal_df_filtered.groupby(['hour', 'day_of_week']).size().reset_index(name='count')
        matrix = heatmap_data.pivot(index='hour', columns='day_of_week', values='count').fillna(0)
        
        # Ensure all hours and days
        all_hours = list(range(24))
        all_days = list(range(1, 8))
        matrix = matrix.reindex(index=all_hours, columns=all_days, fill_value=0)
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hour_labels = [f"{h:02d}:00" for h in range(24)]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=day_names,
            y=hour_labels,
            colorscale='RdYlBu_r',
            colorbar=dict(title=seizure_label),
            hovertemplate='<b>%{x}</b><br>%{y}<br>'+seizure_label+': %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Hour of Day × Day of Week - {seizure_label}',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            height=600,
            template='plotly_white',
            xaxis=dict(side='top'),
            yaxis=dict(autorange='reversed')
        )
        return fig
    
    elif granularity == 'Day x Week':
        # Day of week x Week heatmap (like the screenshot)
        # Create a matrix: rows = weeks, columns = days of week
        heatmap_data = daily_df.groupby(['week', 'day_of_week'])[seizure_col].sum().reset_index()
        matrix = heatmap_data.pivot(index='week', columns='day_of_week', values=seizure_col).fillna(0)
        
        # Ensure all weeks and days
        all_weeks = list(range(1, 53))
        all_days = list(range(1, 8))
        matrix = matrix.reindex(index=all_weeks, columns=all_days, fill_value=0)
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        week_labels = [f'Week {w}' if w % 2 == 1 else '' for w in range(1, 53)]  # Show odd weeks only
        
        # Enhanced medical-themed colorscale
        colorscale = [
            [0, '#F8F9FA'],
            [0.1, '#E3F2FD'],
            [0.2, '#BBDEFB'],
            [0.35, '#90CAF9'],
            [0.5, '#FFF9C4'],
            [0.65, '#FFE082'],
            [0.75, '#FFAB91'],
            [0.85, '#FF7043'],
            [0.92, '#E53935'],
            [1.0, '#B71C1C']
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=day_names,
            y=week_labels,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text=f"<b>{seizure_label}</b>", font=dict(size=11)),
                thickness=18,
                len=0.7
            ),
            hovertemplate='<b>Week %{y}</b><br>%{x}<br>'+seizure_label+': %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Week × Day of Week Heatmap - {seizure_label}',
            xaxis_title='Day of Week',
            yaxis_title='Week',
            height=700,
            template='plotly_white',
            xaxis=dict(side='bottom', tickfont=dict(size=11)),
            yaxis=dict(autorange='reversed', tickfont=dict(size=9))
        )
        return fig
    
    elif granularity == 'Day x Month':
        # Day of month x Month heatmap
        daily_df_copy = daily_df.copy()
        daily_df_copy['month'] = ((daily_df_copy['day'] - 1) // 30) + 1
        daily_df_copy['month'] = daily_df_copy['month'].clip(upper=12)
        daily_df_copy['day_of_month'] = ((daily_df_copy['day'] - 1) % 30) + 1
        
        heatmap_data = daily_df_copy.groupby(['day_of_month', 'month'])[seizure_col].sum().reset_index()
        matrix = heatmap_data.pivot(index='day_of_month', columns='month', values=seizure_col).fillna(0)
        
        month_names = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        day_labels = [str(d) if d % 5 == 1 else '' for d in range(1, 31)]
        
        colorscale = [
            [0, '#F8F9FA'],
            [0.1, '#E3F2FD'],
            [0.2, '#BBDEFB'],
            [0.35, '#90CAF9'],
            [0.5, '#FFF9C4'],
            [0.65, '#FFE082'],
            [0.75, '#FFAB91'],
            [0.85, '#FF7043'],
            [0.92, '#E53935'],
            [1.0, '#B71C1C']
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=month_names[:len(matrix.columns)],
            y=day_labels,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text=f"<b>{seizure_label}</b>", font=dict(size=11)),
                thickness=18,
                len=0.7
            ),
            hovertemplate='<b>%{x}</b><br>Day %{y}<br>'+seizure_label+': %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Day of Month × Month Heatmap - {seizure_label}',
            xaxis_title='Month',
            yaxis_title='Day of Month',
            height=600,
            template='plotly_white',
            xaxis=dict(side='top', tickangle=0),
            yaxis=dict(autorange='reversed')
        )
        return fig
    
    else:  # 'Week x Month'
        # Week of month x Month heatmap
        daily_df_copy = daily_df.copy()
        daily_df_copy['month'] = ((daily_df_copy['day'] - 1) // 30) + 1
        daily_df_copy['month'] = daily_df_copy['month'].clip(upper=12)
        daily_df_copy['week_of_month'] = ((daily_df_copy['day'] - 1) % 30) // 7 + 1
        
        heatmap_data = daily_df_copy.groupby(['week_of_month', 'month'])[seizure_col].sum().reset_index()
        matrix = heatmap_data.pivot(index='week_of_month', columns='month', values=seizure_col).fillna(0)
        
        month_names = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        week_labels = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']
        
        colorscale = [
            [0, '#F8F9FA'],
            [0.1, '#E3F2FD'],
            [0.2, '#BBDEFB'],
            [0.35, '#90CAF9'],
            [0.5, '#FFF9C4'],
            [0.65, '#FFE082'],
            [0.75, '#FFAB91'],
            [0.85, '#FF7043'],
            [0.92, '#E53935'],
            [1.0, '#B71C1C']
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=month_names[:len(matrix.columns)],
            y=week_labels[:len(matrix.index)],
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text=f"<b>{seizure_label}</b>", font=dict(size=11)),
                thickness=18,
                len=0.7
            ),
            hovertemplate='<b>%{x}</b><br>%{y}<br>'+seizure_label+': %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Week of Month × Month Heatmap - {seizure_label}',
            xaxis_title='Month',
            yaxis_title='Week of Month',
            height=400,
            template='plotly_white',
            xaxis=dict(side='top', tickangle=0),
            yaxis=dict(autorange='reversed')
        )
        return fig

def generate_temporal_seizure_data(daily_df, num_seizures_per_day=None):
    """
    Generate realistic temporal (time-of-day) data for seizures.
    Uses dummy data with realistic circadian patterns.
    
    Args:
        daily_df: DataFrame with daily seizure totals
        num_seizures_per_day: Optional override for seizure distribution
    
    Returns:
        DataFrame with columns: day, hour, minute, week, day_of_week, is_severe
    """
    np.random.seed(42)  # For reproducibility
    
    temporal_data = []
    
    # Create realistic hourly distribution (some hours more likely than others)
    # Typical seizure patterns: peaks in early morning (5-8am) and late evening (8-11pm)
    hourly_weights = {
        0: 0.5, 1: 0.3, 2: 0.2, 3: 0.3, 4: 0.5,
        5: 1.2, 6: 1.5, 7: 1.8, 8: 1.3, 9: 1.0,
        10: 0.8, 11: 0.7, 12: 0.8, 13: 0.7, 14: 0.6,
        15: 0.7, 16: 0.8, 17: 0.9, 18: 1.0, 19: 1.2,
        20: 1.5, 21: 1.7, 22: 1.4, 23: 0.8
    }
    
    for idx, row in daily_df.iterrows():
        day = row['day']
        total_seizures = int(row['daily_total'])
        severe_seizures = int(row['daily_severe'])
        
        if total_seizures > 0:
            # Distribute seizures across hours based on weights
            hours = np.random.choice(
                list(range(24)), 
                size=total_seizures,
                p=[hourly_weights[h]/sum(hourly_weights.values()) for h in range(24)]
            )
            
            # Mark which ones are severe (first severe_seizures count)
            severe_flags = [True] * severe_seizures + [False] * (total_seizures - severe_seizures)
            np.random.shuffle(severe_flags)
            
            for hour, is_severe in zip(hours, severe_flags):
                minute = np.random.randint(0, 60)
                temporal_data.append({
                    'day': day,
                    'hour': hour,
                    'minute': minute,
                    'week': row['week'],
                    'day_of_week': row['day_of_week'],
                    'is_severe': is_severe
                })
    
    return pd.DataFrame(temporal_data)

def plot_circadian_polar(temporal_df, granularity='hour', show_severe_only=False):
    """
    Create a polar/radial bar chart showing seizure distribution by time.
    
    Args:
        temporal_df: DataFrame with temporal seizure data
        granularity: 'hour' or '6hour'
        show_severe_only: If True, plot only severe seizures
    
    Returns:
        Plotly Figure object
    """
    # Filter for severe seizures if requested
    if show_severe_only:
        temporal_df = temporal_df[temporal_df['is_severe'] == True].copy()
    
    if len(temporal_df) == 0:
        # No seizures - create empty chart
        fig = go.Figure()
        seizure_type = "severe seizure" if show_severe_only else "seizure"
        fig.add_annotation(
            text=f"No {seizure_type} data available for temporal analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
    
    # Aggregate by hour
    hourly_counts = temporal_df.groupby('hour').size().reset_index(name='count')
    
    # Create complete 24-hour range
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly_counts = all_hours.merge(hourly_counts, on='hour', how='left').fillna(0)
    
    # Apply granularity
    if granularity == '6hour':
        hourly_counts['hour_group'] = (hourly_counts['hour'] // 6) * 6
        plot_data = hourly_counts.groupby('hour_group')['count'].sum().reset_index()
        plot_data.columns = ['hour', 'count']
        period_names = ['Night', 'Morning', 'Afternoon', 'Evening']
        theta_labels = [f"{period_names[i]}<br>({h:02d}:00-{(h+6):02d}:00)" 
                       for i, h in enumerate(plot_data['hour'])]
    else:  # hourly
        plot_data = hourly_counts
        # Create 12-hour format labels
        theta_labels = []
        for h in plot_data['hour']:
            if h == 0:
                theta_labels.append("12a")
            elif h < 12:
                theta_labels.append(f"{h}a")
            elif h == 12:
                theta_labels.append("12p")
            else:
                theta_labels.append(f"{h-12}p")
    
    # Calculate theta (angles) for polar plot
    theta = plot_data['hour'] * 360 / 24  # Convert hours to degrees
    
    # Create color scale based on intensity
    max_count = plot_data['count'].max()
    if max_count > 0:
        colors = plot_data['count'] / max_count
    else:
        colors = [0] * len(plot_data)
    
    # Create the polar bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Barpolar(
        r=plot_data['count'],
        theta=theta,
        width=[360/len(plot_data)] * len(plot_data),
        marker=dict(
            color=colors,
            colorscale='Blues',
            cmin=0,
            cmax=1,
            colorbar=dict(
                title="Relative<br>Intensity",
                thickness=15,
                len=0.5,
                x=1.15
            ),
            line=dict(color='white', width=2)
        ),
        text=[f"{int(c)} seizures" if c != 1 else "1 seizure" for c in plot_data['count']],
        hovertemplate='<b>%{text}</b><br>Time: ' + 
                     '<br>'.join([f'{label}' for label in theta_labels]) +
                     '<extra></extra>',
        opacity=0.8
    ))
    
    # Update layout for clock-like appearance
    # Create title based on seizure type
    seizure_type = "Severe Seizure" if show_severe_only else "Seizure"
    gran_text = granularity.replace("hour", "Hour").title()
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=True,
                gridcolor='rgba(200, 200, 200, 0.3)',
                tickfont=dict(size=10, color='#666')
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=theta,
                ticktext=theta_labels,
                direction='clockwise',
                rotation=90,  # Start at top (12 o'clock)
                tickfont=dict(size=11, color='#2C3E50', family='Arial, sans-serif'),
                gridcolor='rgba(200, 200, 200, 0.3)',
                linecolor='rgba(150, 150, 150, 0.5)'
            ),
            bgcolor='rgba(240, 240, 250, 0.3)'
        ),
        showlegend=False,
        title=dict(
            text=f'Circadian {seizure_type} Pattern - {gran_text} View',
            font=dict(size=16, color='#2C3E50', family='Arial, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        height=600,
        margin=dict(l=80, r=120, t=100, b=80),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def plot_by_granularity(daily_df, temporal_df, granularity, show_severe_only=False):
    """
    Create visualization based on selected granularity.
    Routes to appropriate chart type.
    
    Args:
        daily_df: Daily seizure DataFrame
        temporal_df: Temporal seizure DataFrame
        granularity: Time scale ('hour', '6hour', 'day', 'week', 'month', 'year')
        show_severe_only: If True, plot only severe seizures
    """
    if granularity in ['hour', '6hour']:
        # Hourly circadian patterns
        return plot_circadian_polar(temporal_df, granularity, show_severe_only)
    elif granularity == 'day':
        # Day of week distribution
        return plot_day_of_week_polar(daily_df, show_severe_only)
    elif granularity == 'week':
        # Weekly distribution
        return plot_weekly_polar(daily_df, show_severe_only)
    elif granularity == 'month':
        # Monthly distribution
        return plot_monthly_polar(daily_df, show_severe_only)
    elif granularity == 'year':
        # Yearly overview
        return plot_yearly_summary(daily_df, show_severe_only)
    
def plot_day_of_week_polar(daily_df, show_severe_only=False):
    """Create polar chart for day of week distribution."""
    # Select which column to aggregate
    seizure_col = 'daily_severe' if show_severe_only else 'daily_total'
    seizure_type = "Severe Seizure" if show_severe_only else "Seizure"
    
    # Aggregate by day of week
    dow_counts = daily_df.groupby('day_of_week')[seizure_col].sum().reset_index()
    
    # Ensure all days present
    all_days = pd.DataFrame({'day_of_week': range(1, 8)})
    dow_counts = all_days.merge(dow_counts, on='day_of_week', how='left').fillna(0)
    dow_counts.columns = ['day_of_week', 'count']  # Rename for consistency
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts['day_name'] = day_names
    
    # Create polar chart
    theta = (dow_counts['day_of_week'] - 1) * 360 / 7
    
    fig = go.Figure()
    
    colors = dow_counts['count'] / dow_counts['count'].max() if dow_counts['count'].max() > 0 else [0]*7
    
    fig.add_trace(go.Barpolar(
        r=dow_counts['count'],
        theta=theta,
        width=[360/7] * 7,
        marker=dict(
            color=colors,
            colorscale='Reds',
            cmin=0,
            cmax=1,
            colorbar=dict(title="Relative<br>Intensity", thickness=15, len=0.5, x=1.15),
            line=dict(color='white', width=2)
        ),
        text=[f"{int(c)} {'severe ' if show_severe_only else ''}seizures" for c in dow_counts['count']],
        hovertemplate='<b>%{customdata}</b><br>%{text}<extra></extra>',
        customdata=day_names,
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, gridcolor='rgba(200,200,200,0.3)'),
            angularaxis=dict(
                tickmode='array',
                tickvals=theta,
                ticktext=day_names,
                direction='clockwise',
                rotation=90,
                tickfont=dict(size=12, color='#2C3E50'),
                gridcolor='rgba(200,200,200,0.3)'
            ),
            bgcolor='rgba(240,240,250,0.3)'
        ),
        title=f'Day of Week {seizure_type} Distribution',
        height=600,
        margin=dict(l=80, r=120, t=100, b=80)
    )
    
    return fig

def plot_weekly_polar(daily_df, show_severe_only=False):
    """Create polar chart for weekly distribution across 52 weeks."""
    # Select which column to aggregate
    seizure_col = 'daily_severe' if show_severe_only else 'daily_total'
    seizure_type = "Severe Seizure" if show_severe_only else "Seizure"
    
    # Aggregate by week
    weekly_counts = daily_df.groupby('week')[seizure_col].sum().reset_index()
    weekly_counts.columns = ['week', 'count']
    
    # Create polar chart
    theta = (weekly_counts['week'] - 1) * 360 / 52
    
    fig = go.Figure()
    
    colors = weekly_counts['count'] / weekly_counts['count'].max() if weekly_counts['count'].max() > 0 else [0]*52
    
    fig.add_trace(go.Barpolar(
        r=weekly_counts['count'],
        theta=theta,
        width=[360/52] * 52,
        marker=dict(
            color=colors,
            colorscale='Greens',
            cmin=0,
            cmax=1,
            colorbar=dict(title="Relative<br>Intensity", thickness=15, len=0.5, x=1.15),
            line=dict(color='white', width=1)
        ),
        text=[f"Week {w}: {int(c)} {'severe ' if show_severe_only else ''}seizures" for w, c in zip(weekly_counts['week'], weekly_counts['count'])],
        hovertemplate='%{text}<extra></extra>',
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, gridcolor='rgba(200,200,200,0.3)'),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 90, 180, 270],
                ticktext=['Week 1', 'Week 13', 'Week 26', 'Week 39'],
                direction='clockwise',
                rotation=90,
                tickfont=dict(size=11, color='#2C3E50'),
                gridcolor='rgba(200,200,200,0.3)'
            ),
            bgcolor='rgba(240,240,250,0.3)'
        ),
        title=f'52-Week {seizure_type} Distribution',
        height=600,
        margin=dict(l=80, r=120, t=100, b=80)
    )
    
    return fig

def plot_monthly_polar(daily_df, show_severe_only=False):
    """Create polar chart for monthly distribution."""
    # Select which column to aggregate
    seizure_col = 'daily_severe' if show_severe_only else 'daily_total'
    seizure_type = "Severe Seizure" if show_severe_only else "Seizure"
    
    # Add month column (assuming 30-31 days per month, approximate)
    daily_df_copy = daily_df.copy()
    daily_df_copy['month'] = ((daily_df_copy['day'] - 1) // 30) + 1
    daily_df_copy['month'] = daily_df_copy['month'].clip(upper=12)
    
    # Aggregate by month
    monthly_counts = daily_df_copy.groupby('month')[seizure_col].sum().reset_index()
    monthly_counts.columns = ['month', 'count']
    
    month_names = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
    monthly_counts['month_name'] = [month_names[i-1] if i <= 12 else 'Dec' for i in monthly_counts['month']]
    
    # Create polar chart
    theta = (monthly_counts['month'] - 1) * 360 / 12
    
    fig = go.Figure()
    
    colors = monthly_counts['count'] / monthly_counts['count'].max() if monthly_counts['count'].max() > 0 else [0]*len(monthly_counts)
    
    fig.add_trace(go.Barpolar(
        r=monthly_counts['count'],
        theta=theta,
        width=[360/12] * len(monthly_counts),
        marker=dict(
            color=colors,
            colorscale='Oranges',
            cmin=0,
            cmax=1,
            colorbar=dict(title="Relative<br>Intensity", thickness=15, len=0.5, x=1.15),
            line=dict(color='white', width=2)
        ),
        text=[f"{name}: {int(c)} {'severe ' if show_severe_only else ''}seizures" for name, c in zip(monthly_counts['month_name'], monthly_counts['count'])],
        hovertemplate='%{text}<extra></extra>',
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, gridcolor='rgba(200,200,200,0.3)'),
            angularaxis=dict(
                tickmode='array',
                tickvals=theta,
                ticktext=monthly_counts['month_name'],
                direction='clockwise',
                rotation=90,
                tickfont=dict(size=12, color='#2C3E50', family='Arial, sans-serif'),
                gridcolor='rgba(200,200,200,0.3)'
            ),
            bgcolor='rgba(240,240,250,0.3)'
        ),
        title=f'Monthly {seizure_type} Distribution',
        height=600,
        margin=dict(l=80, r=120, t=100, b=80)
    )
    
    return fig

def plot_yearly_summary(daily_df, show_severe_only=False):
    """Create comprehensive yearly summary chart."""
    # Select which column to aggregate
    seizure_col = 'daily_severe' if show_severe_only else 'daily_total'
    seizure_type = "Severe Seizure" if show_severe_only else "Seizure"
    
    # Create quarters
    daily_df_copy = daily_df.copy()
    daily_df_copy['quarter'] = ((daily_df_copy['day'] - 1) // 91) + 1
    daily_df_copy['quarter'] = daily_df_copy['quarter'].clip(upper=4)
    
    # Aggregate by quarter
    quarterly_counts = daily_df_copy.groupby('quarter')[seizure_col].sum().reset_index()
    quarterly_counts.columns = ['quarter', 'count']
    
    quarter_names = ['Q1 (Sep-Nov)', 'Q2 (Dec-Feb)', 'Q3 (Mar-May)', 'Q4 (Jun-Aug)']
    quarterly_counts['quarter_name'] = [quarter_names[i-1] for i in quarterly_counts['quarter']]
    
    # Create polar chart
    theta = (quarterly_counts['quarter'] - 1) * 360 / 4
    
    fig = go.Figure()
    
    colors = quarterly_counts['count'] / quarterly_counts['count'].max() if quarterly_counts['count'].max() > 0 else [0]*4
    
    fig.add_trace(go.Barpolar(
        r=quarterly_counts['count'],
        theta=theta,
        width=[360/4] * 4,
        marker=dict(
            color=colors,
            colorscale='Purples',
            cmin=0,
            cmax=1,
            colorbar=dict(title="Relative<br>Intensity", thickness=15, len=0.5, x=1.15),
            line=dict(color='white', width=3)
        ),
        text=[f"{name}<br>{int(c)} {'severe ' if show_severe_only else ''}seizures" for name, c in zip(quarterly_counts['quarter_name'], quarterly_counts['count'])],
        hovertemplate='%{text}<extra></extra>',
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, gridcolor='rgba(200,200,200,0.3)', tickfont=dict(size=11)),
            angularaxis=dict(
                tickmode='array',
                tickvals=theta,
                ticktext=[qn.split('(')[0].strip() for qn in quarterly_counts['quarter_name']],
                direction='clockwise',
                rotation=90,
                tickfont=dict(size=13, color='#2C3E50', family='Arial, sans-serif', weight='bold'),
                gridcolor='rgba(200,200,200,0.3)'
            ),
            bgcolor='rgba(240,240,250,0.3)'
        ),
        title=f'Yearly Overview - Quarterly {seizure_type} Distribution',
        height=600,
        margin=dict(l=80, r=120, t=100, b=80)
    )
    
    # Add center annotation
    total_seizures = daily_df[seizure_col].sum()
    seizure_label = "Severe" if show_severe_only else "Total"
    fig.add_annotation(
        text=f"<b>{int(total_seizures)}</b><br>{seizure_label}<br>Seizures",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color='#2C3E50'),
        align='center'
    )
    
    return fig

def plot_day_of_week_distribution(temporal_df):
    """Create bar chart showing seizure distribution by day of week."""
    if len(temporal_df) == 0:
        return None
    
    # Count by day of week
    dow_counts = temporal_df.groupby('day_of_week').size().reset_index(name='count')
    
    # Ensure all days are present
    all_days = pd.DataFrame({'day_of_week': range(1, 8)})
    dow_counts = all_days.merge(dow_counts, on='day_of_week', how='left').fillna(0)
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts['day_name'] = [day_names[i-1] for i in dow_counts['day_of_week']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dow_counts['day_name'],
        y=dow_counts['count'],
        marker=dict(
            color=dow_counts['count'],
            colorscale='Viridis',
            showscale=False,
            line=dict(color='white', width=2)
        ),
        text=[f"{int(c)}" for c in dow_counts['count']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Seizures: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Seizure Distribution by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Total Seizures',
        template='plotly_white',
        height=400,
        showlegend=False,
        xaxis=dict(tickangle=0),
        yaxis=dict(gridcolor='#ECF0F1')
    )
    
    return fig

def plot_weekly_temporal_heatmap(temporal_df):
    """Create heatmap showing seizures by hour and day of week."""
    if len(temporal_df) == 0:
        return None
    
    # Create hour x day-of-week matrix
    heatmap_data = temporal_df.groupby(['hour', 'day_of_week']).size().reset_index(name='count')
    
    # Pivot to create matrix
    matrix = heatmap_data.pivot(index='hour', columns='day_of_week', values='count').fillna(0)
    
    # Ensure all hours and days are present
    all_hours = list(range(24))
    all_days = list(range(1, 8))
    matrix = matrix.reindex(index=all_hours, columns=all_days, fill_value=0)
    
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=day_names,
        y=hour_labels,
        colorscale='YlOrRd',
        colorbar=dict(title='Seizures'),
        hovertemplate='<b>%{x}</b><br>%{y}<br>Seizures: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Seizure Heatmap: Hour of Day × Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Hour of Day',
        height=700,
        template='plotly_white',
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig

# ========================================
# HELPER FUNCTIONS FOR CONTROLS
# ========================================

def create_time_granularity_selector(key_suffix=""):
    """Create a time granularity selector for charts."""
    col1, col2 = st.columns([1, 3])
    with col1:
        granularity = st.selectbox(
            "View by:",
            options=['24 Hours', 'Week (7 Days)', 'Month (12 Months)', '365 Days', '52 Weeks'],
            key=f'granularity_{key_suffix}'
        )
    return granularity

def aggregate_by_granularity(daily_df, granularity):
    """Aggregate daily data based on selected granularity."""
    if granularity in ['365 Days', '24 Hours (Circadian)']:
        return daily_df
    elif granularity == '52 Weeks':
        weekly = daily_df.groupby('week').agg({
            'daily_total': 'sum',
            'daily_severe': 'sum',
            'day': 'first'
        }).reset_index()
        weekly.columns = ['week', 'total', 'severe', 'first_day']
        return weekly
    elif granularity == 'Week (7 Days)':
        # Aggregate by day of week across the year
        dow_data = daily_df.groupby('day_of_week').agg({
            'daily_total': 'sum',
            'daily_severe': 'sum'
        }).reset_index()
        dow_data.columns = ['day_of_week', 'total', 'severe']
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_data['day_name'] = [day_names[int(d)-1] for d in dow_data['day_of_week']]
        return dow_data
    else:  # 'Month (12 Months)'
        daily_df_copy = daily_df.copy()
        daily_df_copy['month'] = ((daily_df_copy['day'] - 1) // 30) + 1
        daily_df_copy['month'] = daily_df_copy['month'].clip(upper=12)
        monthly = daily_df_copy.groupby('month').agg({
            'daily_total': 'sum',
            'daily_severe': 'sum',
            'day': 'first'
        }).reset_index()
        monthly.columns = ['month', 'total', 'severe', 'first_day']
        month_names = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        monthly['month_name'] = [month_names[int(m)-1] if int(m) <= 12 else 'Aug' for m in monthly['month']]
        return monthly

def plot_timeline_by_view(daily_df, weekly_df, temporal_df, granularity, show_severe_only=False):
    """Create timeline visualization based on selected granularity."""
    
    if granularity == '24 Hours (Circadian)':
        # Show hourly distribution as bar chart
        if temporal_df is None or len(temporal_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No seizure data available for temporal analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
            return fig
        
        # Filter for severe seizures if requested
        if show_severe_only:
            temporal_df_filtered = temporal_df[temporal_df['is_severe'] == True].copy()
        else:
            temporal_df_filtered = temporal_df.copy()
        
        # Aggregate by hour
        hourly_counts = temporal_df_filtered.groupby('hour').size().reset_index(name='count')
        
        # Create complete 24-hour range
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_counts = all_hours.merge(hourly_counts, on='hour', how='left').fillna(0)
        
        # Also get severe seizures for overlay if showing all seizures
        if not show_severe_only:
            severe_hourly = temporal_df[temporal_df['is_severe'] == True].groupby('hour').size().reset_index(name='severe_count')
            hourly_counts = hourly_counts.merge(severe_hourly, on='hour', how='left').fillna(0)
        
        # Create hour labels
        hour_labels = []
        for h in hourly_counts['hour']:
            if h == 0:
                hour_labels.append("12am")
            elif h < 12:
                hour_labels.append(f"{h}am")
            elif h == 12:
                hour_labels.append("12pm")
            else:
                hour_labels.append(f"{h-12}pm")
        
        fig = go.Figure()
        
        # Add bar chart for total (or severe only)
        fig.add_trace(go.Bar(
            x=hour_labels,
            y=hourly_counts['count'],
            name='Severe Seizures' if show_severe_only else 'Total Seizures',
            marker_color='#E74C3C' if show_severe_only else '#3498DB',
            opacity=0.7,
            text=hourly_counts['count'].astype(int),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Seizures: %{y}<extra></extra>'
        ))
        
        # Add line for severe seizures if showing all
        if not show_severe_only:
            fig.add_trace(go.Scatter(
                x=hour_labels,
                y=hourly_counts['severe_count'],
                mode='lines+markers',
                name='Severe Seizures',
                line=dict(color='#E74C3C', width=3),
                marker=dict(size=8, symbol='diamond')
            ))
        
        seizure_label = "Severe Seizures" if show_severe_only else "Total Seizures"
        fig.update_layout(
            title=f'24-Hour Circadian Pattern - {seizure_label}',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Seizures',
            template='plotly_white',
            height=500,
            showlegend=True,
            xaxis=dict(tickangle=45, gridcolor='#ECF0F1'),
            yaxis=dict(gridcolor='#ECF0F1')
        )
        return fig
    
    elif granularity == 'Week (7 Days)':
        # Show day of week distribution
        seizure_col = 'daily_severe' if show_severe_only else 'daily_total'
        dow_data = aggregate_by_granularity(daily_df, granularity)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dow_data['day_name'],
            y=dow_data['severe' if show_severe_only else 'total'],
            marker=dict(
                color=dow_data['severe' if show_severe_only else 'total'],
                colorscale='Reds' if show_severe_only else 'Blues',
                showscale=False
            ),
            text=dow_data['severe' if show_severe_only else 'total'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Seizures: %{y}<extra></extra>'
        ))
        
        seizure_label = "Severe Seizures" if show_severe_only else "Total Seizures"
        fig.update_layout(
            title=f'Weekly Pattern - {seizure_label} by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Total Seizures',
            template='plotly_white',
            height=500,
            xaxis=dict(tickangle=0),
            yaxis=dict(gridcolor='#ECF0F1')
        )
        return fig
    
    elif granularity == 'Month (12 Months)':
        # Show monthly distribution
        monthly_data = aggregate_by_granularity(daily_df, granularity)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_data['month_name'],
            y=monthly_data['severe' if show_severe_only else 'total'],
            name='Severe Seizures' if show_severe_only else 'Total Seizures',
            marker_color='#E74C3C' if show_severe_only else '#3498DB',
            opacity=0.7,
            text=monthly_data['severe' if show_severe_only else 'total'],
            textposition='outside'
        ))
        
        if not show_severe_only:
            fig.add_trace(go.Scatter(
                x=monthly_data['month_name'],
                y=monthly_data['severe'],
                mode='lines+markers',
                name='Severe Seizures',
                line=dict(color='#E74C3C', width=3),
                marker=dict(size=10, symbol='diamond')
            ))
        
        seizure_label = "Severe Seizures" if show_severe_only else "Total Seizures"
        fig.update_layout(
            title=f'Monthly Seizure Pattern - {seizure_label}',
            xaxis_title='Month',
            yaxis_title='Number of Seizures',
            template='plotly_white',
            height=500,
            xaxis=dict(tickangle=45)
        )
        return fig
    
    elif granularity == '365 Days':
        # Show daily timeline
        seizure_col = 'daily_severe' if show_severe_only else 'daily_total'
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_df['day'],
            y=daily_df[seizure_col],
            mode='lines',
            name='Severe Seizures' if show_severe_only else 'Total Seizures',
            line=dict(color='#8E44AD' if show_severe_only else '#E74C3C', width=2),
            fill='tozeroy',
            fillcolor=f'rgba({"142, 68, 173" if show_severe_only else "231, 76, 60"}, 0.1)'
        ))
        
        if not show_severe_only:
            fig.add_trace(go.Scatter(
                x=daily_df['day'],
                y=daily_df['daily_severe'],
                mode='lines',
                name='Severe Seizures',
                line=dict(color='#8E44AD', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(142, 68, 173, 0.1)'
            ))
        
        seizure_label = "Severe Seizures" if show_severe_only else "All Seizures"
        fig.update_layout(
            title=f'365-Day Seizure Timeline - {seizure_label}',
            xaxis_title='Day of Year',
            yaxis_title='Number of Seizures',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            xaxis=dict(gridcolor='#ECF0F1'),
            yaxis=dict(gridcolor='#ECF0F1')
        )
        return fig
    
    else:  # '52 Weeks'
        # Show weekly timeline
        seizure_col = 'severe_seizures' if show_severe_only else 'total_seizures'
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        fig.add_trace(
            go.Bar(
                x=weekly_df['week'],
                y=weekly_df[seizure_col],
                name='Severe Seizures' if show_severe_only else 'Total Weekly Seizures',
                marker_color='#8E44AD' if show_severe_only else '#3498DB',
                opacity=0.7
            )
        )
        
        if not show_severe_only:
            fig.add_trace(
                go.Scatter(
                    x=weekly_df['week'],
                    y=weekly_df['severe_seizures'],
                    mode='lines+markers',
                    name='Severe Seizures',
                    line=dict(color='#E74C3C', width=3),
                    marker=dict(size=8, symbol='diamond')
                )
            )
        
        seizure_label = "Severe Seizures" if show_severe_only else "All Seizures"
        fig.update_layout(
            title=f'52-Week Seizure Timeline - {seizure_label}',
            xaxis_title='Week Number',
            yaxis_title='Number of Seizures',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            barmode='overlay',
            xaxis=dict(gridcolor='#ECF0F1'),
            yaxis=dict(gridcolor='#ECF0F1')
        )
        return fig

def calculate_clinical_alerts(daily_df, weekly_df):
    """Calculate clinical alerts and flags for physician attention."""
    alerts = []
    
    # High seizure burden
    recent_week = weekly_df.iloc[-1] if len(weekly_df) > 0 else None
    if recent_week is not None and recent_week['total_seizures'] > 15:
        alerts.append({
            'type': 'warning',
            'title': 'High Recent Seizure Burden',
            'message': f"Week {recent_week['week']}: {int(recent_week['total_seizures'])} seizures"
        })
    
    # Increasing trend
    if len(weekly_df) >= 4:
        last_4_weeks = weekly_df.tail(4)['total_seizures'].tolist()
        if len(last_4_weeks) == 4 and last_4_weeks[-1] > last_4_weeks[0] * 1.5:
            alerts.append({
                'type': 'warning',
                'title': 'Increasing Seizure Trend',
                'message': 'Seizure frequency has increased significantly in recent weeks'
            })
    
    # Severe seizure cluster
    recent_7_days = daily_df.tail(7)
    severe_in_week = recent_7_days['daily_severe'].sum()
    if severe_in_week >= 3:
        alerts.append({
            'type': 'danger',
            'title': 'Severe Seizure Cluster',
            'message': f'{int(severe_in_week)} severe seizures in the past 7 days'
        })
    
    # Good control (no alerts needed, but info)
    if len(alerts) == 0 and recent_week is not None and recent_week['total_seizures'] < 5:
        alerts.append({
            'type': 'success',
            'title': 'Good Seizure Control',
            'message': f'Only {int(recent_week["total_seizures"])} seizures in the most recent week'
        })
    
    return alerts

# ========================================
# APP LAYOUT FUNCTIONS
# ========================================

def sidebar_menu():
    """Create sidebar navigation menu."""
    st.sidebar.markdown("## 👤 Patient Selection")
    
    # Get available patients from session state
    if 'available_patients' in st.session_state:
        patients = st.session_state.available_patients
        patient_labels = {p: f"Patient {p}" for p in patients}
        
        # Patient selector
        selected_patient = st.sidebar.selectbox(
            "Select Patient ID:",
            options=patients,
            format_func=lambda x: patient_labels[x],
            key='selected_patient'
        )
        
        # Display patient summary if available
        if 'patient_summaries' in st.session_state and selected_patient in st.session_state.patient_summaries:
            summary = st.session_state.patient_summaries[selected_patient]
            severity_rate = (summary['severe_seizures'] / summary['total_seizures'] * 100) if summary['total_seizures'] > 0 else 0
            avg_per_week = summary['total_seizures'] / 52
            
            st.sidebar.markdown(f"""
            <div style="background-color: #EBF5FB; padding: 0.8rem; border-radius: 5px; margin-bottom: 1rem;">
                <b>📊 Patient Summary:</b><br>
                Total Seizures: <b>{summary['total_seizures']}</b><br>
                Severe: <b>{summary['severe_seizures']}</b> ({severity_rate:.1f}%)<br>
                Avg/Week: <b>{avg_per_week:.1f}</b><br>
                Period: {summary['days']} days
            </div>
            """, unsafe_allow_html=True)
    else:
        selected_patient = None
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🏥 Clinical Dashboard")
    
    menu_options = [
        "🏠 Main Dashboard",
        "📈 Seizure Analysis",
        "💊 Treatment Management",
        "🧠 Patient Outcomes",
        "📋 Clinical Report"
    ]
    
    selection = st.sidebar.radio("Navigate to:", menu_options, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "**Clinical Decision Support System**\n\n"
        "Comprehensive 52-week epilepsy monitoring analytics for "
        "clinical decision-making and treatment optimization."
    )
    
    return selection, selected_patient

def render_main_dashboard(daily_df, weekly_df, med_df, assessment_df):
    """Render Main Dashboard - Physician-focused quick overview."""
    st.markdown('<div class="main-header">🏠 Clinical Dashboard - Patient Overview</div>', unsafe_allow_html=True)
    
    # Clinical Alerts Section
    alerts = calculate_clinical_alerts(daily_df, weekly_df)
    if alerts:
        st.markdown('<div class="subsection-header">🚨 Clinical Alerts</div>', unsafe_allow_html=True)
        alert_cols = st.columns(min(len(alerts), 3))
        for idx, alert in enumerate(alerts[:3]):
            with alert_cols[idx % 3]:
                alert_colors = {
                    'danger': '#FADBD8',
                    'warning': '#FCF3CF',
                    'success': '#D5F4E6',
                    'info': '#EBF5FB'
                }
                bg_color = alert_colors.get(alert['type'], '#EBF5FB')
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <b>{alert['title']}</b><br>
                    <small>{alert['message']}</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Performance Indicators
    st.markdown('<div class="subsection-header">📊 Key Clinical Metrics</div>', unsafe_allow_html=True)
    
    total_seizures = int(daily_df['daily_total'].sum())
    total_severe = int(daily_df['daily_severe'].sum())
    avg_per_week = total_seizures / 52
    severity_rate = (total_severe / total_seizures * 100) if total_seizures > 0 else 0
    seizure_free_days = len(daily_df[daily_df['daily_total'] == 0])
    
    # Recent trends (last 4 weeks vs previous 4 weeks)
    recent_4weeks = weekly_df.tail(4)['total_seizures'].sum() if len(weekly_df) >= 4 else 0
    previous_4weeks = weekly_df.iloc[-8:-4]['total_seizures'].sum() if len(weekly_df) >= 8 else recent_4weeks
    trend_change = ((recent_4weeks - previous_4weeks) / previous_4weeks * 100) if previous_4weeks > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📈 Total Seizures",
            value=f"{total_seizures}",
            delta=f"{trend_change:+.1f}% (4 wks)" if trend_change != 0 else "No change",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="🔴 Severe Rate",
            value=f"{severity_rate:.1f}%",
            delta=f"{total_severe} severe"
        )
    
    with col3:
        st.metric(
            label="📅 Avg/Week",
            value=f"{avg_per_week:.1f}",
            delta=f"Last 4: {int(recent_4weeks)}"
        )
    
    with col4:
        st.metric(
            label="✅ Seizure-Free Days",
            value=f"{seizure_free_days}",
            delta=f"{(seizure_free_days/len(daily_df)*100):.1f}% of year"
        )
    
    with col5:
        # Current QoL if available
        recent_qol = assessment_df[assessment_df['QoL'].notna()].tail(1)
        if len(recent_qol) > 0:
            qol_value = recent_qol['QoL'].iloc[0]
            st.metric(
                label="🧠 Latest QoL",
                value=f"{qol_value:.0f}/100",
                delta="Recent assessment"
            )
        else:
            st.metric(label="🧠 QoL", value="N/A", delta="No data")
    
    st.markdown("---")
    
    # Main visualizations with granularity control
    st.markdown('<div class="subsection-header">📈 Seizure Activity Overview</div>', unsafe_allow_html=True)
    
    granularity = create_time_granularity_selector("main")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate temporal data if needed for 24-hour view
        if granularity == '24 Hours (Circadian)':
            with st.spinner("Analyzing temporal patterns..."):
                temporal_df = generate_temporal_seizure_data(daily_df)
        else:
            temporal_df = None
        
        # Create the visualization based on selected granularity
        fig = plot_timeline_by_view(daily_df, weekly_df, temporal_df, granularity, show_severe_only=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recent trends
        fig = go.Figure()
        last_12_weeks = weekly_df.tail(12)
        fig.add_trace(go.Scatter(
            x=last_12_weeks['week'],
            y=last_12_weeks['total_seizures'],
            mode='lines+markers',
            name='Weekly Total',
            line=dict(color='#3498DB', width=3),
            fill='tozeroy'
        ))
        fig.add_trace(go.Scatter(
            x=last_12_weeks['week'],
            y=last_12_weeks['severe_seizures'],
            mode='lines+markers',
            name='Weekly Severe',
            line=dict(color='#E74C3C', width=2),
            marker=dict(size=8, symbol='diamond')
        ))
        fig.update_layout(
            title='Recent 12-Week Trend',
            xaxis_title='Week Number',
            yaxis_title='Seizures',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Quick clinical summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">💊 Current Treatment Status</div>', unsafe_allow_html=True)
        # Show recent medication changes
        changes_df = detect_dose_changes(med_df, threshold=10)
        if len(changes_df) > 0:
            recent_changes = changes_df.tail(3)
            st.markdown("**Recent Dose Changes:**")
            for _, change in recent_changes.iterrows():
                st.markdown(f"- Day {int(change['day'])}: **{change['medication']}** {change['change_type']} ({change['change_amount']:+.0f} mg)")
        else:
            st.info("✅ Stable medication regimen - No recent dose changes")
    
    with col2:
        st.markdown('<div class="subsection-header">🧠 Patient-Reported Outcomes</div>', unsafe_allow_html=True)
        # Show recent assessments
        recent_assessments = assessment_df[assessment_df['QoL'].notna()].tail(1)
        if len(recent_assessments) > 0:
            assessment = recent_assessments.iloc[0]
            st.markdown(f"""
            **Latest Assessment (Day {int(assessment['day'])}):**
            - Quality of Life: **{assessment['QoL']:.0f}/100**
            - Anxiety: **{assessment['Anxiety']:.0f}/100**
            - Depression: **{assessment['Depression']:.0f}/100**
            - Behavioral: **{assessment['Behavioral']:.0f}/100**
            """)
        else:
            st.info("No recent assessment data available")

def render_seizure_analysis(daily_df, weekly_df):
    """Render Seizure Analysis section - comprehensive seizure patterns."""
    st.markdown('<div class="main-header">📈 Comprehensive Seizure Analysis</div>', unsafe_allow_html=True)
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Timeline & Trends",
        "🟩 Activity Heatmap",
        "🕐 Circadian Patterns",
        "📉 Statistical Analysis"
    ])
    
    with tab1:
        st.markdown('<div class="subsection-header">Seizure Timeline with Granularity Control</div>', unsafe_allow_html=True)
        
        st.markdown("""
        **View Options:**
        - **24 Hours (Circadian)**: Hourly breakdown showing time-of-day patterns
        - **Week (7 Days)**: Total seizures by day of week (Mon-Sun)
        - **Month (12 Months)**: Monthly distribution showing seasonal patterns
        - **365 Days**: Complete daily timeline for the entire year
        - **52 Weeks**: Weekly seizure burden across all 52 weeks
        """)
        
        granularity = create_time_granularity_selector("seizure_timeline")
        
        col1, col2 = st.columns([2, 1])
        with col2:
            show_severe_only = st.checkbox("Show severe seizures only", key="timeline_severe")
        
        # Generate temporal data if needed for 24-hour view
        if granularity == '24 Hours (Circadian)':
            with st.spinner("Analyzing temporal patterns..."):
                temporal_df = generate_temporal_seizure_data(daily_df)
        else:
            temporal_df = None
        
        # Create the visualization based on selected granularity
        fig = plot_timeline_by_view(daily_df, weekly_df, temporal_df, granularity, show_severe_only)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics based on granularity
        st.markdown("---")
        st.markdown("**Key Statistics:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        seizure_col = 'daily_severe' if show_severe_only else 'daily_total'
        seizure_label = "Severe" if show_severe_only else "Total"
        
        if granularity == '24 Hours (Circadian)':
            # Hourly statistics
            if temporal_df is not None and len(temporal_df) > 0:
                if show_severe_only:
                    temporal_df_filtered = temporal_df[temporal_df['is_severe'] == True]
                else:
                    temporal_df_filtered = temporal_df
                
                hourly_dist = temporal_df_filtered.groupby('hour').size()
                peak_hour = hourly_dist.idxmax() if len(hourly_dist) > 0 else 0
                
                with col1:
                    st.metric("Peak Hour", f"{peak_hour:02d}:00")
                with col2:
                    st.metric("Peak Hour Count", f"{int(hourly_dist.max()) if len(hourly_dist) > 0 else 0}")
                with col3:
                    morning = len(temporal_df_filtered[(temporal_df_filtered['hour'] >= 6) & (temporal_df_filtered['hour'] < 12)])
                    st.metric("Morning (6-12)", f"{morning}")
                with col4:
                    evening = len(temporal_df_filtered[(temporal_df_filtered['hour'] >= 18) & (temporal_df_filtered['hour'] < 24)])
                    st.metric("Evening (18-24)", f"{evening}")
        
        elif granularity == 'Week (7 Days)':
            # Day of week statistics
            dow_data = aggregate_by_granularity(daily_df, granularity)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            with col1:
                peak_idx = dow_data['severe' if show_severe_only else 'total'].idxmax()
                peak_day = dow_data.loc[peak_idx, 'day_name']
                st.metric("Peak Day", peak_day)
            with col2:
                peak_count = dow_data['severe' if show_severe_only else 'total'].max()
                st.metric("Peak Day Count", f"{int(peak_count)}")
            with col3:
                weekend = dow_data[dow_data['day_of_week'].isin([6, 7])]['severe' if show_severe_only else 'total'].sum()
                st.metric("Weekend Total", f"{int(weekend)}")
            with col4:
                weekday = dow_data[dow_data['day_of_week'].isin([1, 2, 3, 4, 5])]['severe' if show_severe_only else 'total'].sum()
                st.metric("Weekday Total", f"{int(weekday)}")
        
        elif granularity == 'Month (12 Months)':
            # Monthly statistics
            monthly_data = aggregate_by_granularity(daily_df, granularity)
            
            with col1:
                peak_idx = monthly_data['severe' if show_severe_only else 'total'].idxmax()
                peak_month = monthly_data.loc[peak_idx, 'month_name']
                st.metric("Peak Month", peak_month)
            with col2:
                peak_count = monthly_data['severe' if show_severe_only else 'total'].max()
                st.metric("Peak Month Count", f"{int(peak_count)}")
            with col3:
                lowest_idx = monthly_data['severe' if show_severe_only else 'total'].idxmin()
                lowest_month = monthly_data.loc[lowest_idx, 'month_name']
                st.metric("Lowest Month", lowest_month)
            with col4:
                avg = monthly_data['severe' if show_severe_only else 'total'].mean()
                st.metric("Avg per Month", f"{avg:.1f}")
        
        elif granularity == '365 Days':
            # Daily statistics
            with col1:
                median_daily = daily_df[daily_df[seizure_col] > 0][seizure_col].median()
                st.metric(f"Median {seizure_label} (Active Days)", f"{median_daily:.1f}")
            with col2:
                max_day = daily_df[seizure_col].max()
                st.metric(f"Max {seizure_label} in Single Day", f"{int(max_day)}")
            with col3:
                p90 = daily_df[seizure_col].quantile(0.90)
                st.metric("90th Percentile", f"{int(p90)}")
            with col4:
                consecutive_free = 0
                max_consecutive = 0
                for val in daily_df[seizure_col]:
                    if val == 0:
                        consecutive_free += 1
                        max_consecutive = max(max_consecutive, consecutive_free)
                    else:
                        consecutive_free = 0
                st.metric("Max Seizure-Free Streak", f"{max_consecutive} days")
        
        else:  # '52 Weeks'
            # Weekly statistics
            with col1:
                median_weekly = weekly_df['severe_seizures' if show_severe_only else 'total_seizures'].median()
                st.metric(f"Median {seizure_label}/Week", f"{median_weekly:.1f}")
            with col2:
                max_week = weekly_df['severe_seizures' if show_severe_only else 'total_seizures'].max()
                st.metric(f"Max {seizure_label} in Week", f"{int(max_week)}")
            with col3:
                p90 = weekly_df['severe_seizures' if show_severe_only else 'total_seizures'].quantile(0.90)
                st.metric("90th Percentile", f"{int(p90)}")
            with col4:
                zero_weeks = len(weekly_df[weekly_df['severe_seizures' if show_severe_only else 'total_seizures'] == 0])
                st.metric("Seizure-Free Weeks", f"{zero_weeks}")
        
        st.markdown("""
        <div class="insight-box">
            <b>🩺 Clinical Interpretation:</b> Analyze temporal patterns to identify periods of good and poor control. 
            Consider correlating observed peaks with medication changes, stress events, or other clinical factors.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="subsection-header">Activity Heatmap with Granularity Control</div>', unsafe_allow_html=True)
        
        st.markdown("""
        **Heatmap View Options:**
        - **24 Hours × Week**: Hourly patterns across days of the week
        - **Day × Week**: Daily activity across all 52 weeks (best for pattern recognition)
        - **Day × Month**: Daily activity within each month
        - **Week × Month**: Weekly patterns across months
        """)
        
        # Granularity selector for heatmap
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            heatmap_granularity = st.selectbox(
                "Heatmap View:",
                options=['Day x Week', '24 Hours x Week', 'Day x Month', 'Week x Month'],
                key='heatmap_granularity'
            )
        with col3:
            show_severe_heatmap = st.checkbox("Show severe seizures only", key="heatmap_severe")
        
        # Generate temporal data if needed
        if heatmap_granularity == '24 Hours x Week':
            with st.spinner("Analyzing temporal patterns..."):
                temporal_df = generate_temporal_seizure_data(daily_df)
        else:
            temporal_df = None
        
        # Create heatmap
        fig = plot_activity_heatmap_by_granularity(daily_df, temporal_df, heatmap_granularity, show_severe_heatmap)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Add the original TachyGrid as a bonus view
        with st.expander("📅 View TachyGrid (GitHub-Style Calendar)", expanded=False):
            st.markdown("**Year-at-a-Glance Calendar View:**")
            fig_tachygrid = plot_tachygrid(daily_df)
            st.plotly_chart(fig_tachygrid, use_container_width=True)
        
        st.markdown("---")
        
        # Statistics based on selected granularity
        st.markdown("**Key Insights:**")
        col1, col2, col3, col4 = st.columns(4)
        
        seizure_col = 'daily_severe' if show_severe_heatmap else 'daily_total'
        
        if heatmap_granularity == '24 Hours x Week':
            if temporal_df is not None and len(temporal_df) > 0:
                if show_severe_heatmap:
                    filtered_df = temporal_df[temporal_df['is_severe'] == True]
                else:
                    filtered_df = temporal_df
                
                hourly_dist = filtered_df.groupby('hour').size()
                dow_dist = filtered_df.groupby('day_of_week').size()
                
                with col1:
                    peak_hour = hourly_dist.idxmax() if len(hourly_dist) > 0 else 0
                    st.metric("Peak Hour", f"{peak_hour:02d}:00")
                with col2:
                    peak_day = dow_dist.idxmax() if len(dow_dist) > 0 else 1
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    st.metric("Peak Day", day_names[int(peak_day)-1])
                with col3:
                    morning = len(filtered_df[(filtered_df['hour'] >= 6) & (filtered_df['hour'] < 12)])
                    st.metric("Morning (6-12)", f"{morning}")
                with col4:
                    evening = len(filtered_df[(filtered_df['hour'] >= 18) & (filtered_df['hour'] < 24)])
                    st.metric("Evening (18-24)", f"{evening}")
        
        elif heatmap_granularity == 'Day x Week':
            # Week x Day statistics
            daily_total = daily_df[seizure_col].values
            with col1:
                max_day_idx = np.argmax(daily_total)
                st.metric("Highest Activity Day", f"Day {max_day_idx + 1}")
            with col2:
                zero_days = np.sum(daily_total == 0)
                st.metric("Seizure-Free Days", f"{zero_days}")
            with col3:
                median_nonzero = np.median(daily_total[daily_total > 0])
                st.metric("Median (Active)", f"{median_nonzero:.1f}")
            with col4:
                high_days = np.sum(daily_total >= np.percentile(daily_total, 75))
                st.metric("High Activity Days", f"{high_days}")
        
        else:
            # General statistics
            daily_total = daily_df[seizure_col].values
            with col1:
                st.metric("Total Seizures", f"{int(daily_total.sum())}")
            with col2:
                st.metric("Average/Day", f"{daily_total.mean():.1f}")
            with col3:
                st.metric("Max in Single Day", f"{int(daily_total.max())}")
            with col4:
                zero_days = np.sum(daily_total == 0)
                st.metric("Seizure-Free Days", f"{zero_days}")
        
        st.markdown("""
        <div class="insight-box">
            <b>🩺 Clinical Use:</b> Heatmaps reveal temporal patterns and clusters of high-seizure periods. 
            Use different granularities to identify hourly, daily, weekly, or monthly patterns. 
            Darker colors indicate higher seizure burden requiring clinical attention.
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="subsection-header">Circadian and Temporal Patterns</div>', unsafe_allow_html=True)
        
        # Generate temporal data
        with st.spinner("Analyzing temporal patterns..."):
            temporal_df = generate_temporal_seizure_data(daily_df)
        
        # Time view selector
        st.markdown("**Select Time View:**")
        time_view = st.selectbox(
            "View by:",
            options=['24 Hours', 'Week (Days)', 'Month', 'Year (Quarters)'],
            key='circadian_time_view'
        )
        
        st.markdown("---")
        
        # Side-by-side comparison: All Seizures vs Severe Seizures
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**All Seizures**")
            
            # Select appropriate chart based on time view
            if time_view == '24 Hours':
                fig_all = plot_circadian_polar(temporal_df, granularity='hour', show_severe_only=False)
            elif time_view == 'Week (Days)':
                fig_all = plot_day_of_week_polar(daily_df, show_severe_only=False)
            elif time_view == 'Month':
                fig_all = plot_monthly_polar(daily_df, show_severe_only=False)
            else:  # Year (Quarters)
                fig_all = plot_yearly_summary(daily_df, show_severe_only=False)
            
            st.plotly_chart(fig_all, use_container_width=True)
        
        with col2:
            st.markdown("**Severe Seizures Only**")
            
            # Same chart type but for severe seizures only
            if time_view == '24 Hours':
                fig_severe = plot_circadian_polar(temporal_df, granularity='hour', show_severe_only=True)
            elif time_view == 'Week (Days)':
                fig_severe = plot_day_of_week_polar(daily_df, show_severe_only=True)
            elif time_view == 'Month':
                fig_severe = plot_monthly_polar(daily_df, show_severe_only=True)
            else:  # Year (Quarters)
                fig_severe = plot_yearly_summary(daily_df, show_severe_only=True)
            
            st.plotly_chart(fig_severe, use_container_width=True)
        
        st.markdown("---")
        
        # Statistics and breakdown based on selected view
        if time_view == '24 Hours':
            st.markdown("**📊 Hourly Pattern Statistics:**")
            
            # Time period breakdown with all vs severe comparison
            night_count = len(temporal_df[(temporal_df['hour'] >= 0) & (temporal_df['hour'] < 6)])
            morning_count = len(temporal_df[(temporal_df['hour'] >= 6) & (temporal_df['hour'] < 12)])
            afternoon_count = len(temporal_df[(temporal_df['hour'] >= 12) & (temporal_df['hour'] < 18)])
            evening_count = len(temporal_df[(temporal_df['hour'] >= 18) & (temporal_df['hour'] < 24)])
            
            night_severe = len(temporal_df[(temporal_df['hour'] >= 0) & (temporal_df['hour'] < 6) & (temporal_df['is_severe'] == True)])
            morning_severe = len(temporal_df[(temporal_df['hour'] >= 6) & (temporal_df['hour'] < 12) & (temporal_df['is_severe'] == True)])
            afternoon_severe = len(temporal_df[(temporal_df['hour'] >= 12) & (temporal_df['hour'] < 18) & (temporal_df['is_severe'] == True)])
            evening_severe = len(temporal_df[(temporal_df['hour'] >= 18) & (temporal_df['hour'] < 24) & (temporal_df['is_severe'] == True)])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Night (12am-6am)", f"{night_count}", f"{night_severe} severe")
            with col2:
                st.metric("Morning (6am-12pm)", f"{morning_count}", f"{morning_severe} severe")
            with col3:
                st.metric("Afternoon (12pm-6pm)", f"{afternoon_count}", f"{afternoon_severe} severe")
            with col4:
                st.metric("Evening (6pm-12am)", f"{evening_count}", f"{evening_severe} severe")
            
            # Bar chart comparison
            period_data = pd.DataFrame({
                'Period': ['Night\n(12am-6am)', 'Morning\n(6am-12pm)', 'Afternoon\n(12pm-6pm)', 'Evening\n(6pm-12am)'],
                'All Seizures': [night_count, morning_count, afternoon_count, evening_count],
                'Severe Seizures': [night_severe, morning_severe, afternoon_severe, evening_severe]
            })
            
            fig_period = go.Figure()
            fig_period.add_trace(go.Bar(
                x=period_data['Period'],
                y=period_data['All Seizures'],
                name='All Seizures',
                marker_color='#3498DB',
                text=period_data['All Seizures'],
                textposition='outside'
            ))
            fig_period.add_trace(go.Bar(
                x=period_data['Period'],
                y=period_data['Severe Seizures'],
                name='Severe Seizures',
                marker_color='#E74C3C',
                text=period_data['Severe Seizures'],
                textposition='outside'
            ))
            fig_period.update_layout(
                title='All vs Severe Seizures by Time Period',
                yaxis_title='Number of Seizures',
                template='plotly_white',
                height=400,
                showlegend=True,
                barmode='group'
            )
            st.plotly_chart(fig_period, use_container_width=True)
        
        elif time_view == 'Week (Days)':
            st.markdown("**📊 Weekly Pattern Statistics:**")
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            all_by_day = daily_df.groupby('day_of_week')['daily_total'].sum()
            severe_by_day = daily_df.groupby('day_of_week')['daily_severe'].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                peak_day = all_by_day.idxmax() if len(all_by_day) > 0 else 1
                st.metric("Peak Day (All)", day_names[int(peak_day)-1], f"{int(all_by_day.max())} seizures")
            with col2:
                peak_severe = severe_by_day.idxmax() if len(severe_by_day) > 0 and severe_by_day.max() > 0 else 1
                st.metric("Peak Day (Severe)", day_names[int(peak_severe)-1], f"{int(severe_by_day.max())} seizures")
            with col3:
                weekend = all_by_day[all_by_day.index.isin([6, 7])].sum() if len(all_by_day) > 0 else 0
                st.metric("Weekend Total", f"{int(weekend)}", "Sat + Sun")
            with col4:
                weekday = all_by_day[all_by_day.index.isin([1, 2, 3, 4, 5])].sum() if len(all_by_day) > 0 else 0
                st.metric("Weekday Total", f"{int(weekday)}", "Mon-Fri")
        
        elif time_view == 'Month':
            st.markdown("**📊 Monthly Pattern Statistics:**")
            
            month_names = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
            daily_df_copy = daily_df.copy()
            daily_df_copy['month'] = ((daily_df_copy['day'] - 1) // 30) + 1
            daily_df_copy['month'] = daily_df_copy['month'].clip(upper=12)
            
            all_by_month = daily_df_copy.groupby('month')['daily_total'].sum()
            severe_by_month = daily_df_copy.groupby('month')['daily_severe'].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                peak_month = all_by_month.idxmax() if len(all_by_month) > 0 else 1
                st.metric("Peak Month (All)", month_names[int(peak_month)-1], f"{int(all_by_month.max())} seizures")
            with col2:
                peak_severe_month = severe_by_month.idxmax() if len(severe_by_month) > 0 and severe_by_month.max() > 0 else 1
                st.metric("Peak Month (Severe)", month_names[int(peak_severe_month)-1], f"{int(severe_by_month.max())} seizures")
            with col3:
                lowest = all_by_month.idxmin() if len(all_by_month) > 0 else 1
                st.metric("Best Month", month_names[int(lowest)-1], f"{int(all_by_month.min())} seizures")
            with col4:
                avg = all_by_month.mean() if len(all_by_month) > 0 else 0
                st.metric("Avg per Month", f"{avg:.1f}", "Mean")
        
        else:  # Year (Quarters)
            st.markdown("**📊 Quarterly Pattern Statistics:**")
            
            daily_df_copy = daily_df.copy()
            daily_df_copy['quarter'] = ((daily_df_copy['day'] - 1) // 91) + 1
            daily_df_copy['quarter'] = daily_df_copy['quarter'].clip(upper=4)
            
            all_by_quarter = daily_df_copy.groupby('quarter')['daily_total'].sum()
            severe_by_quarter = daily_df_copy.groupby('quarter')['daily_severe'].sum()
            quarter_names = ['Q1 (Sep-Nov)', 'Q2 (Dec-Feb)', 'Q3 (Mar-May)', 'Q4 (Jun-Aug)']
            
            col1, col2, col3, col4 = st.columns(4)
            for i, col in enumerate([col1, col2, col3, col4], 1):
                with col:
                    q_all = int(all_by_quarter.get(i, 0)) if len(all_by_quarter) > 0 else 0
                    q_severe = int(severe_by_quarter.get(i, 0)) if len(severe_by_quarter) > 0 else 0
                    st.metric(quarter_names[i-1], f"{q_all}", f"{q_severe} severe")
        
        st.markdown("""
        <div class="insight-box">
            <b>🩺 Clinical Implications:</b>
            <ul>
            <li><b>24 Hours:</b> Circadian patterns guide medication timing - morning peaks may benefit from bedtime dosing</li>
            <li><b>Week (Days):</b> Day-of-week patterns may relate to routines, stress, sleep schedules, or medication adherence</li>
            <li><b>Month:</b> Seasonal variations could indicate environmental triggers or physiological cycles</li>
            <li><b>Year (Quarters):</b> Long-term trends help assess overall treatment trajectory</li>
            </ul>
            <b>Comparing All vs Severe:</b> Different patterns between all seizures and severe seizures may indicate 
            specific vulnerability periods or triggers requiring targeted intervention.
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="subsection-header">Statistical Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Descriptive Statistics:**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'IQR (25th-75th)'],
                'Total Seizures': [
                    f"{daily_df['daily_total'].mean():.2f}",
                    f"{daily_df['daily_total'].median():.2f}",
                    f"{daily_df['daily_total'].std():.2f}",
                    f"{daily_df['daily_total'].min():.0f}",
                    f"{daily_df['daily_total'].max():.0f}",
                    f"{daily_df['daily_total'].quantile(0.25):.0f} - {daily_df['daily_total'].quantile(0.75):.0f}"
                ],
                'Severe Seizures': [
                    f"{daily_df['daily_severe'].mean():.2f}",
                    f"{daily_df['daily_severe'].median():.2f}",
                    f"{daily_df['daily_severe'].std():.2f}",
                    f"{daily_df['daily_severe'].min():.0f}",
                    f"{daily_df['daily_severe'].max():.0f}",
                    f"{daily_df['daily_severe'].quantile(0.25):.0f} - {daily_df['daily_severe'].quantile(0.75):.0f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Distribution Analysis:**")
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=daily_df['daily_total'],
                nbinsx=20,
                name='Total Seizures',
                marker_color='#3498DB',
                opacity=0.7
            ))
            fig_hist.update_layout(
                title='Seizure Frequency Distribution',
                xaxis_title='Seizures per Day',
                yaxis_title='Number of Days',
                template='plotly_white',
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def render_treatment_management(med_df, weekly_df, daily_df):
    """Render Treatment Management section - medication analysis."""
    st.markdown('<div class="main-header">💊 Treatment Management & Medication Analysis</div>', unsafe_allow_html=True)
    
    
    # Tabs for treatment management
    tab1, tab2, tab3 = st.tabs([
        "💊 Current Regimen & Changes",
        "📊 Dose-Response Analysis",
        "🎯 Treatment Efficacy"
    ])
    
    with tab1:
        st.markdown('<div class="subsection-header">Current Medication Regimen</div>', unsafe_allow_html=True)
        
        # Current doses (most recent)
        medications = ['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']
        current_doses = {}
        for med in medications:
            current_doses[med] = med_df[med].iloc[-1]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Current Doses (Most Recent):**")
            for med, dose in current_doses.items():
                st.metric(label=med, value=f"{dose:.1f} mg")
        
        with col2:
            # Medication trajectories with granularity control
            time_range = st.selectbox(
                "Display period:",
                options=['Last 30 Days', 'Last 90 Days', 'Last 6 Months', 'Full Year'],
                key="med_time_range"
            )
            
            days_to_show = {
                'Last 30 Days': 30,
                'Last 90 Days': 90,
                'Last 6 Months': 182,
                'Full Year': 364
            }[time_range]
            
            med_df_subset = med_df.tail(days_to_show)
            fig = plot_medications(med_df_subset)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Dose change detection and timeline
        st.markdown('<div class="subsection-header">Dose Change Timeline</div>', unsafe_allow_html=True)
        
        changes_df = detect_dose_changes(med_df, threshold=10)
        
        if len(changes_df) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = plot_dose_changes(med_df, changes_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Significant Dose Changes:**")
                st.dataframe(
                    changes_df[['day', 'medication', 'change_type', 'change_amount']].sort_values('day', ascending=False).head(10).style.format({
                        'day': '{:.0f}',
                        'change_amount': '{:+.1f} mg'
                    }),
                    height=350,
                    hide_index=True
                )
        else:
            st.info("✅ Stable regimen - No significant dose changes detected (threshold: 10mg)")
        
        st.markdown("""
        <div class="insight-box">
            <b>🩺 Clinical Note:</b> Track dose adjustments and correlate with seizure control changes. 
            Consider medication half-life when assessing time to effect.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Medication Load Stacked Bar Chart
        st.markdown('<div class="subsection-header">Medication Load Over Time</div>', unsafe_allow_html=True)
        
        st.markdown("""
        **Stacked Bar Chart:** Shows total medication burden and individual drug contributions over time.
        Higher bars indicate increased medication load, useful for identifying polytherapy burden.
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            stack_granularity = st.selectbox(
                "Granularity:",
                options=['Weekly', 'Monthly', 'Daily'],
                key='medication_stack_granularity'
            )
        
        fig_stacked = plot_medication_stacked_bar(med_df, granularity=stack_granularity)
        st.plotly_chart(fig_stacked, use_container_width=True)
        
        # Calculate total medication load statistics
        total_load = med_df[medications].sum(axis=1)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Total Load", f"{total_load.iloc[-1]:.1f} mg")
        with col2:
            st.metric("Average Load", f"{total_load.mean():.1f} mg")
        with col3:
            st.metric("Max Load", f"{total_load.max():.1f} mg")
        with col4:
            change = total_load.iloc[-1] - total_load.iloc[0]
            st.metric("Load Change", f"{change:+.1f} mg", delta=f"{'↑ Increase' if change > 0 else '↓ Decrease'}")
        
        st.markdown("""
        <div class="insight-box">
            <b>🩺 Polytherapy Assessment:</b> Monitor total medication load for potential side effect burden. 
            Higher total doses may increase adverse effects while providing better seizure control. 
            Consider simplifying regimen if seizure control is good and load is high.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="subsection-header">Dose-Response Relationship Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_med = st.selectbox("Select medication:", medications, key="dose_response_med")
        
        with col2:
            show_correlation = st.checkbox("Show correlation statistics", value=True)
        
        fig = plot_dose_response(weekly_df, selected_med)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display correlation
            plot_data = weekly_df[[selected_med, 'total_seizures']].dropna()
            if len(plot_data) > 1 and show_correlation:
                corr, p_value = pearsonr(plot_data[selected_med], plot_data['total_seizures'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Correlation Coefficient", f"{corr:.3f}")
                with col2:
                    st.metric("P-value", f"{p_value:.4f}")
                with col3:
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    st.metric("Statistical Significance", significance)
                
                interpretation = ""
                if corr < -0.3:
                    interpretation = "🟢 **Negative correlation**: Higher doses associated with fewer seizures (suggests efficacy)"
                elif corr > 0.3:
                    interpretation = "🔴 **Positive correlation**: Higher doses associated with more seizures (may indicate reactive dose increases)"
                else:
                    interpretation = "⚪ **Weak correlation**: No clear dose-response relationship observed"
                
                st.markdown(f"""
                <div class="insight-box">
                    <b>🩺 Interpretation:</b> {interpretation}<br><br>
                    <b>Clinical Context:</b> Correlations should be interpreted carefully. Positive correlations 
                    often reflect dose increases in response to worsening seizures rather than medication inefficacy.
                    Consider lag time between dose changes and clinical effects.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"Insufficient data for {selected_med} dose-response analysis")
        
        # Multiple medication comparison
        st.markdown("---")
        st.markdown("**Compare All Medications:**")
        
        corr_data = []
        for med in medications:
            plot_data = weekly_df[[med, 'total_seizures']].dropna()
            if len(plot_data) > 1:
                corr, p_value = pearsonr(plot_data[med], plot_data['total_seizures'])
                corr_data.append({
                    'Medication': med,
                    'Correlation': f"{corr:.3f}",
                    'P-value': f"{p_value:.4f}",
                    'Significance': '✓' if p_value < 0.05 else '✗'
                })
        
        if corr_data:
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown('<div class="subsection-header">Treatment Efficacy Assessment</div>', unsafe_allow_html=True)
        
        # Compare seizure control across treatment periods
        st.markdown("**Seizure Control by Treatment Phase:**")
        
        # Divide into quarters
        total_days = len(daily_df)
        quarter_size = total_days // 4
        
        quarters = []
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else total_days
            quarter_data = daily_df.iloc[start_idx:end_idx]
            quarters.append({
                'Period': f'Q{i+1} (Days {start_idx+1}-{end_idx})',
                'Total Seizures': int(quarter_data['daily_total'].sum()),
                'Severe Seizures': int(quarter_data['daily_severe'].sum()),
                'Avg/Week': quarter_data['daily_total'].sum() / (len(quarter_data) / 7),
                'Seizure-Free Days': int((quarter_data['daily_total'] == 0).sum())
            })
        
        quarters_df = pd.DataFrame(quarters)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(quarters_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Trend chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=quarters_df['Period'],
                y=quarters_df['Total Seizures'],
                mode='lines+markers',
                name='Total Seizures',
                line=dict(color='#3498DB', width=3),
                marker=dict(size=12)
            ))
            fig.update_layout(
                title='Quarterly Seizure Trend',
                yaxis_title='Total Seizures',
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Treatment response summary
        q1_seizures = quarters[0]['Total Seizures']
        q4_seizures = quarters[3]['Total Seizures']
        percent_change = ((q4_seizures - q1_seizures) / q1_seizures * 100) if q1_seizures > 0 else 0
        
        if percent_change < -20:
            response = "🟢 **Excellent Response**: Significant improvement in seizure control"
        elif percent_change < -10:
            response = "🟡 **Good Response**: Moderate improvement observed"
        elif percent_change < 10:
            response = "⚪ **Stable**: No significant change in seizure burden"
        else:
            response = "🔴 **Poor Response**: Seizure burden has increased - consider treatment adjustment"
        
        st.markdown(f"""
        <div class="insight-box">
            <b>🩺 Treatment Response Assessment:</b><br>
            {response}<br><br>
            <b>Change from Q1 to Q4:</b> {percent_change:+.1f}%<br>
            <b>Recommendation:</b> {"Continue current regimen with monitoring" if percent_change < 0 else "Consider medication adjustment or alternative therapy"}
        </div>
        """, unsafe_allow_html=True)
    
def render_patient_outcomes(assessment_df, weekly_df, daily_df):
    """Render Patient Outcomes section - QoL and mental health focus."""
    st.markdown('<div class="main-header">🧠 Patient-Reported Outcomes & Quality of Life</div>', unsafe_allow_html=True)
    
    
    # Tabs for patient outcomes
    tab1, tab2, tab3 = st.tabs([
        "📊 Assessment Overview",
        "🔗 QoL vs Seizure Control",
        "📈 Trend Analysis"
    ])
    
    with tab1:
        st.markdown('<div class="subsection-header">Current Patient-Reported Status</div>', unsafe_allow_html=True)
        
        # Most recent assessment
        assessments = ['QoL', 'Anxiety', 'Depression', 'Behavioral']
        recent_assessment = assessment_df[assessment_df['QoL'].notna()].tail(1)
        
        if len(recent_assessment) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            assessment_data = recent_assessment.iloc[0]
            day = int(assessment_data['day'])
            
            with col1:
                qol_val = assessment_data['QoL']
                qol_status = "🟢 Good" if qol_val >= 70 else "🟡 Fair" if qol_val >= 50 else "🔴 Poor"
                st.metric(
                    label="Quality of Life",
                    value=f"{qol_val:.0f}/100",
                    delta=qol_status
                )
            
            with col2:
                anx_val = assessment_data['Anxiety']
                anx_status = "🟢 Low" if anx_val <= 30 else "🟡 Moderate" if anx_val <= 60 else "🔴 High"
                st.metric(
                    label="Anxiety Level",
                    value=f"{anx_val:.0f}/100",
                    delta=anx_status
                )
            
            with col3:
                dep_val = assessment_data['Depression']
                dep_status = "🟢 Low" if dep_val <= 30 else "🟡 Moderate" if dep_val <= 60 else "🔴 High"
                st.metric(
                    label="Depression Level",
                    value=f"{dep_val:.0f}/100",
                    delta=dep_status
                )
            
            with col4:
                beh_val = assessment_data['Behavioral']
                beh_status = "🟢 Good" if beh_val >= 60 else "🟡 Fair" if beh_val >= 40 else "🔴 Poor"
                st.metric(
                    label="Behavioral Score",
                    value=f"{beh_val:.0f}/100",
                    delta=beh_status
                )
            
            st.markdown(f"<small>*Most recent assessment from Day {day}</small>", unsafe_allow_html=True)
        else:
            st.warning("No recent assessment data available")
        
        st.markdown("---")
        
        # Full timeline
        st.markdown('<div class="subsection-header">Assessment Timeline</div>', unsafe_allow_html=True)
        
        fig = plot_assessments(assessment_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics table
        st.markdown("**Statistical Summary:**")
        
        summary_data = []
        for assessment in assessments:
            data = assessment_df[assessment].dropna()
            if len(data) > 0:
                summary_data.append({
                    'Measure': assessment,
                    'Mean': f"{data.mean():.1f}",
                    'Median': f"{data.median():.1f}",
                    'Min': f"{data.min():.1f}",
                    'Max': f"{data.max():.1f}",
                    'Std Dev': f"{data.std():.1f}",
                    'Assessments': len(data)
                })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>🩺 Clinical Note:</b> Patient-reported outcomes are essential for holistic epilepsy care. 
        These measures capture aspects of health that seizure frequency alone cannot reflect. 
        Monitor trends and consider mental health referrals if anxiety/depression scores are persistently elevated.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="subsection-header">Quality of Life vs Seizure Control</div>', unsafe_allow_html=True)
        
        fig = plot_seizure_qol_overlay(weekly_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        qol_data = weekly_df[['total_seizures', 'QoL']].dropna()
        if len(qol_data) > 1:
            corr, p_value = pearsonr(qol_data['total_seizures'], qol_data['QoL'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correlation", f"{corr:.3f}")
            with col2:
                st.metric("P-value", f"{p_value:.4f}")
            with col3:
                sig = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric("Significance", sig)
            
            interpretation = ""
            if corr < -0.3 and p_value < 0.05:
                interpretation = "🟢 **Strong negative correlation**: More seizures significantly associated with lower quality of life"
            elif corr < -0.2:
                interpretation = "🟡 **Moderate correlation**: Seizures appear to impact quality of life"
            else:
                interpretation = "⚪ **Weak correlation**: QoL may be influenced by factors beyond seizure frequency"
            
            st.markdown(f"""
        <div class="insight-box">
            <b>🩺 Interpretation:</b> {interpretation}<br><br>
            <b>Clinical Implications:</b> If seizure control doesn't correlate strongly with QoL, 
            consider medication side effects, psychosocial factors, or comorbidities affecting wellbeing.
        </div>
        """, unsafe_allow_html=True)
        
        # Mental health correlation
        st.markdown("---")
        st.markdown("**Mental Health Measures vs Seizure Burden:**")
        
        mental_corr = []
        for measure in ['Anxiety', 'Depression']:
            data = weekly_df[['total_seizures', measure]].dropna()
            if len(data) > 1:
                corr, p_val = pearsonr(data['total_seizures'], data[measure])
                mental_corr.append({
                    'Measure': measure,
                    'Correlation with Seizures': f"{corr:.3f}",
                    'P-value': f"{p_val:.4f}",
                    'Interpretation': 'Positive' if corr > 0.2 else 'Negative' if corr < -0.2 else 'Weak'
                })
        
        if mental_corr:
            st.dataframe(pd.DataFrame(mental_corr), use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown('<div class="subsection-header">Longitudinal Trend Analysis</div>', unsafe_allow_html=True)
        
        # Calculate trends over time
        st.markdown("**Assess whether patient-reported outcomes are improving or declining:**")
        
        for measure in ['QoL', 'Anxiety', 'Depression', 'Behavioral']:
            data = assessment_df[['day', measure]].dropna()
            
            if len(data) >= 3:
                # Fit linear trend
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(data['day'], data[measure])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Plot with trendline
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data['day'],
                        y=data[measure],
                        mode='markers+lines',
                        name=measure,
                        marker=dict(size=10)
                    ))
                    
                    # Add trendline
                    trend_line = slope * data['day'] + intercept
                    fig.add_trace(go.Scatter(
                        x=data['day'],
                        y=trend_line,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'{measure} Trend Over Time',
                        xaxis_title='Day',
                        yaxis_title=measure,
                        template='plotly_white',
                        height=250,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    trend_direction = ""
                    if abs(slope) < 0.01:
                        trend_direction = "⚪ **Stable**: No significant change"
                    elif slope > 0:
                        if measure in ['QoL', 'Behavioral']:
                            trend_direction = "🟢 **Improving**: Positive trend"
                        else:
                            trend_direction = "🔴 **Worsening**: Increasing symptoms"
                    else:
                        if measure in ['QoL', 'Behavioral']:
                            trend_direction = "🔴 **Declining**: Negative trend"
                        else:
                            trend_direction = "🟢 **Improving**: Decreasing symptoms"
                    
                    st.markdown(f"""
                    **{measure} Trend:**
                    
                    {trend_direction}
                    
                    Slope: {slope:.3f}/day
                    
                    R²: {r_value**2:.3f}
                    """)
        
        st.markdown("""
        <div class="insight-box">
        <b>🩺 Clinical Guidance:</b> Monitor trends in patient-reported outcomes. Declining QoL or 
        increasing anxiety/depression warrant clinical attention even if seizure control is stable. 
        Consider referral to mental health services if indicated.
        </div>
        """, unsafe_allow_html=True)

def render_clinical_report(daily_df, weekly_df, med_df, assessment_df):
    """Render Clinical Report section - comprehensive analysis and recommendations."""
    st.markdown('<div class="main-header">📋 Comprehensive Clinical Report</div>', unsafe_allow_html=True)
    
    # Clinical Report Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Executive Summary",
        "🔬 Correlation Analysis",
        "📋 Clinical Recommendations"
    ])
    
    with tab1:
        st.markdown('<div class="subsection-header">Executive Summary - Patient at a Glance</div>', unsafe_allow_html=True)
        
        # Key metrics summary
        total_seizures = int(daily_df['daily_total'].sum())
        total_severe = int(daily_df['daily_severe'].sum())
        avg_per_week = total_seizures / 52
        severity_rate = (total_severe / total_seizures * 100) if total_seizures > 0 else 0
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**52-Week Summary:**")
            st.markdown(f"""
            - **Monitoring Period:** {len(daily_df)} days
            - **Total Seizures:** {total_seizures} ({avg_per_week:.1f} per week average)
            - **Severe Seizures:** {total_severe} ({severity_rate:.1f}% severity rate)
            - **Seizure-Free Days:** {len(daily_df[daily_df['daily_total'] == 0])} ({len(daily_df[daily_df['daily_total'] == 0])/len(daily_df)*100:.1f}%)
            - **Medication Regimen:** 5-drug polytherapy
            """)
            
            # Treatment response
            recent_4wks = weekly_df.tail(4)['total_seizures'].sum() if len(weekly_df) >= 4 else 0
            previous_4wks = weekly_df.iloc[-8:-4]['total_seizures'].sum() if len(weekly_df) >= 8 else recent_4wks
            trend = "improving" if recent_4wks < previous_4wks else "worsening" if recent_4wks > previous_4wks else "stable"
            
            st.markdown(f"- **Recent Trend (4 weeks):** {trend.capitalize()} ({int(recent_4wks)} vs {int(previous_4wks)} seizures)")
        
        with col2:
            # Quick status indicators
            control_status = "🟢 Good" if avg_per_week < 3 else "🟡 Moderate" if avg_per_week < 6 else "🔴 Poor"
            st.metric("Seizure Control", control_status)
            
            recent_qol = assessment_df[assessment_df['QoL'].notna()].tail(1)
            if len(recent_qol) > 0:
                qol_val = recent_qol['QoL'].iloc[0]
                qol_status = "🟢 Good" if qol_val >= 70 else "🟡 Fair" if qol_val >= 50 else "🔴 Poor"
                st.metric("Quality of Life", f"{qol_val:.0f}/100")
            
        st.markdown("---")
        
        # Integrated multi-panel view
        st.markdown('<div class="subsection-header">Integrated Clinical View</div>', unsafe_allow_html=True)
        fig = plot_integrated_dashboard(daily_df, weekly_df, med_df, assessment_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>🩺 Holistic Assessment:</b> This integrated view combines seizure activity, medication management, 
        and patient-reported outcomes to provide a comprehensive clinical picture across the monitoring period.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="subsection-header">Correlation Matrix - Multi-Factor Analysis</div>', unsafe_allow_html=True)
        
        fig = plot_correlation_heatmap(weekly_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Key Correlations to Review:**")
        
        # Calculate key correlations
        key_correlations = []
        
        medications = ['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']
        for med in medications:
            data = weekly_df[[med, 'total_seizures']].dropna()
            if len(data) > 1:
                corr, p_val = pearsonr(data[med], data['total_seizures'])
                if abs(corr) > 0.2:  # Only show meaningful correlations
                    key_correlations.append({
                        'Factor 1': med,
                        'Factor 2': 'Total Seizures',
                        'Correlation': f"{corr:.3f}",
                        'Strength': 'Strong' if abs(corr) > 0.5 else 'Moderate',
                        'P-value': f"{p_val:.4f}"
                    })
        
        # QoL correlations
        for measure in ['QoL', 'Anxiety', 'Depression']:
            data = weekly_df[[measure, 'total_seizures']].dropna()
            if len(data) > 1:
                corr, p_val = pearsonr(data[measure], data['total_seizures'])
                if abs(corr) > 0.2:
                    key_correlations.append({
                        'Factor 1': measure,
                        'Factor 2': 'Total Seizures',
                        'Correlation': f"{corr:.3f}",
                        'Strength': 'Strong' if abs(corr) > 0.5 else 'Moderate',
                        'P-value': f"{p_val:.4f}"
                    })
        
        if key_correlations:
            st.dataframe(pd.DataFrame(key_correlations), use_container_width=True, hide_index=True)
        else:
            st.info("No strong correlations detected in the data")
        
        st.markdown("""
        <div class="insight-box">
        <b>🩺 Interpretation Guidelines:</b>
        <ul>
        <li><b>Medication-Seizure Correlations:</b> Negative correlations suggest efficacy. Positive correlations often reflect reactive dose increases.</li>
        <li><b>QoL-Seizure Correlations:</b> Negative correlations demonstrate seizure impact on quality of life.</li>
        <li><b>Mental Health-Seizure Correlations:</b> Strong positive correlations may indicate seizure-related distress or vice versa.</li>
        </ul>
        Remember: Correlation ≠ Causation. Consider temporal relationships and confounding factors.
        </div>
        """, unsafe_allow_html=True)
        
    with tab3:
        st.markdown('<div class="subsection-header">Clinical Recommendations & Action Items</div>', unsafe_allow_html=True)
        
        # Generate recommendations based on data
        recommendations = []
        
        # Seizure control assessment
        if avg_per_week > 7:
            recommendations.append({
                'Priority': '🔴 High',
                'Area': 'Seizure Control',
                'Finding': f'Average of {avg_per_week:.1f} seizures/week indicates suboptimal control',
                'Recommendation': 'Consider medication adjustment, dose optimization, or alternative therapy'
            })
        elif avg_per_week < 2:
            recommendations.append({
                'Priority': '🟢 Good',
                'Area': 'Seizure Control',
                'Finding': f'Excellent control with {avg_per_week:.1f} seizures/week',
                'Recommendation': 'Continue current regimen with ongoing monitoring'
            })
        
        # Severity assessment
        if severity_rate > 25:
            recommendations.append({
                'Priority': '🟡 Medium',
                'Area': 'Seizure Severity',
                'Finding': f'High severity rate ({severity_rate:.1f}%) - many seizures are severe',
                'Recommendation': 'Review rescue medication protocols and patient safety measures'
            })
        
        # Mental health
        recent_anxiety = assessment_df[assessment_df['Anxiety'].notna()].tail(1)
        if len(recent_anxiety) > 0 and recent_anxiety['Anxiety'].iloc[0] > 60:
            recommendations.append({
                'Priority': '🟡 Medium',
                'Area': 'Mental Health',
                'Finding': 'Elevated anxiety scores detected',
                'Recommendation': 'Consider referral to mental health services or counseling'
            })
        
        # QoL assessment
        recent_qol = assessment_df[assessment_df['QoL'].notna()].tail(1)
        if len(recent_qol) > 0 and recent_qol['QoL'].iloc[0] < 50:
            recommendations.append({
                'Priority': '🟡 Medium',
                'Area': 'Quality of Life',
                'Finding': 'Low quality of life score reported',
                'Recommendation': 'Comprehensive review of seizure impact, medication side effects, and psychosocial factors'
            })
        
        # Medication stability
        changes_df = detect_dose_changes(med_df, threshold=20)
        recent_changes = changes_df[changes_df['day'] > len(daily_df) - 30] if len(changes_df) > 0 else pd.DataFrame()
        if len(recent_changes) > 3:
            recommendations.append({
                'Priority': '🟡 Medium',
                'Area': 'Medication Stability',
                'Finding': 'Multiple recent dose changes detected',
                'Recommendation': 'Allow adequate time for medication adjustments to take effect before further changes'
            })
        
        # Positive findings
        seizure_free_days = len(daily_df[daily_df['daily_total'] == 0])
        if seizure_free_days / len(daily_df) > 0.7:
            recommendations.append({
                'Priority': '🟢 Good',
                'Area': 'Treatment Success',
                'Finding': f'{seizure_free_days} seizure-free days ({seizure_free_days/len(daily_df)*100:.1f}% of year)',
                'Recommendation': 'Maintain current successful treatment approach'
            })
        
        if len(recommendations) > 0:
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        else:
            st.info("Clinical data reviewed - no specific action items identified")
        
        st.markdown("---")
        
        st.markdown("**Follow-up Plan:**")
        st.markdown("""
        <div class="insight-box">
        <b>🩺 Suggested Next Steps:</b>
        <ol>
        <li><b>Review recommendations above</b> and prioritize based on clinical judgment</li>
        <li><b>Discuss findings with patient</b> including seizure patterns and quality of life impact</li>
        <li><b>Consider medication adjustments</b> if indicated by seizure control and side effects</li>
        <li><b>Schedule follow-up</b> in 4-8 weeks to reassess response to any interventions</li>
        <li><b>Continue monitoring</b> seizure frequency, medication adherence, and patient-reported outcomes</li>
        <li><b>Refer to specialists</b> if mental health concerns or refractory seizures persist</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

def render_section_tachygrid(daily_df):
    """Render TachyGrid Heatmap section."""
    st.markdown('<div class="main-header">🟩 TachyGrid — Seizure Heatmap</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    The TachyGrid provides a GitHub-style contribution map visualization of seizure activity 
    across the entire year. Each square represents a single day, with color intensity indicating 
    seizure burden. This format makes it easy to spot patterns, clusters of high activity, and 
    periods of better control at a glance. <b>Hover over any square to see details!</b>
    </div>
    """, unsafe_allow_html=True)
    
    # Create the TachyGrid
    fig = plot_tachygrid(daily_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add descriptive caption
    st.markdown("""
    <div style="text-align: center; color: #5D6D7E; font-size: 0.95rem; margin-top: 1rem; font-style: italic;">
    This grid visualizes daily seizure burden across the full year, using a GitHub-style contribution map format.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Add some statistics
    st.markdown('<div class="subsection-header">TachyGrid Insights</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    daily_total = daily_df['daily_total'].values
    
    with col1:
        max_day = np.argmax(daily_total)
        st.metric(
            label="🔴 Highest Day",
            value=f"Day {max_day + 1}",
            delta=f"{daily_total[max_day]:.0f} seizures"
        )
    
    with col2:
        zero_days = np.sum(daily_total == 0)
        st.metric(
            label="🟢 Seizure-Free Days",
            value=f"{zero_days}",
            delta=f"{(zero_days/364*100):.1f}% of year"
        )
    
    with col3:
        median_count = np.median(daily_total[daily_total > 0])
        st.metric(
            label="📊 Median (Active Days)",
            value=f"{median_count:.1f}",
            delta="When seizures occur"
        )
    
    with col4:
        high_days = np.sum(daily_total >= np.percentile(daily_total, 75))
        st.metric(
            label="⚠️ High Activity Days",
            value=f"{high_days}",
            delta=f"≥75th percentile"
        )
    
    st.markdown("""
    <div class="insight-box">
    <b>How to Read the TachyGrid:</b>
    <ul>
    <li><b>Light squares:</b> Few or no seizures (better control)</li>
    <li><b>Dark green squares:</b> Higher seizure counts (increased activity)</li>
    <li><b>Vertical patterns:</b> May indicate day-of-week effects</li>
    <li><b>Horizontal patterns:</b> May show weekly trends or cycles</li>
    <li><b>Clusters:</b> Groups of dark squares indicate periods of poor control</li>
    </ul>
        </div>
        """, unsafe_allow_html=True)

def render_section_circadian(daily_df):
    """Render Circadian Pattern Analysis section."""
    st.markdown('<div class="main-header">🕐 Circadian Pattern Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    Circadian pattern analysis reveals the temporal distribution of seizure activity throughout the day 
    and week. Understanding these patterns can help identify triggers, optimize medication timing, and 
    provide insights into the underlying neurological rhythms affecting seizure occurrence.
    </div>
    """, unsafe_allow_html=True)
    
    # Generate temporal data
    with st.spinner("Generating temporal seizure distribution..."):
        temporal_df = generate_temporal_seizure_data(daily_df)
    
    if len(temporal_df) == 0:
        st.warning("No seizure data available for temporal analysis.")
        return
    
    # Statistics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate peak hour
    hourly_dist = temporal_df.groupby('hour').size()
    peak_hour = hourly_dist.idxmax() if len(hourly_dist) > 0 else 0
    peak_count = hourly_dist.max() if len(hourly_dist) > 0 else 0
    
    # Calculate most common day
    dow_dist = temporal_df.groupby('day_of_week').size()
    peak_dow = dow_dist.idxmax() if len(dow_dist) > 0 else 1
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Morning vs evening
    morning_count = len(temporal_df[(temporal_df['hour'] >= 6) & (temporal_df['hour'] < 12)])
    evening_count = len(temporal_df[(temporal_df['hour'] >= 18) & (temporal_df['hour'] < 24)])
    night_count = len(temporal_df[(temporal_df['hour'] >= 0) & (temporal_df['hour'] < 6)])
    afternoon_count = len(temporal_df[(temporal_df['hour'] >= 12) & (temporal_df['hour'] < 18)])
    
    with col1:
        hour_label = f"{peak_hour:02d}:00" if peak_hour < 12 else f"{peak_hour:02d}:00"
        period = "AM" if peak_hour < 12 else "PM"
        st.metric(
            label="🔴 Peak Hour",
            value=f"{hour_label} {period}",
            delta=f"{peak_count} seizures"
        )
    
    with col2:
        st.metric(
            label="📅 Most Active Day",
            value=day_names[peak_dow-1][:3],
            delta=f"{dow_dist.max()} seizures"
        )
    
    with col3:
        max_period = max([
            ('Morning', morning_count),
            ('Afternoon', afternoon_count),
            ('Evening', evening_count),
            ('Night', night_count)
        ], key=lambda x: x[1])
        st.metric(
            label="⏰ Peak Period",
            value=max_period[0],
            delta=f"{max_period[1]} seizures"
        )
    
    with col4:
        total_seizures = len(temporal_df)
        avg_per_hour = total_seizures / 24
        st.metric(
            label="📊 Avg per Hour",
            value=f"{avg_per_hour:.1f}",
            delta=f"{total_seizures} total"
        )
    
    st.markdown("---")
    
    # Granularity selector
    st.markdown('<div class="subsection-header">⏱️ Temporal Granularity Control</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        granularity = st.radio(
            "Select time granularity:",
            options=['hour', '6hour', 'day', 'week', 'month', 'year'],
            format_func=lambda x: {
                'hour': '⏰ 1 Hour (Detailed)',
                '6hour': '⏰ 6 Hours (Periods)',
                'day': '📅 Day of Week',
                'week': '📆 Weekly',
                'month': '📊 Monthly',
                'year': '🗓️ Yearly'
            }[x],
            index=0
        )
        
        st.markdown("---")
        
        # Severe seizures toggle
        show_severe_only = st.toggle(
            "🔴 Show only severe seizures",
            value=False,
            help="When enabled, displays only severe seizures instead of total seizures"
        )
    
    with col2:
        st.markdown("""
        **Granularity Guide:**
        - **1 Hour**: Detailed circadian patterns - seizures by time of day
        - **6 Hours**: Period view (Night/Morning/Afternoon/Evening)
        - **Day**: Weekly patterns - which days have more activity
        - **Week**: 52-week view - identify problem weeks
        - **Month**: Monthly trends - seasonal patterns
        - **Year**: Overall distribution - big picture view
        
        **Severe Seizures Toggle:**
        - Enable to focus only on severe seizure events
        - Helps identify patterns in most critical episodes
        """)
    
    # Main visualization based on granularity
    seizure_label = "severe seizure" if show_severe_only else "seizure"
    
    if granularity in ['hour', '6hour']:
        st.markdown('<div class="subsection-header">⭕ Circadian Clock Visualization</div>', unsafe_allow_html=True)
        insight_text = f"""
        <b>Reading the Clock:</b> This radial chart displays {seizure_label} frequency around a 24-hour clock face. 
        Longer bars indicate more {seizure_label}s at that time. The color intensity also reflects {seizure_label} density, 
        with darker colors showing peak activity times.
        """
    elif granularity == 'day':
        st.markdown('<div class="subsection-header">⭕ Day of Week Pattern</div>', unsafe_allow_html=True)
        insight_text = f"""
        <b>Weekly Patterns:</b> This chart shows {seizure_label} distribution across days of the week. 
        Identify if certain days are more problematic - this could relate to weekly routines, 
        medication timing, stress patterns, or sleep schedules.
        """
    elif granularity == 'week':
        st.markdown('<div class="subsection-header">⭕ 52-Week Distribution</div>', unsafe_allow_html=True)
        insight_text = f"""
        <b>Annual Patterns:</b> View {seizure_label} burden across all 52 weeks of the year. 
        Identify difficult periods, seasonal trends, or the impact of medication changes. 
        Longer bars indicate weeks with more {seizure_label}s.
        """
    elif granularity == 'month':
        st.markdown('<div class="subsection-header">⭕ Monthly Patterns</div>', unsafe_allow_html=True)
        insight_text = f"""
        <b>Seasonal Analysis:</b> This monthly view reveals seasonal patterns in {seizure_label} activity. 
        Some epilepsy types show seasonal variations related to weather, daylight, stress, 
        or other environmental factors.
        """
    else:  # year
        st.markdown('<div class="subsection-header">⭕ Yearly Overview</div>', unsafe_allow_html=True)
        insight_text = f"""
        <b>Big Picture:</b> This quarterly breakdown provides the highest-level view of {seizure_label} 
        distribution across the year. Compare quarters to identify longer-term trends and patterns.
        """
    
    fig = plot_by_granularity(daily_df, temporal_df, granularity, show_severe_only)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div class="insight-box">
    {insight_text}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Additional visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">📅 Day of Week Distribution</div>', unsafe_allow_html=True)
        fig_dow = plot_day_of_week_distribution(temporal_df)
        if fig_dow:
            st.plotly_chart(fig_dow, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>Insight:</b> This chart shows whether certain days of the week are associated with 
            higher seizure frequency. Patterns may relate to weekly routines, stress levels, sleep 
            schedules, or medication timing.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-header">🌅 Time Period Breakdown</div>', unsafe_allow_html=True)
        
        # Create period breakdown chart
        periods = ['Night\n(12am-6am)', 'Morning\n(6am-12pm)', 
                  'Afternoon\n(12pm-6pm)', 'Evening\n(6pm-12am)']
        counts = [night_count, morning_count, afternoon_count, evening_count]
        colors = ['#2C3E50', '#F39C12', '#3498DB', '#8E44AD']
        
        fig_period = go.Figure(data=[go.Bar(
            x=periods,
            y=counts,
            marker=dict(color=colors),
            text=counts,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Seizures: %{y}<extra></extra>'
        )])
        
        fig_period.update_layout(
            title='Seizures by Time Period',
            yaxis_title='Number of Seizures',
            template='plotly_white',
            height=400,
            showlegend=False,
            yaxis=dict(gridcolor='#ECF0F1')
        )
        
        st.plotly_chart(fig_period, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b> Grouping by major periods of the day can reveal macro patterns. 
        Some epilepsy types show strong circadian preferences, with nocturnal seizures 
        occurring during sleep or morning seizures upon waking.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hourly x Day of Week Heatmap
    st.markdown('<div class="subsection-header">🔥 Comprehensive Temporal Heatmap</div>', unsafe_allow_html=True)
    
    fig_heatmap = plot_weekly_temporal_heatmap(temporal_df)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Advanced Analysis:</b> This heatmap combines both hour-of-day and day-of-week dimensions, 
        revealing complex temporal patterns. Look for:
        <ul>
        <li><b>Hot spots:</b> Specific day/time combinations with high seizure activity</li>
        <li><b>Vertical bands:</b> Particular hours that are problematic across all days</li>
        <li><b>Horizontal bands:</b> Specific days with elevated risk throughout</li>
        <li><b>Diagonal patterns:</b> May indicate weekly cycles or routines</li>
        </ul>
        This level of detail can guide medication timing, activity planning, and trigger identification.
        </div>
        """, unsafe_allow_html=True)
    
    # Clinical recommendations
    st.markdown("---")
    st.markdown('<div class="subsection-header">💡 Clinical Insights & Recommendations</div>', unsafe_allow_html=True)
    
    # Determine pattern type
    morning_pct = (morning_count / len(temporal_df)) * 100 if len(temporal_df) > 0 else 0
    evening_pct = (evening_count / len(temporal_df)) * 100 if len(temporal_df) > 0 else 0
    night_pct = (night_count / len(temporal_df)) * 100 if len(temporal_df) > 0 else 0
    
    recommendations = []
    
    if morning_pct > 35:
        recommendations.append(
            "🌅 **Morning Peak Detected**: Consider taking primary medications before bedtime "
            "to ensure peak blood levels during morning hours. Evaluate sleep quality and "
            "morning routines for potential triggers."
        )
    
    if evening_pct > 35:
        recommendations.append(
            "🌙 **Evening Peak Detected**: Evening seizures may relate to medication wearing off "
            "or accumulated daily stress. Consider split dosing or adding an evening dose. "
            "Ensure adequate rest and stress management in late afternoon."
        )
    
    if night_pct > 30:
        recommendations.append(
            "😴 **Nocturnal Pattern Detected**: Night seizures often relate to sleep stages. "
            "Ensure good sleep hygiene, consider sleep study to rule out sleep apnea, and "
            "discuss nocturnal medication coverage with your neurologist."
        )
    
    # Check for day-of-week patterns
    weekend_count = len(temporal_df[temporal_df['day_of_week'].isin([6, 7])])
    weekday_count = len(temporal_df[temporal_df['day_of_week'].isin([1, 2, 3, 4, 5])])
    
    if weekend_count > 0 and weekday_count > 0:
        weekend_ratio = weekend_count / (2 * 52)  # Average per weekend day
        weekday_ratio = weekday_count / (5 * 52)  # Average per weekday
        
        if weekend_ratio > weekday_ratio * 1.3:
            recommendations.append(
                "📅 **Weekend Elevation**: Seizures are more frequent on weekends. This may relate to "
                "sleep schedule changes, medication timing variations, or alcohol/social activities. "
                "Maintain consistent routines even on weekends."
            )
        elif weekday_ratio > weekend_ratio * 1.3:
            recommendations.append(
                "💼 **Weekday Stress Pattern**: Higher weekday seizure frequency suggests stress, "
                "sleep deprivation, or work-related triggers. Consider stress management techniques "
                "and ensure adequate rest during the work week."
            )
    
    if not recommendations:
        recommendations.append(
            "📊 **Distributed Pattern**: Seizures are relatively evenly distributed across times and days. "
            "This suggests triggers may be less time-dependent. Focus on other factors like medication "
            "compliance, overall sleep quality, and consistent routines."
        )
    
    for rec in recommendations:
        st.markdown(f"""
        <div class="insight-box">
        {rec}
        </div>
        """, unsafe_allow_html=True)

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    """Main application function."""
    
    # Apply custom CSS
    apply_custom_css()
    
    # File uploader
    st.title("🏥 Seizure Analytics Dashboard")
    st.markdown("### Multi-Patient Epilepsy Monitoring System")
    
    # Option to use demo data or upload files
    use_demo = st.checkbox("📁 Use demo data (pre-loaded dummy dataset)", value=True)
    
    if use_demo:
        # Load demo CSV files
        with st.spinner("Loading demo data..."):
            try:
                # Load raw data
                daily_df_all = pd.read_csv('data/seizures_dummy.csv')
                med_df_all = pd.read_csv('data/medications_dummy.csv')
                assessment_sparse_all = pd.read_csv('data/assessments_dummy.csv')
                
                # Get available patients
                available_patients = sorted(daily_df_all['patient_id'].unique())
                st.session_state.available_patients = available_patients
                
                # Calculate patient summaries
                patient_summaries = {}
                for pid in available_patients:
                    patient_data = daily_df_all[daily_df_all['patient_id'] == pid]
                    patient_summaries[pid] = {
                        'total_seizures': int(patient_data['daily_total'].sum()),
                        'severe_seizures': int(patient_data['daily_severe'].sum()),
                        'days': len(patient_data)
                    }
                st.session_state.patient_summaries = patient_summaries
                
                st.success(f"✅ Demo data loaded successfully! {len(available_patients)} patients available.")
                
            except Exception as e:
                st.error(f"Error loading demo data: {str(e)}")
                st.info("Demo files not found. Please upload your own CSV files below.")
                use_demo = False
    
    if not use_demo:
        st.markdown("### Upload your CSV files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            seizures_file = st.file_uploader(
                "Seizures CSV",
                type=['csv'],
                help="CSV with columns: date, day, week, day_of_week, daily_total, daily_severe"
            )
        
        with col2:
            medications_file = st.file_uploader(
                "Medications CSV",
                type=['csv'],
                help="CSV with columns: date, day, Lamictal, Clonazepam, Vimpat, Zonisamide, Fycompa"
            )
        
        with col3:
            assessments_file = st.file_uploader(
                "Assessments CSV",
                type=['csv'],
                help="CSV with columns: date, day, week, QoL, Anxiety, Depression, Behavioral"
            )
        
        if not all([seizures_file, medications_file, assessments_file]):
            st.info("👆 Please upload all three CSV files to start the analysis")
            
            st.markdown("---")
            st.markdown("### 📋 Expected CSV Format")
            st.markdown("""
            Upload three separate CSV files with the following structures:
            
            **1. seizures.csv:**
            - `date`: Date (YYYY-MM-DD)
            - `day`: Day number (1-364)
            - `week`: Week number (1-52)
            - `day_of_week`: Day of week (1=Monday, 7=Sunday)
            - `daily_total`: Total seizures that day
            - `daily_severe`: Severe seizures that day
            
            **2. medications.csv:**
            - `date`: Date (YYYY-MM-DD)
            - `day`: Day number (1-364)
            - `Lamictal`: Dose in mg
            - `Clonazepam`: Dose in mg
            - `Vimpat`: Dose in mg
            - `Zonisamide`: Dose in mg
            - `Fycompa`: Dose in mg
            
            **3. assessments.csv:**
            - `date`: Date (YYYY-MM-DD)
            - `day`: Day number (1-364)
            - `week`: Week number (1-52)
            - `QoL`: Quality of Life score (0-100)
            - `Anxiety`: Anxiety score (0-100)
            - `Depression`: Depression score (0-100)
            - `Behavioral`: Behavioral score (0-100)
            
            Note: Assessments can be sparse (not every day needs an entry).
            """)
            
            st.stop()
        
        # Load and process uploaded data
        with st.spinner("Loading and processing data..."):
            daily_df_all, med_df_all, assessment_sparse_all = load_csv_data(seizures_file, medications_file, assessments_file)
            
            if daily_df_all is None:
                st.error("Failed to load CSV files. Please check the file format.")
                st.stop()
            
            # Get available patients
            if 'patient_id' in daily_df_all.columns:
                available_patients = sorted(daily_df_all['patient_id'].unique())
                st.session_state.available_patients = available_patients
                
                # Calculate patient summaries
                patient_summaries = {}
                for pid in available_patients:
                    patient_data = daily_df_all[daily_df_all['patient_id'] == pid]
                    patient_summaries[pid] = {
                        'total_seizures': int(patient_data['daily_total'].sum()),
                        'severe_seizures': int(patient_data['daily_severe'].sum()),
                        'days': len(patient_data)
                    }
                st.session_state.patient_summaries = patient_summaries
            else:
                # Single patient data without patient_id column
                st.session_state.available_patients = ['Single Patient']
                st.session_state.patient_summaries = {
                    'Single Patient': {
                        'total_seizures': int(daily_df_all['daily_total'].sum()),
                        'severe_seizures': int(daily_df_all['daily_severe'].sum()),
                        'days': len(daily_df_all)
                    }
                }
            
            st.success("✅ Data loaded successfully!")
    
    # Sidebar navigation
    selection, selected_patient = sidebar_menu()
    
    # Filter data by selected patient
    if 'available_patients' in st.session_state and selected_patient is not None:
        # Filter data for selected patient
        if 'patient_id' in daily_df_all.columns:
            daily_df = daily_df_all[daily_df_all['patient_id'] == selected_patient].copy()
            med_df_raw = med_df_all[med_df_all['patient_id'] == selected_patient].copy()
            assessment_sparse = assessment_sparse_all[assessment_sparse_all['patient_id'] == selected_patient].copy()
        else:
            # Single patient data without patient_id column
            daily_df = daily_df_all.copy()
            med_df_raw = med_df_all.copy()
            assessment_sparse = assessment_sparse_all.copy()
        
        # Prepare data
        med_df = prepare_medication_dataframe(med_df_raw)
        assessment_df = prepare_assessment_dataframe(assessment_sparse)
        
        # Create weekly aggregates
        weekly_df = create_weekly_aggregates(daily_df, med_df, assessment_df)
        
        # Show data summary in expander
        with st.expander("📊 Current Patient Data Summary"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Days Tracked", len(daily_df))
                st.metric("Total Seizures", int(daily_df['daily_total'].sum()))
            with col2:
                st.metric("Medications Tracked", 5)
                st.metric("Severe Seizures", int(daily_df['daily_severe'].sum()))
            with col3:
                st.metric("Assessment Points", len(assessment_sparse))
                if len(daily_df) > 0:
                    st.metric("Date Range", f"{daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]}")
        
        # Render selected section
        if selection == "🏠 Main Dashboard":
            render_main_dashboard(daily_df, weekly_df, med_df, assessment_df)
        
        elif selection == "📈 Seizure Analysis":
            render_seizure_analysis(daily_df, weekly_df)
        
        elif selection == "💊 Treatment Management":
            render_treatment_management(med_df, weekly_df, daily_df)
        
        elif selection == "🧠 Patient Outcomes":
            render_patient_outcomes(assessment_df, weekly_df, daily_df)
        
        elif selection == "📋 Clinical Report":
            render_clinical_report(daily_df, weekly_df, med_df, assessment_df)
    else:
        st.warning("⚠️ Please load data and select a patient from the sidebar to view the dashboard.")

# ========================================
# RUN APPLICATION
# ========================================

if __name__ == "__main__":
    main()

