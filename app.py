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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    month_positions = [0, 4, 9, 13, 17, 22, 26, 30, 35, 39, 43, 48]  # Approx week numbers
    month_names = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
    
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
            # Get the actual day number from the mapping, or calculate if not found
            day_num = day_mapping.get((dow, week), week * 7 + dow + 1)
            
            if count == 0:
                hover_str = f"{day_names[dow]}, Day {day_num}<br>No seizures"
            elif count == 1:
                hover_str = f"{day_names[dow]}, Day {day_num}<br>1 seizure"
            else:
                hover_str = f"{day_names[dow]}, Day {day_num}<br>{count} seizures"
            row_hover.append(hover_str)
        hover_text.append(row_hover)
    
    # Create custom colorscale similar to GitHub (light to dark green)
    # Using white/very light gray for 0, then shades of green
    max_val = np.max(grid)
    
    # GitHub-style green colorscale
    colorscale = [
        [0, '#EBEDF0'],      # Light gray for 0
        [0.2, '#C6E48B'],    # Light green
        [0.4, '#7BC96F'],    # Medium-light green
        [0.6, '#239A3B'],    # Medium green
        [0.8, '#196127'],    # Dark green
        [1.0, '#0D3D17']     # Very dark green
    ]
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=list(range(52)),
        y=['Mon', '', 'Wed', '', 'Fri', '', 'Sun'],  # Only show Mon, Wed, Fri, Sun like GitHub
        colorscale=colorscale,
        showscale=True,
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        colorbar=dict(
            title=dict(text="Seizures", side="right"),
            thickness=15,
            len=0.5,
            x=1.02,
            tickmode='linear',
            tick0=0,
            dtick=max(1, max_val // 5) if max_val > 0 else 1
        ),
        xgap=3,  # Gap between cells horizontally
        ygap=3,  # Gap between cells vertically
    ))
    
    # Add month labels at the top
    month_annotations = []
    for i, (pos, name) in enumerate(zip(month_positions, month_names)):
        if pos < 52:
            month_annotations.append(
                dict(
                    x=pos,
                    y=7.3,
                    text=name,
                    showarrow=False,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(size=11, color='#666')
                )
            )
    
    # Update layout for GitHub-style appearance
    fig.update_layout(
        title=dict(
            text='TachyGrid — Seizure Activity Heatmap',
            font=dict(size=16, color='#2C3E50', family='Arial, sans-serif'),
            x=0,
            xanchor='left'
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
            tickfont=dict(size=10, color='#666'),
            autorange='reversed'  # Mon at top, Sun at bottom
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=220,
        margin=dict(l=50, r=100, t=80, b=20),
        annotations=month_annotations
    )
    
    # Remove axis lines
    fig.update_xaxes(showline=False)
    fig.update_yaxes(showline=False)
    
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
# APP LAYOUT FUNCTIONS
# ========================================

def sidebar_menu():
    """Create sidebar navigation menu."""
    st.sidebar.markdown("## 🏥 Navigation")
    
    menu_options = [
        "📊 Overview Dashboard",
        "📈 Seizure Activity",
        "💊 Medication Dashboard",
        "🧠 Mental Health & QoL",
        "🔬 Integrated Clinical View",
        "🟩 TachyGrid Heatmap",
        "🕐 Circadian Pattern Analysis"
    ]
    
    selection = st.sidebar.radio("Go to:", menu_options)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides comprehensive analytics for a 52-week "
        "epilepsy monitoring dataset, including seizure frequency, "
        "medication trajectories, and patient-reported outcomes."
    )
    
    return selection

def render_section_overview(daily_df, weekly_df):
    """Render Overview Dashboard section."""
    st.markdown('<div class="main-header">📊 Overview Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    This dashboard analyzes a comprehensive 52-week dataset tracking seizure activity, 
    medication regimens, and patient-reported mental health assessments for an individual 
    with epilepsy. The data provides insights into disease patterns, treatment responses, 
    and quality of life metrics.
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate KPIs
    total_seizures = int(daily_df['daily_total'].sum())
    total_severe = int(daily_df['daily_severe'].sum())
    avg_per_week = daily_df['daily_total'].sum() / 52
    weeks_with_severe = len(weekly_df[weekly_df['severe_seizures'] > 0])
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Seizures This Year",
            value=f"{total_seizures:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🚨 Total Severe Seizures",
            value=f"{total_severe:,}",
            delta=f"{(total_severe/total_seizures*100):.1f}% of total" if total_seizures > 0 else "0%"
        )
    
    with col3:
        st.metric(
            label="📈 Average Seizures/Week",
            value=f"{avg_per_week:.1f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="⚠️ Weeks with ≥1 Severe",
            value=f"{weeks_with_severe}",
            delta=f"{(weeks_with_severe/52*100):.1f}% of weeks"
        )
    
    st.markdown("---")
    
    # Quick visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_daily_seizures(daily_df)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b> This timeline reveals daily seizure patterns, including periods of 
        increased activity and relative calm, which may correlate with medication adjustments 
        or other factors.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fig = plot_weekly_seizures(weekly_df)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b> Weekly aggregation helps identify broader trends and patterns in 
        seizure burden, with severe seizures highlighted for clinical attention.
        </div>
        """, unsafe_allow_html=True)

def render_section_seizures(daily_df, weekly_df):
    """Render Seizure Activity section."""
    st.markdown('<div class="main-header">📈 Seizure Activity Analysis</div>', unsafe_allow_html=True)
    
    # Daily Timeline
    st.markdown('<div class="subsection-header">Daily Seizure Timeline</div>', unsafe_allow_html=True)
    fig = plot_daily_seizures(daily_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Clinical Insight:</b> The daily timeline shows temporal patterns in seizure occurrence. 
    Peaks may indicate periods of poor seizure control, while valleys represent periods of better 
    disease management. Look for patterns that correlate with medication changes or external factors.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Weekly Burden
    st.markdown('<div class="subsection-header">Weekly Seizure Burden</div>', unsafe_allow_html=True)
    fig = plot_weekly_seizures(weekly_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Clinical Insight:</b> Weekly aggregation smooths out day-to-day variability and reveals 
    longer-term trends. The overlay of severe seizures helps prioritize clinical attention to 
    periods of highest concern.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Heatmap Calendar
    st.markdown('<div class="subsection-header">52-Week Seizure Calendar Heatmap</div>', unsafe_allow_html=True)
    fig = plot_heatmap_calendar(daily_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Clinical Insight:</b> This calendar view provides an intuitive visualization of seizure 
    intensity across the entire year. Darker colors indicate higher seizure counts. Look for 
    patterns by day of week or specific time periods.
    </div>
    """, unsafe_allow_html=True)

def render_section_medications(med_df, weekly_df):
    """Render Medication Dashboard section."""
    st.markdown('<div class="main-header">💊 Medication Dashboard</div>', unsafe_allow_html=True)
    
    # Medication trajectories
    st.markdown('<div class="subsection-header">Medication Dose Trajectories</div>', unsafe_allow_html=True)
    fig = plot_medications(med_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Clinical Insight:</b> This chart tracks all anti-seizure medication doses over the 
    52-week period. Changes in dosing may reflect titration strategies, side effect management, 
    or efforts to optimize seizure control.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dose change detection
    st.markdown('<div class="subsection-header">Dose Change Detection</div>', unsafe_allow_html=True)
    
    changes_df = detect_dose_changes(med_df, threshold=5)
    
    if len(changes_df) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_dose_changes(med_df, changes_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Detected Dose Changes:**")
            st.dataframe(
                changes_df[['day', 'medication', 'change_type', 'change_amount']].style.format({
                    'day': '{:.0f}',
                    'change_amount': '{:+.1f} mg'
                }),
                height=400
            )
    else:
        st.info("No significant dose changes detected (threshold: 5mg)")
    
    st.markdown("""
    <div class="insight-box">
    <b>Clinical Insight:</b> Automatic detection of dose changes helps identify key decision 
    points in treatment. Up-titrations may represent attempts to improve seizure control, while 
    down-titrations might reflect side effect management.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Medication-Seizure Interaction Explorer
    st.markdown('<div class="subsection-header">Medication-Seizure Interaction Explorer</div>', unsafe_allow_html=True)
    
    medications = ['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']
    selected_med = st.selectbox("Select medication to analyze:", medications)
    
    fig = plot_dose_response(weekly_df, selected_med)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        plot_data = weekly_df[[selected_med, 'total_seizures']].dropna()
        if len(plot_data) > 1:
            corr, p_value = pearsonr(plot_data[selected_med], plot_data['total_seizures'])
            
            st.markdown(f"""
            <div class="insight-box">
            <b>Statistical Analysis:</b> Correlation coefficient: {corr:.3f} (p={p_value:.4f})<br>
            {
                "A negative correlation suggests higher doses may be associated with fewer seizures." if corr < 0 
                else "A positive correlation may indicate dose increases in response to worsening seizures."
            }
            <br><br>
            <b>Note:</b> Correlation does not imply causation. These patterns should be interpreted 
            in the context of clinical decision-making and other factors.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(f"Insufficient data for {selected_med} analysis")

def render_section_assessments(assessment_df, weekly_df):
    """Render Mental Health & QoL section."""
    st.markdown('<div class="main-header">🧠 Mental Health & Quality of Life</div>', unsafe_allow_html=True)
    
    # Assessment time series
    st.markdown('<div class="subsection-header">Patient-Reported Assessment Timeline</div>', unsafe_allow_html=True)
    fig = plot_assessments(assessment_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Clinical Insight:</b> Patient-reported outcomes provide crucial context beyond seizure 
    counts alone. These assessments capture the lived experience of epilepsy, including quality 
    of life, mental health symptoms, and behavioral changes.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # QoL vs Seizures
    st.markdown('<div class="subsection-header">Quality of Life vs Seizure Burden</div>', unsafe_allow_html=True)
    fig = plot_seizure_qol_overlay(weekly_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Clinical Narrative:</b> Periods of increased seizure activity often correspond to 
    decreases in self-reported quality of life. This dual-axis visualization illustrates the 
    profound impact that seizure control has on overall well-being and daily functioning.
    <br><br>
    The inverse relationship between seizure frequency and QoL underscores the importance of 
    optimizing treatment to minimize both seizure occurrence and treatment burden.
    </div>
    """, unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown("---")
    st.markdown('<div class="subsection-header">Assessment Summary Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    assessments = ['QoL', 'Anxiety', 'Depression', 'Behavioral']
    cols = [col1, col2, col3, col4]
    
    for assessment, col in zip(assessments, cols):
        data = assessment_df[assessment].dropna()
        if len(data) > 0:
            with col:
                st.metric(
                    label=f"📊 {assessment}",
                    value=f"{data.mean():.1f}",
                    delta=f"Range: {data.min():.1f}-{data.max():.1f}"
                )
        else:
            with col:
                st.metric(label=f"📊 {assessment}", value="N/A")

def render_section_integrated(daily_df, weekly_df, med_df, assessment_df):
    """Render Integrated Clinical Dashboard section."""
    st.markdown('<div class="main-header">🔬 Integrated Clinical Dashboard</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 Disease Activity",
        "💊 Medication Changes",
        "🧠 Patient-Reported Outcomes",
        "📊 Correlations Panel"
    ])
    
    with tab1:
        st.markdown('<div class="subsection-header">Disease Activity Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_daily_seizures(daily_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = plot_weekly_seizures(weekly_df)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Disease Activity Summary:</b> This view provides a comprehensive look at seizure 
        patterns across multiple time scales. Daily data reveals granular patterns, while 
        weekly aggregation shows broader trends in disease control.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="subsection-header">Medication Management</div>', unsafe_allow_html=True)
        
        fig = plot_medications(med_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show dose changes table
        changes_df = detect_dose_changes(med_df, threshold=5)
        
        if len(changes_df) > 0:
            st.markdown("**Timeline of Medication Changes:**")
            st.dataframe(
                changes_df.sort_values('day'),
                use_container_width=True
            )
        
        st.markdown("""
        <div class="insight-box">
        <b>Medication Strategy:</b> The medication trajectories reveal the complexity of 
        polytherapy in epilepsy management. Multiple medications are often required to achieve 
        adequate seizure control, with frequent adjustments to optimize efficacy and minimize 
        side effects.
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="subsection-header">Patient-Reported Outcomes</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_assessments(assessment_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = plot_seizure_qol_overlay(weekly_df)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Holistic Patient View:</b> Patient-reported outcomes are essential for comprehensive 
        epilepsy care. These measures capture dimensions of health that seizure counts alone 
        cannot reflect, including psychological well-being, functional status, and overall 
        quality of life.
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="subsection-header">Clinical Correlations</div>', unsafe_allow_html=True)
        
        fig = plot_correlation_heatmap(weekly_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Correlation Analysis:</b> This heatmap reveals relationships between different 
        clinical variables. Strong correlations (positive or negative) may suggest important 
        clinical relationships, though causation cannot be inferred from correlation alone.
        <br><br>
        <b>Key patterns to look for:</b>
        <ul>
        <li>Negative correlations between medication doses and seizure frequency (suggesting efficacy)</li>
        <li>Negative correlations between seizures and quality of life (showing impact)</li>
        <li>Relationships between mental health metrics and disease activity</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Comprehensive integrated dashboard
        st.markdown("---")
        st.markdown('<div class="subsection-header">Multi-Panel Integrated View</div>', unsafe_allow_html=True)
        
        fig = plot_integrated_dashboard(daily_df, weekly_df, med_df, assessment_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Integrated Clinical Picture:</b> This comprehensive dashboard brings together all 
        key clinical metrics in a single view, enabling holistic assessment of disease activity, 
        treatment response, and patient well-being across the 52-week monitoring period.
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
    st.markdown("### Upload your CSV dataset files to begin analysis")
    
    # Option to use demo data or upload files
    use_demo = st.checkbox("📁 Use demo data (pre-loaded dummy dataset)", value=True)
    
    if use_demo:
        # Load demo CSV files
        with st.spinner("Loading demo data..."):
            try:
                daily_df = pd.read_csv('data/seizures_dummy.csv')
                med_df = pd.read_csv('data/medications_dummy.csv')
                assessment_sparse = pd.read_csv('data/assessments_dummy.csv')
                
                # Prepare data
                med_df = prepare_medication_dataframe(med_df)
                assessment_df = prepare_assessment_dataframe(assessment_sparse)
                
                # Create weekly aggregates
                weekly_df = create_weekly_aggregates(daily_df, med_df, assessment_df)
                
                st.success("✅ Demo data loaded successfully!")
                
                # Show data summary
                with st.expander("📊 Data Summary"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Days", len(daily_df))
                        st.metric("Total Seizures", int(daily_df['daily_total'].sum()))
                    with col2:
                        st.metric("Medications", len(['Lamictal', 'Clonazepam', 'Vimpat', 'Zonisamide', 'Fycompa']))
                        st.metric("Severe Seizures", int(daily_df['daily_severe'].sum()))
                    with col3:
                        st.metric("Assessment Points", len(assessment_sparse))
                        st.metric("Date Range", f"{daily_df['date'].iloc[0]} to {daily_df['date'].iloc[-1]}")
                
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
            daily_df, med_df_raw, assessment_sparse = load_csv_data(seizures_file, medications_file, assessments_file)
            
            if daily_df is None:
                st.error("Failed to load CSV files. Please check the file format.")
                st.stop()
            
            # Prepare data
            med_df = prepare_medication_dataframe(med_df_raw)
            assessment_df = prepare_assessment_dataframe(assessment_sparse)
            
            st.success("✅ Data loaded successfully!")
        
        # Create weekly aggregates
        weekly_df = create_weekly_aggregates(daily_df, med_df, assessment_df)
    
    # Sidebar navigation
    selection = sidebar_menu()
    
    # Render selected section
    if selection == "📊 Overview Dashboard":
        render_section_overview(daily_df, weekly_df)
    
    elif selection == "📈 Seizure Activity":
        render_section_seizures(daily_df, weekly_df)
    
    elif selection == "💊 Medication Dashboard":
        render_section_medications(med_df, weekly_df)
    
    elif selection == "🧠 Mental Health & QoL":
        render_section_assessments(assessment_df, weekly_df)
    
    elif selection == "🔬 Integrated Clinical View":
        render_section_integrated(daily_df, weekly_df, med_df, assessment_df)
    
    elif selection == "🟩 TachyGrid Heatmap":
        render_section_tachygrid(daily_df)
    
    elif selection == "🕐 Circadian Pattern Analysis":
        render_section_circadian(daily_df)

# ========================================
# RUN APPLICATION
# ========================================

if __name__ == "__main__":
    main()

