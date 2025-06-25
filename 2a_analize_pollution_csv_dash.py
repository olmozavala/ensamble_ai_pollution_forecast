import os
from typing import List, Dict, Any
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from os.path import join

# Load config and get data folder path
with open('config.json', 'r') as f:
    config = json.load(f)

# Path to the folder containing the CSV files
data_folder = config['data_loader']['args']['data_folder']
pollution_folder = join(data_folder, "PollutionCSV")

def list_csv_files(folder: str) -> List[str]:
    """
    List all CSV files in the given folder.
    Args:
        folder (str): Path to the folder.
    Returns:
        List[str]: List of CSV filenames (not full paths).
    """
    return [f for f in os.listdir(folder) if f.endswith('.csv')]

def summarize_csv(file_path: str) -> Dict[str, Any]:
    """
    Summarize the CSV file: shape, stats, and column type counts.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        Dict[str, Any]: Summary including shape, stats, and column type counts.
    """
    df = pd.read_csv(file_path, nrows=10000)  # Read a sample for speed
    total_rows, total_cols = df.shape
    stats = df.describe(include='all').T.reset_index().rename(columns={'index': 'column'})
    # Identify column types
    pollutant_cols = [col for col in df.columns if col.startswith('cont_')]
    imputed_cols = [col for col in df.columns if col.startswith('i_cont_')]
    time_cols = [col for col in df.columns if any(t in col for t in ['date', 'hour', 'day', 'week', 'year'])]
    return {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'stats': stats,
        'pollutant_count': len(pollutant_cols),
        'imputed_count': len(imputed_cols),
        'time_count': len(time_cols),
    }

# Dash app setup
app = dash.Dash(__name__)
app.title = "Pollution CSV Explorer"

app.layout = html.Div([
    html.H1("Pollution CSV Explorer"),
    dcc.Dropdown(
        id='csv-file-dropdown',
        options=[{'label': f, 'value': f} for f in sorted(list_csv_files(pollution_folder))],
        placeholder="Select a CSV file",
        style={'width': '50%'},
        value='data_imputed_7_2024.csv'
    ),
    html.Div(id='summary-output'),
    html.Hr(),
    html.H3("Column Statistics (first 10 columns shown)"),
    dash_table.DataTable(
        id='stats-table',
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'minWidth': '120px', 'maxWidth': '300px', 'whiteSpace': 'normal'},
    ),
    html.Hr(),
    html.H3("First 3 Rows of the File"),
    dash_table.DataTable(
        id='preview-table',
        page_size=3,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'minWidth': '120px', 'maxWidth': '300px', 'whiteSpace': 'normal'},
    ),
    html.Hr(),
    html.H3("Time Series Analysis"),
    html.Div([
        html.Label("Start Time (hours from beginning):"),
        dcc.Slider(
            id='start-time-slider',
            min=0,
            max=8760,  # Max hours in a year
            step=24,   # Step by days
            value=0,
            marks={i: f'{i//24}d' for i in range(0, 8761, 168)},  # Mark every week
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Br(),
        html.Label("Number of Weeks to Plot:"),
        dcc.Slider(
            id='weeks-slider',
            min=1,
            max=52,
            step=1,
            value=4,
            marks={i: f'{i}w' for i in range(1, 53, 4)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ]),
    dcc.Graph(id='time-series-plot')
])

@app.callback(
    [Output('summary-output', 'children'), Output('stats-table', 'data'), Output('stats-table', 'columns'),
     Output('preview-table', 'data'), Output('preview-table', 'columns')],
    [Input('csv-file-dropdown', 'value')]
)
def update_summary(selected_file: str):
    """
    Update the summary, stats table, and preview table when a file is selected.
    Args:
        selected_file (str): The selected CSV filename.
    Returns:
        Tuple: (summary HTML, stats data, stats columns, preview data, preview columns)
    """
    if not selected_file:
        return html.Div("Select a file to see summary."), [], [], [], []
    file_path = os.path.join(pollution_folder, selected_file)
    summary = summarize_csv(file_path)
    summary_html = html.Div([
        html.P(f"Total rows: {summary['total_rows']:,}"),
        html.P(f"Total columns: {summary['total_cols']:,}"),
        html.P(f"Pollutant columns: {summary['pollutant_count']}"),
        html.P(f"Imputed columns: {summary['imputed_count']}"),
        html.P(f"Time columns: {summary['time_count']}"),
    ])
    # Show only first 10 columns for stats
    stats_data = summary['stats'].iloc[:10].to_dict('records')
    # Round all float values in stats_data to two decimals
    for row in stats_data:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 2)
    stats_columns = [{'name': c, 'id': c} for c in summary['stats'].columns]
    # Show first 3 rows of the file
    df_preview = pd.read_csv(file_path, nrows=3)
    preview_data = df_preview.to_dict('records')
    # Round all float values in preview_data to two decimals
    for row in preview_data:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 2)
    preview_columns = [{'name': c, 'id': c} for c in df_preview.columns]
    return summary_html, stats_data, stats_columns, preview_data, preview_columns

@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('csv-file-dropdown', 'value'),
     Input('start-time-slider', 'value'),
     Input('weeks-slider', 'value')]
)
def update_time_series_plot(selected_file: str, start_time: int, weeks: int) -> go.Figure:
    """
    Create time series plot showing mean, min, and max values across all stations for each pollutant.
    Args:
        selected_file (str): The selected CSV filename.
        start_time (int): Start time in hours from beginning.
        weeks (int): Number of weeks to plot.
    Returns:
        go.Figure: Plotly figure with time series data.
    """
    if not selected_file:
        return go.Figure()
    
    file_path = os.path.join(pollution_folder, selected_file)
    end_time = start_time + (weeks * 7 * 24)  # Convert weeks to hours
    
    # Read the data for the specified time range
    df = pd.read_csv(file_path)
    df = df.iloc[start_time:end_time]
    
    # Get pollutant columns (excluding imputed columns)
    pollutant_cols = [col for col in df.columns if col.startswith('cont_')]
    
    # Group by pollutant type
    pollutants = {}
    for col in pollutant_cols:
        parts = col.split('_')
        if len(parts) >= 3:
            pollutant_type = parts[1]  # e.g., 'otres', 'co', 'pmdiez'
            if pollutant_type not in pollutants:
                pollutants[pollutant_type] = []
            pollutants[pollutant_type].append(col)
    
    # Calculate the maximum value of all otres columns for reference line
    otres_cols = pollutants.get('otres', [])
    otres_max_per_hour = None
    if otres_cols:
        otres_data = df[otres_cols]
        otres_max_per_hour = otres_data.max(axis=1)  # Maximum across all otres columns for each hour
    
    # Create subplots
    n_pollutants = len(pollutants)
    fig = make_subplots(
        rows=n_pollutants, cols=1,
        subplot_titles=list(pollutants.keys()),
        vertical_spacing=0.03
    )
    
    for i, (pollutant_type, cols) in enumerate(pollutants.items(), 1):
        # Calculate statistics across all stations for this pollutant
        pollutant_data = df[cols]
        
        # Calculate mean, min, max across stations for each time point
        mean_vals = pollutant_data.mean(axis=1)
        min_vals = pollutant_data.min(axis=1)
        max_vals = pollutant_data.max(axis=1)
        
        # Create time index
        time_index = pd.date_range(start=df.index[0], periods=len(df), freq='H')
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=time_index, y=mean_vals, mode='lines', name=f'{pollutant_type} Mean',
                      line=dict(color='blue'), showlegend=(i==1)),
            row=i, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_index, y=min_vals, mode='lines', name=f'{pollutant_type} Min',
                      line=dict(color='red'), showlegend=(i==1)),
            row=i, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_index, y=max_vals, mode='lines', name=f'{pollutant_type} Max',
                      line=dict(color='green'), showlegend=(i==1)),
            row=i, col=1
        )
        
        # Add otres max reference line if available
        if otres_max_per_hour is not None and pollutant_type in ['nox', 'pmdoscinco', 'nodos', 'no', 'pmco', 'sodos', 'pmdiez']:
            fig.add_trace(
                go.Scatter(
                    x=time_index, 
                    y=otres_max_per_hour, 
                    mode='lines', 
                    name='Otres Max Reference',
                    line=dict(color='darkgrey', dash='dash', width=2),
                    showlegend=(i==1)
                ),
                row=i, col=1
            )
    
    fig.update_layout(
        height=max(500, 600 * n_pollutants),
        title_text=f"Pollutant Time Series (Weeks {start_time//(7*24)} to {(start_time + weeks*7*24)//(7*24)})",
        showlegend=True
    )
    
    return fig

if __name__ == "__main__":
    app.run(debug=True,  port=8027) 