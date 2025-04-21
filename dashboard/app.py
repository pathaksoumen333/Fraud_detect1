import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import os
from datetime import datetime, timedelta

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "Real-Time Fraud Detection"

DATA_FILE = 'results/predictions.csv'

def load_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["TransactionID", "Amount", "Prediction", "Timestamp"])
    try:
        df = pd.read_csv(DATA_FILE)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df
    except:
        return pd.DataFrame(columns=["TransactionID", "Amount", "Prediction", "Timestamp"])

# ---------------------- Layout ---------------------- #
app.layout = dbc.Container([
    html.H1("🔍 Real-Time Fraud Detection Dashboard", className="text-center text-primary my-4"),

    html.Div(id="alert-container"),

    html.P("This dashboard simulates and displays fraud detection in banking transactions using a ML model. Filters below help you explore insights.",
           className="lead text-center mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='prediction-filter',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Fraud', 'value': 1},
                    {'label': 'Normal', 'value': 0}
                ],
                value='All',
                clearable=False,
                style={"margin-bottom": "10px"},
                placeholder="Filter by Prediction"
            ),
        ], md=4),
        dbc.Col([
            dcc.DatePickerRange(
                id='date-filter',
                display_format='YYYY-MM-DD',
                start_date_placeholder_text="Start Date",
                end_date_placeholder_text="End Date",
                style={"margin-bottom": "10px"},
            )
        ], md=6),
        dbc.Col([
            dbc.Button("⬇️ Download Report", id="download-btn", color="success", className="mb-2", n_clicks=0),
            dcc.Download(id="download-report"),
        ], md=2),
    ]),

    html.Div(id="accuracy-display", className="text-info text-center mb-3 fw-bold"),

    # Stats Panel
    dbc.Row(id="stats-panel", className="text-center my-3"),

    dbc.Row([
        dbc.Col([
            html.H5("🔍 Fraud vs Normal Transactions"),
            dcc.Graph(id='fraud-trend'),
            html.Small("Bar chart showing count of fraudulent and normal transactions.")
        ], md=6),
        dbc.Col([
            html.H5("📊 Fraud Distribution"),
            dcc.Graph(id='fraud-pie'),
            html.Small("Pie chart showing the share of fraud and normal activity.")
        ], md=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("📈 Transaction Timeline"),
            dcc.Graph(id='fraud-timeline'),
            html.Small("Line graph showing transaction amounts over time.")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("📊 Today vs Yesterday: Fraud Comparison"),
            dcc.Graph(id='trend-comparison'),
            html.Small("Compares hourly fraud detection for today and yesterday.")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("🔥 Fraud Activity Heatmap (Day vs Hour)"),
            dcc.Graph(id='fraud-heatmap'),
            html.Small("Heatmap showing fraud frequency across days and hours.")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("📋 Recent Transactions Table"),
            dash_table.DataTable(id='transactions-table',
                                 columns=[
                                     {"name": "TransactionID", "id": "TransactionID"},
                                     {"name": "Amount", "id": "Amount"},
                                     {"name": "Prediction", "id": "Prediction"},
                                     {"name": "Timestamp", "id": "Timestamp"}
                                 ],
                                 style_table={'overflowX': 'auto'},
                                 style_cell={'textAlign': 'center'},
                                 style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                                 page_size=10)
        ])
    ]),

    dcc.Interval(
        id='interval-component',
        interval=5 * 1000,
        n_intervals=0
    )
], fluid=True)

# ---------------------- Callbacks ---------------------- #

@app.callback(
    Output('fraud-trend', 'figure'),
    Output('fraud-pie', 'figure'),
    Output('fraud-timeline', 'figure'),
    Output('accuracy-display', 'children'),
    Output('transactions-table', 'data'),
    Output('alert-container', 'children'),
    Output('trend-comparison', 'figure'),
    Output('fraud-heatmap', 'figure'),
    Output('stats-panel', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('prediction-filter', 'value'),
    Input('date-filter', 'start_date'),
    Input('date-filter', 'end_date')
)
def update_dashboard(n, prediction_value, start_date, end_date):
    df = load_data()
    if df.empty:
        return go.Figure(), go.Figure(), go.Figure(), "No data yet", [], None, go.Figure(), go.Figure(), []

    # Filter
    if prediction_value != 'All':
        df = df[df["Prediction"] == prediction_value]
    if start_date:
        df = df[df["Timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Timestamp"] <= pd.to_datetime(end_date)]

    if df.empty:
        return go.Figure(), go.Figure(), go.Figure(), "No data found for filters.", [], None, go.Figure(), go.Figure(), []

    df["Label"] = df["Prediction"].map({0: 'Normal', 1: 'Fraud'})
    df["Time"] = df["Timestamp"].dt.strftime('%H:%M:%S')

    # Main Graphs
    trend = df.groupby('Label').size().reset_index(name='Count')
    fig_trend = px.bar(trend, x='Label', y='Count', color='Label',
                       color_discrete_map={'Normal': 'skyblue', 'Fraud': 'crimson'})

    fig_pie = px.pie(trend, values='Count', names='Label', color='Label',
                     color_discrete_map={'Normal': 'skyblue', 'Fraud': 'crimson'})

    fig_line = px.line(df, x='Time', y='Amount', color='Label', markers=True,
                       color_discrete_map={'Normal': 'skyblue', 'Fraud': 'crimson'})

    # Accuracy & Stats
    total = len(df)
    fraud_count = df["Prediction"].sum()
    accuracy = (1 - fraud_count / total) * 100
    acc_text = f"✅ Model Accuracy: {accuracy:.2f}% ({total - fraud_count} Normal / {fraud_count} Fraud)"

    stats_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("🔢 Total Transactions"), dbc.CardBody(f"{total}")], color="light"), md=3),
        dbc.Col(dbc.Card([dbc.CardHeader("❗ Fraud Count"), dbc.CardBody(f"{fraud_count}")], color="danger", inverse=True), md=3),
        dbc.Col(dbc.Card([dbc.CardHeader("✅ Accuracy"), dbc.CardBody(f"{accuracy:.2f}%")], color="info", inverse=True), md=3),
        dbc.Col(dbc.Card([dbc.CardHeader("📊 Fraud %"), dbc.CardBody(f"{(fraud_count / total * 100):.2f}%")], color="warning", inverse=True), md=3),
    ])

    # Table
    table_data = df.sort_values(by='Timestamp', ascending=False).to_dict('records')

    # Alert
    alert_box = None
    if fraud_count / total >= 0.3:
        alert_box = dbc.Alert("🚨 High fraud activity detected!", color="danger", className="text-center")

    # Today vs Yesterday Trend
    today = df['Timestamp'].max().date()
    yesterday = today - timedelta(days=1)

    today_data = df[df["Timestamp"].dt.date == today]
    yesterday_data = df[df["Timestamp"].dt.date == yesterday]

    today_fraud = today_data[today_data['Prediction'] == 1].resample('H', on='Timestamp').size()
    yesterday_fraud = yesterday_data[yesterday_data['Prediction'] == 1].resample('H', on='Timestamp').size()

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=today_fraud.index, y=today_fraud.values, mode='lines+markers', name="Today"))
    fig_compare.add_trace(go.Scatter(x=yesterday_fraud.index, y=yesterday_fraud.values, mode='lines+markers', name="Yesterday"))
    fig_compare.update_layout(title="Today vs Yesterday - Hourly Fraud Comparison", xaxis_title="Time", yaxis_title="Fraud Count")

    # Heatmap
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.date
    heatmap_data = df[df['Prediction'] == 1].groupby(['Day', 'Hour']).size().unstack(fill_value=0)
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index.astype(str),
        colorscale='YlOrRd',
        colorbar=dict(title="Fraud Count")
    ))
    fig_heatmap.update_layout(title="Fraud Activity Heatmap", xaxis_title="Hour of Day", yaxis_title="Date")

    return fig_trend, fig_pie, fig_line, acc_text, table_data, alert_box, fig_compare, fig_heatmap, stats_cards

# ---------------------- Download ---------------------- #
@app.callback(
    Output("download-report", "data"),
    Input("download-btn", "n_clicks"),
    State('prediction-filter', 'value'),
    State('date-filter', 'start_date'),
    State('date-filter', 'end_date'),
    prevent_initial_call=True
)
def download_report(n_clicks, prediction_value, start_date, end_date):
    df = load_data()
    if prediction_value != 'All':
        df = df[df["Prediction"] == prediction_value]
    if start_date:
        df = df[df["Timestamp"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Timestamp"] <= pd.to_datetime(end_date)]

    file_name = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(file_name, index=False)
    return dcc.send_file(file_name)

# ---------------------- Run ---------------------- #
if __name__ == '__main__':
    app.run(debug=True)
