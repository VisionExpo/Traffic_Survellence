#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web dashboard for traffic surveillance.
"""

import os
import base64
import logging
import threading
import time
import cv2
import numpy as np
from datetime import datetime
from io import BytesIO

# Import Dash components
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class DashboardThread(threading.Thread):
    """Thread class for running the dashboard."""
    
    def __init__(self, video_processor, db_manager, host="0.0.0.0", port=8050, theme="dark"):
        """
        Initialize the dashboard thread.
        
        Args:
            video_processor: Video processor instance
            db_manager: Database manager instance
            host (str): Host to run the dashboard on
            port (int): Port to run the dashboard on
            theme (str): Dashboard theme (light, dark)
        """
        threading.Thread.__init__(self)
        self.daemon = True
        self.video_processor = video_processor
        self.db_manager = db_manager
        self.host = host
        self.port = port
        self.theme = theme
        self.app = None
    
    def run(self):
        """Run the dashboard."""
        self.app = create_dashboard_app(self.video_processor, self.db_manager, self.theme)
        self.app.run_server(host=self.host, port=self.port, debug=False)


def create_dashboard_app(video_processor, db_manager, theme="dark"):
    """
    Create the Dash app for the dashboard.
    
    Args:
        video_processor: Video processor instance
        db_manager: Database manager instance
        theme (str): Dashboard theme (light, dark)
        
    Returns:
        dash.Dash: Dash app
    """
    # Initialize Dash app
    theme_stylesheet = dbc.themes.DARKLY if theme == "dark" else dbc.themes.BOOTSTRAP
    app = dash.Dash(
        __name__,
        external_stylesheets=[theme_stylesheet, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True
    )
    
    # Define app layout
    app.layout = html.Div([
        dbc.Navbar(
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.I(className="fas fa-video me-2"), width="auto"),
                        dbc.Col(dbc.NavbarBrand("Traffic Surveillance Dashboard", className="ms-2")),
                    ], align="center"),
                    href="#",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Live Feed", href="#live-feed")),
                        dbc.NavItem(dbc.NavLink("Violations", href="#violations")),
                        dbc.NavItem(dbc.NavLink("Statistics", href="#statistics")),
                    ], className="ms-auto", navbar=True),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]),
            color="primary",
            dark=True,
        ),
        
        dbc.Container([
            # Live Feed Section
            html.Div([
                html.H2("Live Feed", className="mt-4 mb-3", id="live-feed"),
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Img(id="live-video", style={"width": "100%"}),
                            dcc.Interval(id="interval-update", interval=100, n_intervals=0),
                        ]),
                    ]),
                ]),
            ]),
            
            # Violations Section
            html.Div([
                html.H2("Recent Violations", className="mt-4 mb-3", id="violations"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Violation Type"),
                                        dcc.Dropdown(
                                            id="violation-type-dropdown",
                                            options=[
                                                {"label": "All", "value": "all"},
                                                {"label": "Speeding", "value": "speeding"},
                                                {"label": "No Helmet", "value": "no_helmet"},
                                            ],
                                            value="all",
                                            clearable=False,
                                        ),
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Button("Refresh", id="refresh-violations", color="primary"),
                                    ], width=2, className="d-flex align-items-end"),
                                ], className="mb-3"),
                                html.Div(id="violations-table-container"),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
            
            # Statistics Section
            html.Div([
                html.H2("Statistics", className="mt-4 mb-3", id="statistics"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Violations by Type"),
                            dbc.CardBody([
                                dcc.Graph(id="violations-by-type-chart"),
                            ]),
                        ]),
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Violations by Hour"),
                            dbc.CardBody([
                                dcc.Graph(id="violations-by-hour-chart"),
                            ]),
                        ]),
                    ], width=6),
                ]),
            ]),
            
            # Footer
            html.Footer([
                html.P("Traffic Surveillance System © 2025", className="text-center mt-4 mb-2"),
            ]),
        ], className="my-3"),
    ])
    
    # Define callbacks
    @app.callback(
        Output("live-video", "src"),
        Input("interval-update", "n_intervals")
    )
    def update_live_feed(n):
        """Update the live feed image."""
        if video_processor is None:
            return ""
        
        # Get the current frame
        frame = video_processor.get_frame()
        if frame is None:
            return ""
        
        # Convert to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        
        # Convert to base64
        encoded_image = base64.b64encode(buffer).decode("utf-8")
        
        return f"data:image/jpeg;base64,{encoded_image}"
    
    @app.callback(
        Output("violations-table-container", "children"),
        [Input("refresh-violations", "n_clicks"),
         Input("violation-type-dropdown", "value")]
    )
    def update_violations_table(n_clicks, violation_type):
        """Update the violations table."""
        if db_manager is None:
            return html.Div("Database not available")
        
        # Get violations from database
        violations = db_manager.get_violations(
            limit=10,
            violation_type=None if violation_type == "all" else violation_type
        )
        
        if not violations:
            return html.Div("No violations found")
        
        # Create table data
        table_data = []
        for violation in violations:
            # Format timestamp
            timestamp = datetime.fromtimestamp(violation["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            
            # Create table row
            row = {
                "ID": violation["id"],
                "Type": violation["violation_type"].replace("_", " ").title(),
                "Time": timestamp,
                "License Plate": violation["license_plate"] or "N/A",
                "Speed": f"{violation['speed']:.1f} km/h" if violation["speed"] else "N/A",
                "Image": html.Img(
                    src=violation.get("image", ""),
                    style={"height": "60px"}
                ) if "image" in violation else "N/A"
            }
            
            table_data.append(row)
        
        # Create table
        table = dash_table.DataTable(
            id="violations-table",
            columns=[
                {"name": "ID", "id": "ID"},
                {"name": "Type", "id": "Type"},
                {"name": "Time", "id": "Time"},
                {"name": "License Plate", "id": "License Plate"},
                {"name": "Speed", "id": "Speed"},
                {"name": "Image", "id": "Image", "presentation": "markdown"},
            ],
            data=table_data,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "padding": "10px",
                "backgroundColor": "#303030" if theme == "dark" else "white",
                "color": "white" if theme == "dark" else "black",
            },
            style_header={
                "backgroundColor": "#404040" if theme == "dark" else "#f8f9fa",
                "fontWeight": "bold",
                "color": "white" if theme == "dark" else "black",
            },
            style_data_conditional=[
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#383838" if theme == "dark" else "#f1f1f1",
                }
            ],
            page_size=10,
        )
        
        return table
    
    @app.callback(
        [Output("violations-by-type-chart", "figure"),
         Output("violations-by-hour-chart", "figure")],
        Input("refresh-violations", "n_clicks")
    )
    def update_statistics_charts(n_clicks):
        """Update the statistics charts."""
        if db_manager is None:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Database not available",
                template="plotly_dark" if theme == "dark" else "plotly"
            )
            return empty_fig, empty_fig
        
        # Get violation statistics
        stats = db_manager.get_violation_stats()
        
        # Create violations by type chart
        by_type_data = []
        for violation_type, count in stats["by_type"].items():
            by_type_data.append({
                "Violation Type": violation_type.replace("_", " ").title(),
                "Count": count
            })
        
        by_type_fig = px.bar(
            by_type_data,
            x="Violation Type",
            y="Count",
            color="Violation Type",
            template="plotly_dark" if theme == "dark" else "plotly"
        )
        
        by_type_fig.update_layout(
            title="Violations by Type",
            xaxis_title="Violation Type",
            yaxis_title="Count",
            showlegend=False
        )
        
        # Create violations by hour chart
        by_hour_data = []
        for hour, count in stats["by_hour"].items():
            by_hour_data.append({
                "Hour": int(hour),
                "Count": count
            })
        
        # Sort by hour
        by_hour_data.sort(key=lambda x: x["Hour"])
        
        by_hour_fig = px.line(
            by_hour_data,
            x="Hour",
            y="Count",
            markers=True,
            template="plotly_dark" if theme == "dark" else "plotly"
        )
        
        by_hour_fig.update_layout(
            title="Violations by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Count",
            xaxis=dict(tickmode="linear", tick0=0, dtick=1)
        )
        
        return by_type_fig, by_hour_fig
    
    return app


def create_dashboard(video_processor, db_manager, host="0.0.0.0", port=8050, theme="dark"):
    """
    Create and start the dashboard.
    
    Args:
        video_processor: Video processor instance
        db_manager: Database manager instance
        host (str): Host to run the dashboard on
        port (int): Port to run the dashboard on
        theme (str): Dashboard theme (light, dark)
    """
    dashboard_thread = DashboardThread(
        video_processor=video_processor,
        db_manager=db_manager,
        host=host,
        port=port,
        theme=theme
    )
    dashboard_thread.start()
    
    logger.info(f"Dashboard started at http://{host}:{port}")
    
    # Start video processing if not already running
    if video_processor and not video_processor.processing:
        video_thread = threading.Thread(target=video_processor.process)
        video_thread.daemon = True
        video_thread.start()
        
        logger.info("Video processing started")
    
    return dashboard_thread
