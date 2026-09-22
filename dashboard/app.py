"""Plotly Dash dashboard for Mirage Adaptive IDS."""

import hmac
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import MATCH, Input, Output, State, callback_context, dcc, html

from app.honeypot import DB_PATH
from app.model.ensemble import list_model_versions, load_model
from app.retrain.champion_challenger import ChampionChallenger
from app.retrain.drift import DriftMonitor
from app.retrain.review_gate import ReviewGate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DASHBOARD_DB = DATA_DIR / "dashboard.db"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Mirage Adaptive IDS",
)

server = app.server

secret_key = os.environ.get("MIRAGE_SECRET_KEY")
if not secret_key:
    logger.warning("MIRAGE_SECRET_KEY not set - dashboard admin actions are disabled")
    secret_key = "dev-only-insecure"
server.secret_key = secret_key

REFRESH_INTERVAL = 5000


def check_admin_key(provided: str) -> bool:
    """Check a dashboard admin password against ``MIRAGE_ADMIN_KEY``.

    Fails closed: with no key configured, nothing unlocks.

    Args:
        provided: Password supplied by the operator.

    Returns:
        True on a match.
    """
    expected = os.environ.get("MIRAGE_ADMIN_KEY")
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided, expected)


def is_admin() -> bool:
    """Return whether the current session passed the admin gate."""
    from flask import session

    return bool(session.get("mirage_admin"))


def get_recent_attacks(limit: int = 50) -> list[dict]:
    """Get recent honeypot attacks from database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT * FROM requests 
        ORDER BY timestamp DESC 
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    attacks = []
    for row in rows:
        attacks.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "source_ip": row["source_ip"],
                "method": row["method"],
                "path": row["path"],
                "attack_type": row["attack_type"],
                "decoy_indicator": row["decoy_indicator"],
            }
        )
    return attacks


def get_confidence_history(hours: int = 24) -> pd.DataFrame:
    """Get model confidence history from logged ``/api/score`` events.

    Args:
        hours: Lookback window in hours.

    Returns:
        DataFrame with timestamp/confidence/prediction/model_version columns
        (empty when nothing has been logged yet).
    """
    cutoff = time.time() - hours * 3600
    try:
        conn = sqlite3.connect(str(DASHBOARD_DB))
        rows = conn.execute(
            "SELECT timestamp, score, prediction, model_version FROM score_log"
            " WHERE timestamp >= ? ORDER BY timestamp ASC",
            (cutoff,),
        ).fetchall()
        conn.close()
    except Exception:
        logger.debug("No score log available yet")
        return pd.DataFrame(columns=["timestamp", "confidence", "prediction", "model_version"])
    if not rows:
        return pd.DataFrame(columns=["timestamp", "confidence", "prediction", "model_version"])
    df = pd.DataFrame(rows, columns=["timestamp", "score", "prediction", "model_version"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["confidence"] = df["score"].clip(0.0, 1.0).apply(lambda s: max(s, 1 - s))
    return df[["timestamp", "confidence", "prediction", "model_version"]]


def get_drift_history() -> list[dict]:
    """Get drift check history."""
    monitor = DriftMonitor()
    return [d.to_dict() for d in monitor.get_drift_history(50)]


def get_review_queue() -> list[dict]:
    """Get pending reviews."""
    gate = ReviewGate()
    return [r.to_dict() for r in gate.get_pending_reviews()]


def get_validation_history() -> list[dict]:
    """Get champion/challenger validation history."""
    validator = ChampionChallenger()
    return [v.to_dict() for v in validator.get_validation_history(20)]


def get_model_versions() -> list[dict]:
    """Get all model versions."""
    return [v.to_dict() for v in list_model_versions()]


def get_attack_type_breakdown(hours: int = 24) -> pd.DataFrame:
    """Get attack type breakdown over time."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cutoff = time.time() - (hours * 3600)
    cursor = conn.execute(
        """
        SELECT attack_type, COUNT(*) as count,
               strftime('%Y-%m-%d %H:00:00', datetime(timestamp, 'unixepoch')) as hour
        FROM requests 
        WHERE timestamp > ?
        GROUP BY attack_type, hour
        ORDER BY hour
        """,
        (cutoff,),
    )
    rows = cursor.fetchall()
    conn.close()

    return pd.DataFrame([dict(r) for r in rows])


app.layout = dbc.Container(
    [
        dcc.Interval(id="refresh-interval", interval=REFRESH_INTERVAL, n_intervals=0),
        dcc.Store(id="data-store"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Mirage Adaptive IDS", className="text-center mb-4"),
                        html.Hr(),
                    ],
                    width=12,
                ),
            ]
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="Live Attack Feed", tab_id="tab-live"),
                dbc.Tab(label="Model Confidence", tab_id="tab-confidence"),
                dbc.Tab(label="Feature Drift", tab_id="tab-drift"),
                dbc.Tab(label="Retraining History", tab_id="tab-retrain"),
                dbc.Tab(label="Attack Breakdown", tab_id="tab-breakdown"),
                dbc.Tab(label="Admin", tab_id="tab-admin"),
            ],
            id="tabs",
            active_tab="tab-live",
        ),
        html.Div(id="tab-content", className="mt-4"),
    ],
    fluid=True,
)


@app.callback(
    Output("tab-content", "children"), [Input("tabs", "active_tab"), Input("refresh-interval", "n_intervals")]
)
def render_tab(active_tab: str, n_intervals: int):
    """Render the active tab content."""
    if active_tab == "tab-live":
        return render_live_feed()
    elif active_tab == "tab-confidence":
        return render_confidence()
    elif active_tab == "tab-drift":
        return render_drift()
    elif active_tab == "tab-retrain":
        return render_retrain()
    elif active_tab == "tab-breakdown":
        return render_breakdown()
    elif active_tab == "tab-admin":
        return render_admin()
    return html.Div("Select a tab")


def render_live_feed():
    """Render live attack feed panel."""
    attacks = get_recent_attacks(50)

    if not attacks:
        return dbc.Alert("No attacks recorded yet", color="info")

    rows = []
    for attack in attacks:
        ts = pd.Timestamp(attack["timestamp"], unit="s").strftime("%Y-%m-%d %H:%M:%S")
        badge_color = "danger" if attack["attack_type"] == "sqli" else "warning"
        rows.append(
            html.Tr(
                [
                    html.Td(ts),
                    html.Td(attack["source_ip"]),
                    html.Td(attack["method"]),
                    html.Td(attack["path"]),
                    html.Td(dbc.Badge(attack["attack_type"].upper(), color=badge_color)),
                    html.Td("Yes" if attack["decoy_indicator"] else "No"),
                ]
            )
        )

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Time"),
                        html.Th("Source IP"),
                        html.Th("Method"),
                        html.Th("Path"),
                        html.Th("Type"),
                        html.Th("Decoy"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-dark",
    )


def render_confidence():
    """Render model confidence trend panel."""
    df = get_confidence_history(24)
    if df.empty:
        return dbc.Alert(
            "No scored requests logged yet — confidence trend appears after /api/score traffic", color="info"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["confidence"],
            mode="lines+markers",
            name="Confidence",
            line=dict(color="#00d4aa", width=2),
            marker=dict(size=4),
        )
    )

    colors = ["#ff6b6b" if p == 1 else "#00d4aa" for p in df["prediction"]]
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["confidence"],
            mode="markers",
            name="Prediction",
            marker=dict(color=colors, size=6, symbol="circle"),
            showlegend=False,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Model Confidence Over Time",
        xaxis_title="Time",
        yaxis_title="Confidence",
        yaxis_range=[0, 1],
        height=500,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    return dcc.Graph(figure=fig, id="confidence-graph")


def render_drift():
    """Render feature drift (PSI) panel."""
    drift_data = get_drift_history()

    if not drift_data:
        return dbc.Alert("No drift checks performed yet", color="info")

    df = pd.DataFrame(drift_data)
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")

    fig = go.Figure()

    if df["psi_scores"].iloc[0]:
        features = list(df["psi_scores"].iloc[0].keys())
        for feat in features:
            psi_vals = [d["psi_scores"].get(feat, 0) for d in drift_data]
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=psi_vals,
                    mode="lines+markers",
                    name=feat,
                    line=dict(width=2),
                    marker=dict(size=4),
                )
            )

    fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="PSI Threshold (0.25)")

    fig.update_layout(
        template="plotly_dark",
        title="Feature Drift (PSI) Over Time",
        xaxis_title="Time",
        yaxis_title="PSI",
        height=500,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )

    return dcc.Graph(figure=fig, id="drift-graph")


def render_retrain():
    """Render retraining history panel."""
    reviews = get_review_queue()
    validations = get_validation_history()
    versions = get_model_versions()

    review_cards = []
    for r in reviews:
        status_color = {
            "pending": "warning",
            "approved": "success",
            "rejected": "danger",
            "auto_proceeded": "info",
            "done": "secondary",
        }.get(r["status"], "secondary")

        ts = pd.Timestamp(r["timestamp"], unit="s").strftime("%Y-%m-%d %H:%M:%S")
        card_body = [
            html.P(f"Created: {ts}"),
            html.P(f"Max PSI: {r['max_psi']:.4f}"),
            html.P(f"Samples: {r['sample_size']}"),
            html.Pre(json.dumps(r["psi_scores"], indent=2)),
        ]
        if r["status"] == "pending":
            card_body += [
                dbc.Button(
                    "Approve",
                    id={"type": "btn-approve", "index": r["id"]},
                    color="success",
                    size="sm",
                    className="me-2",
                ),
                dbc.Button("Reject", id={"type": "btn-reject", "index": r["id"]}, color="danger", size="sm"),
                html.Div(id={"type": "review-output", "index": r["id"]}, className="mt-2"),
            ]
        review_cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Strong(f"Review #{r['id']}"),
                            dbc.Badge(r["status"].replace("_", " ").title(), color=status_color, className="ms-2"),
                        ]
                    ),
                    dbc.CardBody(card_body),
                ],
                className="mb-3",
            )
        )

    val_rows = []
    for v in validations:
        ts = pd.Timestamp(v["timestamp"], unit="s").strftime("%Y-%m-%d %H:%M:%S")
        promoted_badge = (
            dbc.Badge("Promoted", color="success") if v["promoted"] else dbc.Badge("Rejected", color="danger")
        )
        val_rows.append(
            html.Tr(
                [
                    html.Td(ts),
                    html.Td(v["challenger_version"]),
                    html.Td(v["champion_version"]),
                    html.Td(f"{v['challenger_metrics'].get('f1', 0):.4f}"),
                    html.Td(f"{v['champion_metrics'].get('f1', 0):.4f}"),
                    html.Td(promoted_badge),
                    html.Td(v["reason"]),
                ]
            )
        )

    if val_rows:
        val_table = dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Time"),
                            html.Th("Challenger"),
                            html.Th("Champion"),
                            html.Th("Challenger F1"),
                            html.Th("Champion F1"),
                            html.Th("Result"),
                            html.Th("Reason"),
                        ]
                    )
                ),
                html.Tbody(val_rows),
            ],
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="table-dark",
        )
    else:
        val_table = dbc.Alert("No validations yet", color="info")

    version_cards = []
    for v in versions[:5]:
        ts = pd.Timestamp(v["timestamp"], unit="s").strftime("%Y-%m-%d %H:%M:%S")
        version_cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Strong(v["version_id"]),
                            dbc.Badge("Champion", color="success", className="ms-2") if v["is_champion"] else "",
                        ]
                    ),
                    dbc.CardBody(
                        [
                            html.P(f"Trained: {ts}"),
                            html.P(f"Samples: {v['training_samples']}"),
                            html.P(f"F1: {v['validation_metrics'].get('f1', 0):.4f}"),
                            html.P(f"AUC: {v['validation_metrics'].get('auc', 0):.4f}"),
                        ]
                    ),
                ],
                className="mb-3",
            )
        )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Review Queue"),
                            html.Div(review_cards) if review_cards else dbc.Alert("No pending reviews", color="info"),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H4("Model Versions"),
                            html.Div(version_cards) if version_cards else dbc.Alert("No model versions", color="info"),
                        ],
                        width=6,
                    ),
                ]
            ),
            html.Hr(),
            html.H4("Validation History"),
            val_table,
        ]
    )


def render_breakdown():
    """Render attack type breakdown panel."""
    df = get_attack_type_breakdown(24)

    if df.empty:
        return dbc.Alert("No attack data for breakdown", color="info")

    fig = px.bar(
        df,
        x="hour",
        y="count",
        color="attack_type",
        title="Attack Types Over Time (Last 24 Hours)",
        color_discrete_map={"sqli": "#ff6b6b", "rce": "#ffa500", "unknown": "#6c757d"},
        template="plotly_dark",
    )
    fig.update_layout(height=500, margin=dict(l=40, r=20, t=50, b=40))

    return dcc.Graph(figure=fig, id="breakdown-graph")


def render_admin():
    """Render admin panel (protected in production)."""
    try:
        model = load_model("champion")
        version = model.version.version_id if model.version else "unknown"
        metrics = model.version.validation_metrics if model.version else {}
        top_features = model.get_drift_tracking_features(8)
    except FileNotFoundError:
        version = "none"
        metrics = {}
        top_features = []

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Current Model"),
                                    dbc.CardBody(
                                        [
                                            html.H5(f"Version: {version}"),
                                            html.P(f"F1: {metrics.get('f1', 'N/A')}"),
                                            html.P(f"Precision: {metrics.get('precision', 'N/A')}"),
                                            html.P(f"Recall: {metrics.get('recall', 'N/A')}"),
                                            html.P(f"AUC: {metrics.get('auc', 'N/A')}"),
                                            html.Hr(),
                                            html.H6("Top Drift-Tracking Features:"),
                                            html.Ul([html.Li(f) for f in top_features]),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("System Actions"),
                                    dbc.CardBody(
                                        [
                                            dbc.Input(
                                                id="admin-key",
                                                type="password",
                                                placeholder="Admin key",
                                                className="mb-2",
                                            ),
                                            dbc.Button(
                                                "Unlock Admin",
                                                id="btn-unlock",
                                                color="secondary",
                                                className="me-2 mb-2",
                                            ),
                                            html.Div(id="admin-unlock-output"),
                                            dbc.Button(
                                                "Reload Model", id="btn-reload", color="primary", className="me-2"
                                            ),
                                            dbc.Button(
                                                "Trigger Drift Check",
                                                id="btn-drift-check",
                                                color="warning",
                                                className="me-2",
                                            ),
                                            dbc.Button("Create Validation Set", id="btn-create-val", color="info"),
                                            html.Div(id="admin-output", className="mt-3"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )


@app.callback(
    Output("admin-unlock-output", "children"),
    Input("btn-unlock", "n_clicks"),
    State("admin-key", "value"),
    prevent_initial_call=True,
)
def unlock_admin(n_clicks, provided):
    """Unlock admin actions for this session on a correct admin key."""
    from flask import session

    if check_admin_key(provided or ""):
        session["mirage_admin"] = True
        return dbc.Alert("Admin unlocked for this session", color="success")
    return dbc.Alert("Wrong admin key", color="danger")


@app.callback(
    Output("admin-output", "children"),
    [Input("btn-reload", "n_clicks"), Input("btn-drift-check", "n_clicks"), Input("btn-create-val", "n_clicks")],
    prevent_initial_call=True,
)
def admin_actions(reload_clicks, drift_clicks, val_clicks):
    """Handle admin button clicks (requires unlocked admin session)."""
    if not is_admin():
        return dbc.Alert("Admin locked — unlock with the admin key first", color="warning")
    ctx = callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "btn-reload":
        from app.api import reload_model

        try:
            reload_model()
            return dbc.Alert("Model reloaded successfully", color="success")
        except Exception as e:
            return dbc.Alert(f"Reload failed: {e}", color="danger")

    elif button_id == "btn-drift-check":
        try:
            monitor = DriftMonitor()
            if not monitor.load_reference():
                return dbc.Alert("No reference loaded. Train model first.", color="warning")

            import sqlite3

            from app.honeypot import DB_PATH

            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM requests ORDER BY timestamp DESC LIMIT 200")
            rows = cursor.fetchall()
            conn.close()

            if rows:
                from app.schema.capture import rows_to_features

                X = rows_to_features(rows)
                result = monitor.check_drift(X)
                return dbc.Alert(
                    f"Drift check complete. Max PSI: {result.max_psi:.4f}. " f"Triggered: {result.triggered}",
                    color="info",
                )
            return dbc.Alert("No data to check", color="warning")
        except Exception as e:
            return dbc.Alert(f"Drift check failed: {e}", color="danger")

    elif button_id == "btn-create-val":
        try:
            from app.retrain.champion_challenger import create_validation_set

            create_validation_set()
            return dbc.Alert("Validation set created", color="success")
        except Exception as e:
            return dbc.Alert(f"Failed: {e}", color="danger")

    return ""


@app.callback(
    Output({"type": "review-output", "index": MATCH}, "children"),
    [
        Input({"type": "btn-approve", "index": MATCH}, "n_clicks"),
        Input({"type": "btn-reject", "index": MATCH}, "n_clicks"),
    ],
    prevent_initial_call=True,
)
def review_actions(approve_clicks, reject_clicks):
    """Handle review approve/reject clicks (decision takes effect on next sweep)."""
    if not is_admin():
        return dbc.Alert("Admin locked — unlock with the admin key first", color="warning")
    ctx = callback_context
    if not ctx.triggered:
        return ""
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        review_id = json.loads(prop_id)["index"]
    except (ValueError, KeyError, TypeError):
        return dbc.Alert("Could not identify review", color="danger")

    gate = ReviewGate()
    if "btn-approve" in prop_id:
        ok = gate.approve(review_id, "dashboard")
        return (
            dbc.Alert(f"Review #{review_id} approved — retrains on next sweep", color="success")
            if ok
            else dbc.Alert(f"Review #{review_id} already decided", color="warning")
        )
    ok = gate.reject(review_id, "dashboard")
    return (
        dbc.Alert(f"Review #{review_id} rejected", color="info")
        if ok
        else dbc.Alert(f"Review #{review_id} already decided", color="warning")
    )


if __name__ == "__main__":
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    if debug:
        logger.warning("Running dashboard with debug=True (development only)")
    app.run_server(host="0.0.0.0", port=8050, debug=debug)
