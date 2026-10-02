"""Kronos WebUI — Flask app factory."""

import os
import logging
import warnings

from flask import Flask
from flask_cors import CORS

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

from .config import API_KEY
from .routes import register_routes


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Security: deny-all CORS by default; require explicit KRONOS_ALLOWED_ORIGINS env var
    if os.environ.get('KRONOS_ALLOWED_ORIGINS'):
        _origins = [o.strip() for o in os.environ['KRONOS_ALLOWED_ORIGINS'].split(',') if o.strip()]
        CORS(app, origins=_origins)
    else:
        CORS(app, origins=[])
        logger.warning("KRONOS_ALLOWED_ORIGINS not set — CORS is deny-all. "
                       "Set KRONOS_ALLOWED_ORIGINS in production.")

    # Register all routes
    register_routes(app)

    return app


# Legacy: allow `from app import app` for backwards compat with run.py
app = create_app()

# Re-export for backwards compatibility with tests
from .services import (  # noqa: E402, F401
    load_data_files, load_data_file,
    save_prediction_results, create_prediction_chart,
    MODEL_AVAILABLE, AVAILABLE_MODELS,
)
from .config import API_KEY  # noqa: E402, F401


def get_server_config() -> tuple[bool, str]:
    """Return (debug, host) for the dev server.

    Security: the Werkzeug debugger is an RCE vector when reachable, so debug
    mode is opt-in via KRONOS_WEBUI_DEBUG=1 only. The default bind is
    loopback; override with KRONOS_WEBUI_HOST when LAN access is deliberate.
    """
    debug = os.environ.get('KRONOS_WEBUI_DEBUG', '') == '1'
    host = os.environ.get('KRONOS_WEBUI_HOST', '127.0.0.1')
    return debug, host


if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    from .services import MODEL_AVAILABLE
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")

    _debug, _host = get_server_config()
    app.run(debug=_debug, host=_host, port=7070)
