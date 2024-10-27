from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from .data_processor import DataProcessor
import os
import signal
import threading
import logging

app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))
socketio = SocketIO(app, cors_allowed_origins="*")
data_processor = DataProcessor()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    logger.debug("Rendering dashboard.html")
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    logger.debug("Fetching latest metrics")
    return jsonify(data_processor.get_latest_metrics())

@app.route('/api/control/<action>')
def control_training(action):
    logger.debug(f"Received control action: {action}")
    return jsonify(data_processor.control_training(action))

@socketio.on('connect')
def handle_connect():
    logger.debug("Client connected")
    emit('connected', {'status': 'Connected to server'})

@socketio.on('request_data')
def send_data():
    logger.debug("Received data request from client")
    metrics = data_processor.get_latest_metrics()
    emit('update_metrics', metrics)

def start_dashboard(host='0.0.0.0', port=5002, debug=False):
    """
    Start the dashboard server.
    Modified to handle threading properly.
    """
    logger.info(f"Starting dashboard on {host}:{port}")
    if threading.current_thread() is threading.main_thread():
        # If in main thread, run directly
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    else:
        # If in a separate thread, run without signal handling
        socketio.run(app, host=host, port=port, debug=debug, 
                    allow_unsafe_werkzeug=True, use_reloader=False)

if __name__ == '__main__':
    start_dashboard()
