# dashboard/src/app.py

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from data_processor import DataProcessor  # Changed this line
import os
import signal
import threading

app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))
socketio = SocketIO(app, cors_allowed_origins="*")
data_processor = DataProcessor()

@app.route('/')
def index():
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'Connected to server'})

@socketio.on('request_data')
def send_data():
    metrics = data_processor.get_latest_metrics()
    emit('update_metrics', metrics)

def start_dashboard(host='0.0.0.0', port=5002, debug=False):
    """
    Start the dashboard server.
    Modified to handle threading properly.
    """
    if threading.current_thread() is threading.main_thread():
        # If in main thread, run directly
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    else:
        # If in a separate thread, run without signal handling
        socketio.run(app, host=host, port=port, debug=debug, 
                    allow_unsafe_werkzeug=True, use_reloader=False)

if __name__ == '__main__':
    start_dashboard()