import os.path

from flask import send_file, send_from_directory
from setuptools.sandbox import save_path

from routers.start import start
from routers.user import user
from routers.report import report
from routers.mangager import manager
from routers.history import history
from config import app,socketio,rep_file_path


app.register_blueprint(start, url_prefix="/start")
app.register_blueprint(user, url_prefix="/user")
app.register_blueprint(report, url_prefix="/report")
app.register_blueprint(manager, url_prefix="/manager")
app.register_blueprint(history, url_prefix="/history")

@app.route('/api/download/<filename>')
def download_file(filename):
    return send_from_directory('reportFile',filename)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    socketio.run(app, host='0.0.0.0',port=5000,debug=True)
    print('ab')