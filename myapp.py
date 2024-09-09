from routers.start import start
from routers.user import user
from routers.main import main
from routers.report import report
from routers.mangager import manager
from routers.history import history
from config import app,socketio


app.register_blueprint(start, url_prefix="/start")
app.register_blueprint(user, url_prefix="/user")
app.register_blueprint(main, url_prefix="/main")
app.register_blueprint(report, url_prefix="/report")
app.register_blueprint(manager, url_prefix="/manager")
app.register_blueprint(history, url_prefix="/history")

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    socketio.run(app, host='0.0.0.0',port=5000,debug=True)
    print('ab')