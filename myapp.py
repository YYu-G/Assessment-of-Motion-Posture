from flask import Flask,request
from routers.start import start
from config import app
from flask_cors import CORS

app.register_blueprint(start, url_prefix="/start")
CORS(app)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)