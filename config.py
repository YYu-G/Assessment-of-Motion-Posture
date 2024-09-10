from flask import Flask
from flask_jwt_extended import JWTManager
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_socketio import SocketIO

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# LSTM parameters
INPUT_DIM = 34  # 17 keypoints (x, y)
HIDDEN_DIM = 8  # hidden layers
NUM_LAYERS = 2
OUTPUT_DIM = 5  # sit-up; pushup; squat;pullup

app = Flask(__name__)
CORS(app)

# 连接数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:yy161161@127.0.0.1:3306/sport'

# 数据库连接对象
db_init = SQLAlchemy(app)

# JWT密钥、算法、有效时间
SECRET_KEY = '775ykt-8'
algorithm='HS256'
piece=1
app.config['JWT_SECRET_KEY']=SECRET_KEY
jwt=JWTManager(app)

#文件相对路径
rep_file_path='reportFile'
model_file_path='modelFile'
temp_path='temp'

# 创建一个 SocketIO 对象，并将其与 Flask 应用绑定
socketio = SocketIO(app, cors_allowed_origins='*')