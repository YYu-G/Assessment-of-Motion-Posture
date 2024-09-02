from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# 连接数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:yy161161@127.0.0.1:3306/sport'

# 数据库连接对象
db_init = SQLAlchemy(app)

# JWT密钥、算法、有效时间
SECRET_KEY = '775ykt-8'
algorithm='HS256'
piece=1