from config import db_init as db

class Movement(db.Model):
    __tablename__ ='movement'
    mID=db.Column(db.Integer, primary_key=True, autoincrement=True)
    mName=db.Column(db.String(50), nullable=False)
    sportType=db.Column(db.String(50), nullable=False)