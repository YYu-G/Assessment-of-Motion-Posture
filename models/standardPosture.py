from config import db_init as db

class StandardPosture(db.Model):
    __tablename__ ='standardPosture'
    postureID=db.Column(db.Integer, primary_key=True, autoincrement=True)
    repID=db.Column(db.Integer, nullable=False)
    mID=db.Column(db.Integer, nullable=False)