from models.reportData import ReportData
from config import db_init as db
from sqlalchemy import or_

class report_dao:
    def add_report(ownerID, type, des, dataFileURL, repDate):
        new_report = ReportData(ownerID=ownerID, sptType=type, des=des, dataFileURL=dataFileURL, repDate=repDate)
        db.session.add(new_report)
        db.session.commit()
        return new_report.repID

    def delete_report(report):
        db.session.delete(report)

    def search_report_by_id(id):
        return ReportData.query.filter_by(repID=id).first()

    def search_report_by_owner_id(id):
        return ReportData.query.filter_by(ownerID=id).all()

    def search_report_by_type(type):
        return ReportData.query.filter_by(sptType=type).all()

    def search_report_by_date(min_date, max_date):
        return ReportData.query.filter(ReportData.repDate >= min_date, ReportData.repDate <= max_date).all()

    def search_report_all_condition(id, type1,type2,type3,min_date,max_date):
        conditions = []
        if type1==1:
            conditions.append(ReportData.sptType == 'shape')
            print('shape')
        if type2==1:
            conditions.append(ReportData.sptType == 'fitness')
        if type3==1:
            conditions.append(ReportData.sptType == 'yoga')
        if conditions:
            return ReportData.query.filter(ReportData.repDate >= min_date, ReportData.repDate <= max_date,
                                             ReportData.ownerID==id).all()
        else:
            return ReportData.query.filter(ReportData.repDate >= min_date, ReportData.repDate <= max_date,
                                             ReportData.ownerID==id).all()
        # if type1==1:
        #     query1=ReportData.query.filter(ReportData.repDate >= min_date, ReportData.repDate <= max_date,
        #                                      ReportData.ownerID==id,ReportData.sptType=='shape')
        # if type2 == 1:
        #     query2 = ReportData.query.filter(ReportData.repDate >= min_date, ReportData.repDate <= max_date,
        #                                        ReportData.ownerID==id, ReportData.sptType=='fitness')
        # if type3 == 1:
        #     query3 = ReportData.query.filter(ReportData.repDate >= min_date, ReportData.repDate <= max_date,
        #                                        ReportData.ownerID==id, ReportData.sptType=='yoga')
        #return list

    def modify_report(repID, ownerID, type, des, dataFileURL, repDate):
        rep = ReportData.query.filter_by(repID=repID).first()
        rep.ownerID = ownerID
        rep.sptType = type
        rep.des = des
        rep.dataFileURL = dataFileURL
        rep.repDate = repDate

# def add_report(ownerID,type ,des,dataFileURL,repDate):
#     new_report=ReportData(ownerID=ownerID,sptType=type,des=des,dataFileURL=dataFileURL,repDate=repDate)
#     db.session.add(new_report)
#     db.session.commit()
#     return new_report.repID
#
# def delete_report(report):
#     db.session.delete(report)
#
# def search_report_by_id(id):
#     return ReportData.query.filter_by(repID=id).first()
#
# def search_report_by_owner_id(id):
#     return ReportData.query.filter_by(ownerID=id).all()
#
# def search_report_by_type(type):
#     return ReportData.query.filter_by(sptType=type).all()
#
# def search_report_by_date(min_date,max_date):
#     return ReportData.query.filter_by(ReportData.repDate>=min_date,ReportData.repDate<=max_date).all()
#
# def modify_report(repID,ownerID,type ,des,dataFileURL,repDate):
#     rep=search_report_by_id(repID)
#     rep.ownerID=ownerID
#     rep.sptType=type
#     rep.des=des
#     rep.dataFileURL=dataFileURL
#     rep.repDate=repDate