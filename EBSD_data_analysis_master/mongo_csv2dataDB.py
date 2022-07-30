#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pymongo import MongoClient
import csv


'''
mongo一键存储.csv 文件
'''




# 创建连接MongoDB数据库函数
def connection():
    # 1:连接本地MongoDB数据库服务

    # 2:连接本地数据库(guazidata)。没有时会自动创建

    # 3:创建集合
    mongo_client = MongoClient('127.0.0.1', 27017)
    mongo_db = mongo_client['MYdatabase']
    mongo_collection = mongo_db['MYcollection_NEW']

    # mongo_collection.remove(None)

    return mongo_collection


def insertToMongoDB(set1):
    # 打开文件guazi.csv
    with open('Data.csv','r',encoding='utf-8')as csvfile:
        # 调用csv中的DictReader函数直接获取数据为字典形式
        reader=csv.DictReader(csvfile)
        # 创建一个counts计数一下 看自己一共添加了了多少条数据
        counts=0
        for each in reader:
            # 将数据中需要转换类型的数据转换类型。原本全是字符串（string）。
            each['Zn百分比']=float(each['Zn百分比'])
            each['Al百分比']=float(each['Al百分比'])
            each['Mn百分比']=float(each['Mn百分比'])
            each['Ca百分比']=float(each['Ca百分比'])
            each['Si百分比']=float(each['Si百分比'])
            each['均匀化温度/K']=str(each['均匀化温度/K'])
            each['均匀化时长/h']=str(each['均匀化时长/h'])
            each['extusion温度/K'] = str(each['extusion温度/K'])
            each['extrusion ratio'] = str(each['extrusion ratio'])
            each['extrusion speed/(mm/s)'] = str(each['extrusion speed/(mm/s)'])
            each['rolling温度/K'] = str(each['rolling温度/K'])
            each['rolling reduction/%'] = str(each['rolling reduction/%'])
            each['rolling speed/(mm/s)'] = str(each['rolling speed/(mm/s)'])
            each['annealing温度/K'] = str(each['annealing温度/K'])
            each['annealing时长/h'] = str(each['annealing时长/h'])
            each['grain size/μm'] = str(each['grain size/μm'])
            each['texture strength'] = str(each['texture strength'])
            each['LD和ED/RD之间的角度/度'] = str(each['LD和ED/RD之间的角度/度'])
            each['YS/Mpa'] = str(each['YS/Mpa'])
            each['UTS/Mpa'] = str(each['UTS/Mpa'])
            each['Elongation/%'] = str(each['Elongation/%'])
            each['来源(DOI)'] = str(each['来源(DOI)'])

            set1.insert_one(each)
            counts+=1
            print('成功添加了'+str(counts)+'条数据 ')
# 创建主函数
def main():
    set1=connection()
    insertToMongoDB(set1)

# 判断是不是调用的main函数。这样以后调用的时候就可以防止不会多次调用 或者函数调用错误
if __name__=='__main__':
    main()




