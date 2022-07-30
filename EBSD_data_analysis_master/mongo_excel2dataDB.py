import datetime
import xlrd
import json
from pymongo import MongoClient

'''
mongo一键存储.xls 文件
'''



def excel2mongodb(name):
    # 连接数据库
    mongo_client = MongoClient('127.0.0.1', 27017)
    mongo_db = mongo_client['MYdatabase']
    mongo_collection = mongo_db['MYcollection']

    # 读取Excel文件
    data = xlrd.open_workbook(name)
    table = data.sheets()[0]

    # 读取excel第一行数据作为存入mongodb的字段名
    row_stag = table.row_values(0)
    n_rows = table.nrows
    return_data = {}

    save_size = 0
    save_collection = []
    for i in range(1, n_rows):
        # 将字段名和excel数据存储为字典形式，并转换为json格式
        return_data[i] = json.dumps(dict(zip(row_stag, table.row_values(i))))
        # 通过编解码还原数据
        return_data[i] = json.loads(return_data[i])
        # 转换表头为所需字段名
        # del return_data[i]['序号']
        # return_data[i]['Al%'] = return_data[i].pop('Al')
        # return_data[i]['Sn%'] = return_data[i].pop('Sn')
        # return_data[i]['Zn%'] = return_data[i].pop('Zn')
        # return_data[i]['Ca%'] = return_data[i].pop('Ca')
        # return_data[i]['Mn%'] = return_data[i].pop('Mn')

        save_collection.insert(save_size, return_data[i])
        if save_size >= 1000:
            mongo_collection.insert_many(save_collection)
            save_size = 0
            save_collection.clear()
        else:
            save_size += 1

    mongo_collection.insert_many(save_collection)



if __name__ == '__main__':
    excel2mongodb('dataset.xlsx')
