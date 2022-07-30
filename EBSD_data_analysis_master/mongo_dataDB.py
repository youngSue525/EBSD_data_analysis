from pymongo import MongoClient
import bson.binary
import datetime


mongo_client = MongoClient('127.0.0.1', 27017)
mongo_db = mongo_client['MYdatabase']
mongo_collection = mongo_db['MYcollection']

with open('11.png', 'rb') as f:
    content = bson.binary.Binary(f.read())

    info = {
                'filename': f.name,
                'Chemical formula' : 'Mg',
                'Space group' : 'P 63/m m c',
                'Lattice params' : [3.209, 3.209, 5.211, 90 , 90, 120],
                'Grain size(μm)' : '50',
                'Sigma' : '0.11',
                'Eular params': [[260.00669169, 166.03698512, 64.62214586],
                                 [334.12484994,   9.19087635,  67.33733493],
                                 [ 21.78112698,  75.53885952, 311.24417951],
                                 [1301, 1113, 523]],
                'data': content,
                'Creation data' : '1980-01-01',
                'Recoed time' :  datetime.datetime.now(),
                'Detail': '',
             }

    mongo_collection.insert_one(info)

    # mongo_collection.delete_one({'Chemical formula': 'Mg'})