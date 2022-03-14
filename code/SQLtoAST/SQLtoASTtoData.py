from pglast import parse_sql
import pandas as pd
import numpy as np
import xlrd
import pickle
import random

# if key is '@', create dict
def createDict(name):
    tempDict = {}
    tempDict[u'name'] = name
    tempDict[u'clen'] = 0
    tempDict[u'childcase'] = 0
    tempDict[u'children'] = []
    return tempDict

# recursive traversal dict
def cycleDict(dictData, parentDict, currentDict):
    for key,value in dictData.items():
        tempDict = None
        if key is '@':
            if listStatementsMeta.get(value) is None:
                listStatementsMeta[value] = len(listStatementsMeta)
            tempDict = createDict(listStatementsMeta[value])

            if parentDict is None:
                parentDict = tempDict
                currentDict = parentDict
                continue
            
            currentDict = tempDict
            parentDict[u'clen'] += 1
            if(parentDict[u'clen'] <= 2):
                parentDict[u'childcase'] += 1
            parentDict[u'children'].append(currentDict)
                
        elif type(value) is tuple:
            cycleTuple(value, currentDict,currentDict)
                
        elif type(value) is dict:
            cycleDict(value, currentDict, currentDict)
    return parentDict

#  recursive traversal tuple
def cycleTuple(tupleData, parentDict, currentDict):
    for item in tupleData:
        if(item is None):
            return parentDict
        for key, value in item.items():
            tempDict = None
            if key is '@':
                if listStatementsMeta.get(value) is None:
                    listStatementsMeta[value] = len(listStatementsMeta)
                tempDict = createDict(listStatementsMeta[value])

                currentDict = tempDict
                parentDict[u'clen'] += 1
                if(parentDict[u'clen'] <= 2):
                    parentDict[u'childcase'] += 1
                parentDict[u'children'].append(currentDict)
                    
            elif type(value) is tuple:
                cycleTuple(value, currentDict, currentDict)
            
            elif type(value) is dict:
                cycleDict(value, currentDict, currentDict)  
            
def _get_filename(basename, split):
    return '{}.{}.obj'.format(basename, split)

# .meta.obj file
def write_to_file_total(word2int, nodes, basename, train_frac=0.6, val_frac=0.2):  # word2int is type dict, and node is type list
    def split(items, frac1, frac2):
        p1 = int(len(items) * frac1)
        p2 = int(len(items) * (frac1 + frac2))
        return items[:p1], items[p1:p2], items[p2:]

    def write_to_file(samples, split):
        filepath = _get_filename(basename, split)
        with open(filepath, 'wb') as f:
            pickle.dump(len(samples), f, protocol=2)
            for sample in samples:
                pickle.dump(sample, f, protocol=2)

    random.shuffle(nodes)
    samples_train, samples_val, samples_test = split(nodes, train_frac, val_frac)  # sample_train is type list
    
    random.shuffle(samples_train)
    random.shuffle(samples_val)
    random.shuffle(samples_test)
    
    with open(_get_filename(basename, 'meta'), 'wb') as f:
        pickle.dump(word2int, f, protocol=2)
        pickle.dump(['train', 'val', 'test'], f, protocol=2)
    write_to_file(samples_train, 'train')  # samples_train is type list
    write_to_file(samples_val, 'val')
    write_to_file(samples_test, 'test')


if __name__ == '__main__':
    # define the final data
    listStatementsMeta = {}
    
    # read SQL data from .xlsx
    df = pd.read_excel("data/total_question.xlsx", usecols=["答案"])  # df is type DataFrame
    df = df.replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n', ' ', regex=True)
    train_data = np.array(df)  # np.ndarray()
    sql_data = train_data.tolist()  # list
    
    # read difficulty from .xlsx
    difficulty = pd.read_excel("data/total_question.xlsx",usecols=["答错率"])  
    difficulty = np.array(difficulty)  # difficulty.dtype is float64
    difficulty = difficulty.tolist()          
    
    list_nodes = []
    word2int = {}
    j = 0
    
    for i in sql_data:
        diffi = 0.
        listStatementsData = []
        # convert SQL data to dict
        parse_data = parse_sql("".join(i))[0].stmt  # paser_data is type dict
        print(parse_data)
        parentDict = None
        currentDict = None
        parentDict = cycleDict(parse_data(), parentDict, currentDict)
        listStatementsData.append(parentDict)
        listStatementsData.append(difficulty[j][0])  # difficulty
        listStatementsData = tuple(listStatementsData)  # tuple type
       
        list_nodes.append(listStatementsData)  
        j = j + 1

    # save
    basename = 'data/statements'
    write_to_file_total(listStatementsMeta, list_nodes, basename, train_frac=0.6, val_frac=0.2)

