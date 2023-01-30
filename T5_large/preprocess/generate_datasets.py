import re
import ast
import json
import copy
import numpy as np
import pandas as pd

#make datasets
def clean_dataset(origin_src, origin_tgt, cleaned_src, cleaned_tgt):
    xy_train = pd.read_table(origin_src, header = None)
    xy_train_target = pd.read_table(origin_tgt, header = None)
    
    xy_train_both = pd.concat((xy_train, xy_train_target), axis=1)
    xy_train_both_new = xy_train_both.drop_duplicates()
    xy_train_both_new = xy_train_both_new.reset_index()
    del xy_train_both_new['index']
    xy_train_both_new.columns=[1,2]
    
    xy_train_src = np.split(xy_train_both_new, [1], axis=1)[0]
    xy_train_tgt = np.split(xy_train_both_new, [1], axis=1)[1]
    
    with open(cleaned_src, 'a') as f1:
        for i in range(len(xy_train_src)):
            f1.write(xy_train_src[1][i] + '\n')
        
    with open(cleaned_tgt, 'a') as f2:
        for i in range(len(xy_train_tgt)):
            f2.write(xy_train_tgt[2][i] + '\n')
            
def generate_dataset_triple(pair_src, pair_tgt, l, remove = False):
    wf_src = open(pair_src, 'a')
    wf_tgt = open(pair_tgt, 'a')
    
    tag = 1
    tag2 = 1
    
    for i in range(len(l)):
        replace_token = ['<X>']
        y = '<X> '
        x = copy.deepcopy(l)
        
        if len(l) != 1:
            if x[i][-1] == 1 and tag != 0:
                tag -= 1
                y += str(x[i][:-1])+' <Z>'
                x[i] = replace_token + x[i][-1:]
                x_ = x[:]
                if remove:
                    # remove triples without connections
                    for j in x_:
                        if '<X>' not in j and 'S| '+l[i][0][3:] not in j and 'O| '+l[i][0][3:] not in j and 'S| '+l[i][2][3:] not in j and 'O| '+l[i][2][3:] not in j:
                            x.remove(j)
                wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
                wf_tgt.write(y + '\n')
            elif x[i][-1] == 2 and tag2 != 0:
                tag2 -= 1
                y += str(x[i][:-1])+' <Z>'
                x[i] = replace_token + x[i][-1:]
                x_ = x[:]
                if remove:
                    # remove triples without connections
                    for j in x_:
                        if '<X>' not in j and 'S| '+l[i][0][3:] not in j and 'O| '+l[i][0][3:] not in j and 'S| '+l[i][2][3:] not in j and 'O| '+l[i][2][3:] not in j:
                            x.remove(j)
                wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
                wf_tgt.write(y + '\n')        
            elif x[i][-1] == 3:
                y += str(x[i][:-1])+' <Z>'
                x[i] = replace_token + x[i][-1:]
                x_ = x[:]
                if remove:
                    # remove triples without connections
                    for j in x_:
                        if '<X>' not in j and 'S| '+l[i][0][3:] not in j and 'O| '+l[i][0][3:] not in j and 'S| '+l[i][2][3:] not in j and 'O| '+l[i][2][3:] not in j:
                            x.remove(j)
                wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
                wf_tgt.write(y + '\n')        
            elif x[i][-1] == 4:
                y += str(x[i][:-1])+' <Z>'
                x[i] = replace_token + x[i][-1:]
                x_ = x[:]
                if remove:
                    # remove triples without connections
                    for j in x_:
                        if '<X>' not in j and 'S| '+l[i][0][3:] not in j and 'O| '+l[i][0][3:] not in j and 'S| '+l[i][2][3:] not in j and 'O| '+l[i][2][3:] not in j:
                            x.remove(j)
                wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
                wf_tgt.write(y + '\n')        

    wf_tgt.close()
    wf_src.close()


def generate_dataset_relation(pair_src, pair_tgt, l):
    wf_src = open(pair_src, 'a')
    wf_tgt = open(pair_tgt, 'a')

    for i in range(len(l)):
        replace_token2 = '<Y>'
        x = copy.deepcopy(l)
        x[i][1] = replace_token2
        y = '<Y> ' + l[i][1] +' <Z>'
        wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
        wf_tgt.write(y + '\n')

def generate_dataset_triple_and_relation(pair_src, pair_tgt, l):
    wf_src = open(pair_src, 'a')
    wf_tgt = open(pair_tgt, 'a')
    
    tag = 1
    tag2 = 1
    
    for i in range(len(l)):
        replace_token1 = ['<X>']
        replace_token2 = '<Y>'
        y = '<X> '
        x = copy.deepcopy(l)  
        
        if len(l) != 1:
            if x[i][-1] == 1 and tag !=0:
                tag -= 1
                y += str(x[i][:-1])+' <Z>'
                x[i] = replace_token1 + x[i][-1:]
                x_ = x[:]
                for j in x_:
                    if '<X>' not in j and 'S| '+l[i][0][3:] not in j and 'O| '+l[i][0][3:] not in j and 'S| '+l[i][2][3:] not in j and 'O| '+l[i][2][3:] not in j:
                        y = y[:-3] + '<Y> ' + x[x.index(j)][1] +'<Z>'
                        x[x.index(j)][1] = replace_token2
                        break
                wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
                wf_tgt.write(y + '\n')

            elif x[i][-1] == 2 and tag2 !=0:
                tag2 -= 1
                y += str(x[i][:-1])+' <Z>'
                x[i] = replace_token1 + x[i][-1:]
                x_ = x[:]
                for j in x_:
                    if '<X>' not in j and 'S| '+l[i][0][3:] not in j and 'O| '+l[i][0][3:] not in j and 'S| '+l[i][2][3:] not in j and 'O| '+l[i][2][3:] not in j:
                        index = x.index(j)
                        y = y[:-3] + '<Y> ' + x[index][1] +'<Z>'       
                        x[index][1] = replace_token2
                        break
                wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
                wf_tgt.write(y + '\n')

            elif x[i][-1] == 3:
                y += str(x[i][:-1])+' <Z>'
                x[i] = replace_token1 + x[i][-1:]
                x_ = x[:]
                for j in x_:
                    if '<X>' not in j and 'S| '+l[i][0][3:] not in j and 'O| '+l[i][0][3:] not in j and 'S| '+l[i][2][3:] not in j and 'O| '+l[i][2][3:] not in j:
                        y = y[:-3] + '<Y> ' + x[x.index(j)][1] +'<Z>'
                        x[x.index(j)][1] = replace_token2
                        break
                wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
                wf_tgt.write(y + '\n')

            elif x[i][-1] == 4:
                y += str(x[i][:-1])+' <Z>'
                x[i] = replace_token1 + x[i][-1:]
                x_ = x[:]
                for j in x_:
                    if '<X>' not in j and 'S| '+l[i][0][3:] not in j and 'O| '+l[i][0][3:] not in j and 'S| '+l[i][2][3:] not in j and 'O| '+l[i][2][3:] not in j:
                        y = y[:-3] + '<Y> ' + x[x.index(j)][1] +'<Z>'
                        x[x.index(j)][1] = replace_token2
                        break
                wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
                wf_tgt.write(y + '\n')
                
        else:
            y = '<Y> ' + x[i][1] +'<Z>'
            x[i][1] = replace_token2
            wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
            wf_tgt.write(y + '\n')

    wf_tgt.close()
    wf_src.close()

def generate_dataset_triple_eventNar(pair_src, pair_tgt, l):
    wf_src = open(pair_src, 'a')
    wf_tgt = open(pair_tgt, 'a')
    
    if len(l) != 1:
        x = copy.deepcopy(l)
        y = ''
        replace_token1 = '<X>'
        y = ''
        x[0] = replace_token1
        y += replace_token1 + ' ' + str(l[0]) + ' '
        y += '<Z>'
        wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
        wf_tgt.write(y + '\n')

def generate_dataset_triple_and_relation_eventNar(pair_src, pair_tgt, l):
    wf_src = open(pair_src, 'a')
    wf_tgt = open(pair_tgt, 'a')
    
    x = copy.deepcopy(l)
    replace_token1 = '<X>'
    replace_token2 = '<Y>'
    if len(l) != 1:
        y = ''
        x[0] = replace_token1
        y += replace_token1 + ' ' + str(l[0]) + ' '
        x[-1][1] = replace_token2
        y += replace_token2 + ' ' + l[-1][1] + ' '
    else:
        x[0][1] = replace_token2
        y = replace_token2 + ' ' + l[0][1] + ' '
    y += '<Z>'
    wf_src.write(json.dumps(x, ensure_ascii=False) + '\n')
    wf_tgt.write(y + '\n')

data_path1 = '/mnt/worknfs/jyliu/hw9/Graph-Masking-Pre-training-main/preprocess/'         #1

pair_train_src1 = data_path1 + "masked_webnlg20/train_triple.source"
pair_train_tgt1 = data_path1 + "masked_webnlg20/train_triple.target"

pair_train_src2 = data_path1 + "masked_webnlg20/train_triple_and_relation.source"
pair_train_tgt2 = data_path1 + "masked_webnlg20/train_triple_and_relation.target"

pair_val_src1 = data_path1 + "masked_webnlg20/val_triple.source"             #1
pair_val_tgt1 = data_path1 + "masked_webnlg20/val_triple.target"

pair_val_src2 = data_path1 + "masked_webnlg20/val_triple_and_relation.source"
pair_val_tgt2 = data_path1 + "masked_webnlg20/val_triple_and_relation.target"

pair_test_src1 = data_path1 + "masked_webnlg20/test_triple.source"             #1
pair_test_tgt1 = data_path1 + "masked_webnlg20/test_triple.target"

pair_test_src2 = data_path1 + "masked_webnlg20/test_triple_and_relation.source"
pair_test_tgt2 = data_path1 + "masked_webnlg20/test_triple_and_relation.target"

with open("/mnt/worknfs/jyliu/hw9/Graph-Masking-Pre-training-main/preprocess/webnlg20_ordered/train.source") as f:   #1
    for l in f.readlines():
        # change string to list
        l = ast.literal_eval(l)
        generate_dataset_triple(pair_train_src1, pair_train_tgt1, l, remove = False)
        generate_dataset_triple_and_relation(pair_train_src2, pair_train_tgt2, l)
        
with open("/mnt/worknfs/jyliu/hw9/Graph-Masking-Pre-training-main/preprocess/webnlg20_ordered/val.source") as f:   #1
    for l in f.readlines():
        # change string to list
        l = ast.literal_eval(l)
        generate_dataset_triple(pair_val_src1, pair_val_tgt1, l, remove = False)
        generate_dataset_triple_and_relation(pair_val_src2, pair_val_tgt2, l)
        
with open("/mnt/worknfs/jyliu/hw9/Graph-Masking-Pre-training-main/preprocess/webnlg20_ordered/test.source") as f:   #1
    for l in f.readlines():
        # change string to list
        l = ast.literal_eval(l)
        generate_dataset_triple(pair_test_src1, pair_test_tgt1, l, remove = False)
        generate_dataset_triple_and_relation(pair_test_src2, pair_test_tgt2, l)
          
          
#train_origin_src = "masked_webnlg20/triple_and_relation/train.source"          #1
#train_origin_tgt = "masked_webnlg20/triple_and_relation/train.target"
#train_cleaned_src = "masked_webnlg20/triple_and_relation_cleaned/train.source"
#train_cleaned_tgt = "masked_webnlg20/triple_and_relation_cleaned/train.target"

train_origin_src = data_path1 + "masked_webnlg20/train_triple_and_relation.source"          #1
train_origin_tgt = data_path1 + "masked_webnlg20/train_triple_and_relation.target"
train_cleaned_src = data_path1 + "masked_webnlg20/triple_and_relation_cleaned/train.source"
train_cleaned_tgt = data_path1 + "masked_webnlg20/triple_and_relation_cleaned/train.target"

val_origin_src = data_path1 + "masked_webnlg20/val_triple_and_relation.source"          #1
val_origin_tgt = data_path1 + "masked_webnlg20/val_triple_and_relation.target"
val_cleaned_src = data_path1 + "masked_webnlg20/triple_and_relation_cleaned/val.source"
val_cleaned_tgt = data_path1 + "masked_webnlg20/triple_and_relation_cleaned/val.target"

test_origin_src = data_path1 + "masked_webnlg20/test_triple_and_relation.source"          #1
test_origin_tgt = data_path1 + "masked_webnlg20/test_triple_and_relation.target"
test_cleaned_src = data_path1 + "masked_webnlg20/triple_and_relation_cleaned/test.source"
test_cleaned_tgt = data_path1 + "masked_webnlg20/triple_and_relation_cleaned/test.target"

clean_dataset(train_origin_src, train_origin_tgt, train_cleaned_src, train_cleaned_tgt)
clean_dataset(val_origin_src, val_origin_tgt, val_cleaned_src, val_cleaned_tgt)
clean_dataset(test_origin_src, test_origin_tgt, test_cleaned_src, test_cleaned_tgt)