import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import argparse

def celeba():
    label_path = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno.txt'
    test_path = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno-test.csv'
    train_path = 'data/CelebA-HQ/CelebAMask-HQ-attribute-anno-train.csv'
    df = pd.read_csv(label_path,skiprows = [0],sep = ' ')
    df.reset_index(level=1, drop=True, inplace=True)
    df.index.name = 'images'
    headers =df.columns.values.tolist()
    for header in headers:
        df.loc[df[header] == -1, header] = 0 
    feat2 = {
        'r1': 'Black_Hair',
        'r2': 'Brown_Hair',
        'r3': 'Mustache',
        'r4': 'Wavy_Hair',
        'r5': 'Bald',
        'r6': 'Wearing_Hat'
    }

    feat1 = {
        'r1': 'Eyeglasses',
        'r2': 'Eyeglasses',
        'r3': 'Eyeglasses',
        'r4': 'Eyeglasses',
        'r5': 'Eyeglasses',
        'r6': 'Eyeglasses'
    }

    req = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
    return df, req, feat1, feat2, test_path, train_path
def mnist():
    label_path = 'data/MNIST/train_data_for_reqs.csv'
    test_path = 'data/MNIST/test.csv'
    train_path = 'data/MNIST/train.csv'
    df = pd.read_csv(label_path)
    df.set_index('images')
    headers =df.columns.values.tolist()
    for header in headers:
        df.loc[df[header] == -1, header] = 0 
    feat2 = {
        'r1': '2',
        'r2': '3',
        'r3': '7',
        'r4': '9',
        'r5': '6',
        'r6': '0'
    }

    feat1 = {
        'r1': 'Very_Low_Height',
        'r2': 'Very_Thick',
        'r3': 'Very_Thick',
        'r4': 'Very_Left_Leaning',
        'r5': 'Very_Right_Leaning',
        'r6': 'Very_Low_Height'
    }

    req = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
    return df, req, feat1, feat2, test_path, train_path
def sgsm():
    label_path = 'data/SGSM/sgsm_features_dataset.csv'
    train_path = 'data/SGSM/sgsm_train_dataset.csv'
    test_path = 'data/SGSM/sgsm_test_dataset.csv'
    df = pd.read_csv(label_path)
    df.set_index('images',inplace = True)
    req = ['r1','r2','r3','r4','r5','r6','r7']
    feat1 = {
        'r1': 'f1',
        'r2': 'f2',
        'r3': 'f3a',
        'r4': 'f4a',
        'r5': 'f5a',
        'r6': 'f6',
        'r7': 'f7'
    }

    feat2 = {
        'r1': None,
        'r2': None,
        'r3': 'f3b',
        'r4': 'f4b',
        'r5': 'f5b',
        'r6': None,
        'r7': None
    }
    return df, req, feat1, feat2, test_path, train_path

def precondition_data(df, req, feat1, feat2=None, pos = True):
    r = []
    if pos:
        for i in range(len(df.index)):
            if feat2[req] is not None:
                if df[feat1[req]].iloc[i] == 1 and df[feat2[req]].iloc[i] == 1:
                    r.append(df.index[i])
            else:
                if df[feat1[req]].iloc[i] == 1:
                    r.append(df.index[i])
    else:
        for i in range(len(df.index)):
            if feat2[req] is not None:
                if df[feat1[req]].iloc[i] == 0 and df[feat2[req]].iloc[i] == 0:
                    r.append(df.index[i])
            else:
                if df[feat1[req]].iloc[i] == 0:
                    r.append(df.index[i])
    return r 

def is_a_req(df, index, req, feat1, feat2=None, pos = True):
    if pos:
        if feat2[req] is not None:
            if df[feat1[req]].iloc[index] == 1 and df[feat2[req]].iloc[index] == 1:
                return True
            else:
                return False
        else:
            if df[feat1[req]].iloc[i] == 1:
                return True
            else:
                return False
    else:
        if feat2[req] is not None:
            if df[feat1[req]].iloc[index] != 1 and df[feat2[req]].iloc[index] != 1:
                return True
            else:
                return False
        else:
            if df[feat1[req]].iloc[i] != 1:
                return True
            else:
                return False




def pos_data_add( df, req, feat1, feat2,test_rate = 0.1):
    req_list = []
    req_list_len = []
    for r in req:
        r_list = precondition_data(df, r, feat1, feat2)
        r_list_len = len(r_list)
        req_list.append(r_list) 
        req_list_len.append(r_list_len)

    print('num data for each req: ',req_list_len)
    ##sort it so that we can start with the smallest set
    sorted_list = np.argsort(req_list_len)
    print('index for asc sort: ',sorted_list)

    for i in range(len(sorted_list)):
        ####determine the size of test suit to select
        size = int(req_list_len[sorted_list[i]]*test_rate)

        ####if the test set is empty
        if i == 0:
            test = random.sample(req_list[sorted_list[i]], size)
        else:
            ##### make two sets having values in test set and not in test set
            in_list = []
            not_inlist = []
            for j in req_list[sorted_list[i]]:
                if j in test:
                    in_list.append(j)
                else:
                    not_inlist.append(j)
            if len(in_list)<size:
                rem_size = size - len(in_list)

                ####find if the data clashes with the previous requirements
                valid_set = []
                for k in not_inlist:
                    temp = 1
                    for index in range(0,i):
                        if k in req_list[sorted_list[index]]:
                            temp = 0
                            break
                    if temp == 1:
                        valid_set.append(k)
                ####check that the requirement is not having more data
                if len(valid_set)>rem_size:
                    valid_set = random.sample(valid_set, rem_size)
                ####insert data for the requirement
                for v in valid_set:
                    test.append(v)


    print('test suit size: ',len(test))
    return test, req_list, sorted_list

def check_list_for_value(lst, value, dim = 1):
    if dim ==1:
        for i in lst:
            if value == i:
                return True
        return False
    else:
        for i in lst:
            for j in i:
                if value == j:
                    return True
        return False
def neg_data_add(df, req, feat1, feat2, test, pos_list, sorted_list, test_rate = 0.1):
    neg_list = []
    neg_list_len = []
    for r in req:
        r_list = precondition_data(df, r, feat1, feat2, pos = False)
        r_list_len = len(r_list)
        neg_list.append(r_list) 
        neg_list_len.append(r_list_len)

    print('num data for negation of each req: ',neg_list_len)
    neg_sorted_list = np.argsort(neg_list_len)

    for i in range(len(neg_sorted_list)):
        count = 0
        for j in neg_list[neg_sorted_list[i]]:
            if check_list_for_value(test, j):
                count += 1
        if count < int(len(pos_list[neg_sorted_list[i]])):
            size = int(len(pos_list[neg_sorted_list[i]]))- count
            if count + size > len(neg_list[neg_sorted_list[i]]) * test_rate:
                size = int(len(neg_list[neg_sorted_list[i]]) * test_rate)
                if size>count:
                    size = size - count
            neg = []
            for j in neg_list[neg_sorted_list[i]]:
                if check_list_for_value(test, j) == False and check_list_for_value(pos_list, j, dim = 2) == False:
                    temp = 1
                    if i == 0:
                        neg.append(j)
                    else:
                        for k in range(0,i):
                            if check_list_for_value(neg_list[k], j):
                                temp = 0
                                break
                        if temp == 1:
                            neg.append(j)
                    

            if len(neg)> size:
                neglst = random.sample(neg, size)
            else:
                neglst = neg
            for d in neglst:
                test.append(d)
    return test 
            
def check_feature_count(df, test_df, req):
    print(f'len of testset: {len(test_df.index)}')
    for r in req:
        countpos1 = 0
        countneg1 = 0

        countpos2 = 0
        countneg2 = 0

        testcountpos1 = 0
        testcountneg1 = 0

        testcountpos2 = 0
        testcountneg2 = 0

        for i in range(len(df.index)):
            if df[feat1[r]].iloc[i] == 1:
                countpos1 += 1
            else:
                countneg1 += 1
            if feat2[r] is not None:
                if df[feat2[r]].iloc[i] == 1:
                    countpos2 += 1
                else:
                    countneg2 += 1
        for i in range(len(test_df.index)):  
            if test_df[feat1[r]].iloc[i] == 1:
                testcountpos1 += 1
            else:
                testcountneg1 += 1
            if feat2[r] is not None:
                if test_df[feat2[r]].iloc[i] == 1:
                    testcountpos2 += 1
                else:
                    testcountneg2 += 1
        print(f"total pos for {feat1[r]}: ", countpos1)
        print(f"total neg for {feat1[r]}: ", countneg1)
        print(f"\ntotal pos for {feat1[r]} in test set: ", testcountpos1)
        print(f"total neg for {feat1[r]} in test set: ", testcountneg1, "\n")
        
        if feat2[r] is not None:
            print(f"total pos for {feat2[r]}: ", countpos2)
            print(f"total neg for {feat2[r]}: ", countneg2)
            print(f"\ntotal pos for {feat2[r]} in test set: ", testcountpos2)
            print(f"total neg for {feat2[r]} in test set: ", testcountneg2, "\n")

def check_r_count(df, test_df, req):
    print(f'len of testset: {len(test_df.index)}')

    for r in req:
        countpos = 0
        countneg = 0

        for i in range(len(df.index)):
            if feat2[r] is not None:
                if df[feat1[r]].iloc[i] == 1 and df[feat2[r]].iloc[i] == 1:
                        countpos +=1
                elif df[feat1[r]].iloc[i] != 1 and df[feat2[r]].iloc[i] != 1:
                        countneg +=1
            else:
                if df[feat1[r]].iloc[i] == 1:
                    countpos += 1
                else:
                    countneg +=1
        print(f"total pos for {r}: ", countpos)
        print(f"total neg for {r}: ", countneg)

        testcountpos = 0
        testcountneg = 0
        
        for i in range(len(test_df.index)):
            if feat2[r] is not None:
                if test_df[feat1[r]].iloc[i] == 1 and test_df[feat2[r]].iloc[i] == 1:
                        testcountpos +=1
                elif test_df[feat1[r]].iloc[i] != 1 and test_df[feat2[r]].iloc[i] != 1:
                        testcountneg +=1
            else:
                if test_df[feat1[r]].iloc[i] == 1:
                    testcountpos += 1
                else:
                    testcountneg +=1
        print(f"\ntotal pos for {r} in test set: ", testcountpos)
        print(f"total neg for {r} in test set: ", testcountneg, "\n")


def make_train_test(dataset, df, req, feat1, feat2, test_path, train_path):

    test = []
    test, pos_list, sorted_list = pos_data_add(df, req, feat1, feat2)
    test = neg_data_add(df, req, feat1, feat2, test, pos_list, sorted_list)
    column_name = df.columns
    ###create new test df
    lst = []
    for t in test:
        lst.append(df.loc[t])

    test_df = pd.DataFrame(lst, columns = column_name)

    ################make the training set
    ind = list(df.index)
    test_ind = list(test_df.index)
    print('ind: ',len(ind))
    for t in test_ind:
        # print('t: ',t)
        ind.remove(t)

    lst = []
    for t in ind:
        lst.append(df.loc[t])
    
    train_df = pd.DataFrame(lst, columns = column_name)
    print('saving data...')
    if dataset == 'celeba':
        test_df.to_csv(test_path, index_label = 'images')
        train_df.to_csv(train_path, index_label = 'images')
    else:
        test_df.to_csv(test_path, index = False)
        train_df.to_csv(train_path, index = False)


def make_train_test_dataset(dataset):
    if dataset == 'mnist':
        df, req, feat1, feat2, test_path, train_path  = mnist()
    elif dataset == 'celeba':
        df, req, feat1, feat2, test_path, train_path  = celeba()
    elif dataset == 'sgsm':
        df, req, feat1, feat2, test_path, train_path  = sgsm()
    column_name = df.columns



    make_train_test(dataset, df, req, feat1, feat2, test_path, train_path)

    test_df = pd.read_csv(test_path)
    test_df = test_df.set_index('images')

    print("check requirement wise data")
    check_r_count(df, test_df, req)
    print('check feature wise data')
    check_feature_count(df, test_df, req)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for classifier training')
    parser.add_argument('--dataset', default='celeba', type=str)
    args = parser.parse_args()
    make_train_test_dataset(args.dataset)
    
