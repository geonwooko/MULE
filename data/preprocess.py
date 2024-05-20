import pandas as pd
import numpy as np
import json

data2nusers = {
    'taobao': 15449,
    'tmall': 41738,
    'jdata': 93334
}

data2nitems = {
    'taobao': 11953,
    'tmall': 11953,
    'jdata': 24624
}

def list2set(l):
    return set([tuple(edge) for edge in l])

def list2dict(l, n_users):
    user_interaction_dict = dict()
    for edge in l:
        user, item = int(edge[0]), int(edge[1])
        if user not in user_interaction_dict.keys():
            user_interaction_dict[user] = []
        user_interaction_dict[user].append(item)
    return user_interaction_dict

def preprocess():
    for data in ['taobao', 'tmall', 'jdata']:
        print(f'Preprocessing {data}...')
        nusers = data2nusers[data]
        
        view = np.loadtxt(f'data/{data}/view.txt', dtype=int)
        cart = np.loadtxt(f'data/{data}/cart.txt', dtype=int)
        if data != 'taobao':
            collect = np.loadtxt(f'data/{data}/collect.txt', dtype=int)
        train_buy = np.loadtxt(f'data/{data}/train.txt', dtype=int)
        test_buy = np.loadtxt(f'data/{data}/test.txt', dtype=int)
        
        view_not_buy = list2set(view).difference(list2set(train_buy))
        cart_not_buy = list2set(cart).difference(list2set(train_buy))
        if data != 'taobao':
            collect_not_buy = list2set(collect).difference(list2set(train_buy))
        
        view_buy = list2set(view).intersection(list2set(train_buy))
        cart_buy = list2set(cart).intersection(list2set(train_buy))
        if data != 'taobao':
            collect_buy = list2set(collect).intersection(list2set(train_buy))
        
        assert len(view) == len(view_not_buy) + len(view_buy)
        assert len(cart) == len(cart_not_buy) + len(cart_buy)
        if data != 'taobao':
            assert len(collect) == len(collect_not_buy) + len(collect_buy)
        
        np.savetxt(f'data/{data}/view_not_buy.txt', sorted(list(view_not_buy)), fmt='%d')
        np.savetxt(f'data/{data}/cart_not_buy.txt', sorted(list(cart_not_buy)), fmt='%d')
        if data != 'taobao':
            np.savetxt(f'data/{data}/collect_not_buy.txt', sorted(list(collect_not_buy)), fmt='%d')
        
        np.savetxt(f'data/{data}/view_buy.txt', sorted(list(view_buy)), fmt='%d')
        np.savetxt(f'data/{data}/cart_buy.txt', sorted(list(cart_buy)), fmt='%d')
        if data != 'taobao':
            np.savetxt(f'data/{data}/collect_buy.txt', sorted(list(collect_buy)), fmt='%d')
            
        all_edge = list2set(view).union(list2set(cart)).union(list2set(train_buy))
        if data != 'taobao':
            all_edge = all_edge.union(list2set(collect))
        np.savetxt(f'data/{data}/ubg.txt', sorted(list(all_edge)), fmt='%d')
            
        train_dict = list2dict(train_buy, nusers)
        test_dict = list2dict(test_buy, nusers)
        
        with open(f'data/{data}/train.json', 'w') as f:
            json.dump(train_dict, f)
        with open(f'data/{data}/test.json', 'w') as f:
            json.dump(test_dict, f)
            

if __name__ == '__main__':
    preprocess()