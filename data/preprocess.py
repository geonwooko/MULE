import pandas as pd
import numpy as np
import json



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
        
        # Load behavior-specific graph
        view = np.loadtxt(f'data/{data}/view.txt', dtype=int)
        cart = np.loadtxt(f'data/{data}/cart.txt', dtype=int)
        if data != 'taobao':
            collect = np.loadtxt(f'data/{data}/collect.txt', dtype=int)
        train_buy = np.loadtxt(f'data/{data}/train.txt', dtype=int)
        test_buy = np.loadtxt(f'data/{data}/test.txt', dtype=int)
        
        
        # Preprocess target-complemented behaviors
        view_not_buy = list2set(view).difference(list2set(train_buy))
        cart_not_buy = list2set(cart).difference(list2set(train_buy))
        if data != 'taobao':
            collect_not_buy = list2set(collect).difference(list2set(train_buy))
        
        np.savetxt(f'data/{data}/view_not_buy.txt', sorted(list(view_not_buy)), fmt='%d')
        np.savetxt(f'data/{data}/cart_not_buy.txt', sorted(list(cart_not_buy)), fmt='%d')
        if data != 'taobao':
            np.savetxt(f'data/{data}/collect_not_buy.txt', sorted(list(collect_not_buy)), fmt='%d')
        
        
        # Preprocess target-intersected behaviors
        view_buy = list2set(view).intersection(list2set(train_buy))
        cart_buy = list2set(cart).intersection(list2set(train_buy))
        if data != 'taobao':
            collect_buy = list2set(collect).intersection(list2set(train_buy))
        
        np.savetxt(f'data/{data}/view_buy.txt', sorted(list(view_buy)), fmt='%d')
        np.savetxt(f'data/{data}/cart_buy.txt', sorted(list(cart_buy)), fmt='%d')
        if data != 'taobao':
            np.savetxt(f'data/{data}/collect_buy.txt', sorted(list(collect_buy)), fmt='%d')
        

        assert len(view) == len(view_not_buy) + len(view_buy)
        assert len(cart) == len(cart_not_buy) + len(cart_buy)
        if data != 'taobao':
            assert len(collect) == len(collect_not_buy) + len(collect_buy)
        
        
        # Preprocess unified behavior graph
        all_edge = list2set(view).union(list2set(cart)).union(list2set(train_buy))
        if data != 'taobao':
            all_edge = all_edge.union(list2set(collect))
        all_edge = np.array(sorted(list(all_edge)))
        np.savetxt(f'data/{data}/ubg.txt', all_edge, fmt='%d')
        
        
        # Generate train/test buy interaction dict
        n_users = all_edge[:,0].max()
        n_items = all_edge[:,1].max()
            
        train_dict = list2dict(train_buy, n_users)
        test_dict = list2dict(test_buy, n_items)
        
        with open(f'data/{data}/train.json', 'w') as f:
            json.dump(train_dict, f)
        with open(f'data/{data}/test.json', 'w') as f:
            json.dump(test_dict, f)
        
        
        # Generate data statistics
        bsg_types = ['view', 'cart', 'buy'] if data == 'taobao' else ['view', 'cart', 'collect', 'buy']
        tcb_types = ['view_not_buy', 'cart_not_buy'] if data == 'taobao' else ['view_not_buy', 'collect_not_buy', 'cart_not_buy']
        tib_types = ['view_buy', 'cart_buy'] if data == 'taobao' else ['view_buy', 'collect_buy', 'cart_buy']
        trbg_types = tcb_types + tib_types
        
        data_statistics = {
            'n_users': int(n_users),
            'n_items': int(n_items),
            'bsg_types': bsg_types,
            'tcb_types': tcb_types,
            'tib_types': tib_types,
            'trbg_types': trbg_types,
            'n_view': len(view),
            'n_cart': len(cart),
            'n_collect': len(collect) if data != 'taobao' else 0,
            'n_buy': len(train_buy),
            'n_view_not_buy': len(view_not_buy),
            'n_cart_not_buy': len(cart_not_buy),
            'n_collect_not_buy': len(collect_not_buy) if data != 'taobao' else 0,
            'n_view_buy': len(view_buy),
            'n_cart_buy': len(cart_buy),
            'n_collect_buy': len(collect_buy) if data != 'taobao' else 0,
            'n_ubg': len(all_edge)
        }
        with open(f'data/{data}/statistics.json', 'w') as f:
            json.dump(data_statistics, f)
                

if __name__ == '__main__':
    preprocess()