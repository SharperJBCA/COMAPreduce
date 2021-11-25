import numpy as np

def read_2bit(feature):
    """
    For a given 2 bit feature, read all features
    """

    top_feature = np.floor(np.log(feature)/np.log(2))

    features = [top_feature]
    while True:
        fs = feature - np.sum([2**f for f in features])

        if fs <= 0:
            break

        next_feature = np.floor(np.log(fs)/np.log(2))
        features += [next_feature]

    return [int(f) for f in features]


def get_obsfeatures(features):

    u_features = np.unique(features)

    features = []
    for ufeature in u_features:
        features += read_2bit(ufeature)

    return np.unqiue(features)


import h5py

class visitor_func:

    def __init__(self):
        self.values = {}

    def __call__(self,name,node):
        if isinstance(node,h5py.Dataset):
            if len(node.attrs.keys()) != 1:
                self.values[name] = node[...]
            else:
                self.values[name] = node.attrs[name].split(',')
visitor_func_instance = visitor_func()


class parse_dictionary:
    """
    Make a dictionary flat so it is easily written to an HDF5 file
    """

    def __init__(self):
        flat_d = {}
        keys = []
        level = 0

    def __call__(self,d):
        self.flat_d = {}
        self.keys = []
        self.level = 0
        return self.print_dict(d)

    def print_dict(self,d):
        for k,v in d.items():
            if isinstance(v,dict):
                self.level += 1
                self.keys += [k]
                self.print_dict(v)
                self.keys = self.keys[:-1]
                self.level -= 1
            else:
                self.flat_d['/'.join(self.keys+[k])] = v
        return self.flat_d


def create_dataset(h,name,data):
    if name in h:
        del h[name]
        
    if data.size == 1:
        data = np.array([data])
    if data.dtype.kind in ['U','S']:
        if not (isinstance(data,list) | isinstance(data,np.ndarray)):
            data = [data]
        string = u','.join([d for d in data])
        z =  h.create_dataset(name,(1,))
        z.attrs[name] = string
        return z
    else:
        return h.create_dataset(name,data=data)


def save_dict_hdf5(data,output_file):
    """
    Given a python dictionary, save to an HDF5 file with the same format
    """
    pd = parse_dictionary()
    flat_data = pd(data)

    h = h5py.File(output_file,'a')
    print(output_file)
    for k,v in flat_data.items():
        print(k)
        if isinstance(v,np.ndarray):
            if v.dtype.type is np.object_:
                print(f'WARNING: {k} not written, need to define python object write format')
            create_dataset(h,k,v)
        elif isinstance(v,list):
            create_dataset(h,k,np.array(v))
        else:
            create_dataset(h,k,np.array([v]))
    h.close()

def load_dict_hdf5(plot_data,no_set=False):

    with h5py.File(plot_data,'r') as h:
        h.visititems(visitor_func_instance)

    def load_dict(k,v):
        if isinstance(v,dict):
            load_dict(k,v)
        else:
            v[k] = v
        
    data = {}
    for k,v in visitor_func_instance.values.items():
        grps = k.split('/')
        this_level = data
        for ikey, key in enumerate(grps):
            if ikey < len(grps)-1:
                if not key in this_level:
                    this_level[key] = {}
                this_level = this_level[key]
            else:
                this_level[key] = v

    return data
