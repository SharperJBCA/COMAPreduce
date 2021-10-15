import h5py
#from mpi4py import MPI
import numpy as np
import configparser
import time
from astropy.time import Time
from datetime import datetime
#comm = MPI.COMM_WORLD

class DataStructure(object):

    def __init__(self,logger=print,overwrite=False,
                 bad_keywords = ['halt','current','zenith','AlpBoo'],**kwargs):
        self.logger = logger
        self.overwrite = overwrite

        self.bad_keywords = bad_keywords

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        self.run(data)
        #self.plot(data)
        self.write(data)
        try:
            #self.plot(data)
            pass
        except:
            print('Plotting Failed')

        return data

    def run(self,data):
        pass

    def write(self, data):
        pass

    def plot(self,data):
        pass

    def __str__(self):
        return "Unknown COMAP Reduce Module"

    def featureBits(self,features, target, trim=1000):
        """
        Return list of features encoded into feature bit
        """
        
        features[features == 0] = 0.1
        p2 = np.floor(np.log(features)/np.log(2))
        
        select = (p2 == target)
        a = np.where(select)[0]
        select[a[:trim]] = False
        return select

    @staticmethod
    def getFeatures(data):
        """
        Get the feature bits, will check to see if it exists in level 1 first,
        otherwise it will create the features.
        """
        if 'level1' in data:
            target = 'level1/spectrometer'
        else:
            target = 'spectrometer'

        if 'features' in data[target]:
            return data[target][features][...]
        else:
            return 

    @staticmethod
    def getMJD(data):
        """
        Return start MJD
        """
        if 'level1' in data:
            target = 'level1/comap'
        else:
            target = 'comap'

        dt = datetime.strptime(data[target].attrs['utc_start'].decode(),'%Y-%m-%d-%H:%M:%S')

        return Time(dt).mjd

    @staticmethod
    def getObsID(data):
        """
        Return ObsID
        """
        if 'level1' in data:
            target = 'level1/comap'
        else:
            target = 'comap'

        s = data[target].attrs['obsid']
        if isinstance(s,str):
            return s
        else:
            return s.decode('utf-8')

    @staticmethod
    def getFeeds(data,feeds_select):
        """
        Return list of feeds and feed indices
        """
        fname = data.filename.split('/')[-1]
        if 'level1' in data:
            target = 'level1/spectrometer/feeds'
        else:
            target = 'spectrometer/feeds'

        data_feeds = data[target][:]
        if feeds_select == 'all':
            feeds = data_feeds
        else:
            if not isinstance(feeds_select,np.ndarray) and not isinstance(feeds_select,list):
                feeds_select = [feeds_select]
            feeds = [int(f) for f in feeds_select]

        # Now find the feed indices
        feed_indices = np.array([i for i,f in enumerate(feeds) if f in data_feeds])
        feed_dict    = {f:i for i,f in enumerate(feeds)}
        feed_strs = '..'.join([str(f) for f in feeds])
        return feeds, feed_indices, feed_dict

    @staticmethod
    def getComment(data):
        """
        """
        fname = data.filename.split('/')[-1]
        comment = DataStructure.getAttr(data,'comment')
        if comment == '':
            comment = 'No Comment'
        return comment

    def getSource(self,data):
        """
        """
        fname = data.filename.split('/')[-1]
        source = DataStructure.getAttr(data,'source')
        
        
        source_split = source.split(',')
        if len(source_split) > 1:
            source = [s for s in source_split if s not in self.bad_keywords]
            if len(source) > 0:
                source = source[0]
            else:
                source = ''
        if source == '':
            self.logger(f'{fname}:{self.name}: No source found.')
            source = ''
        return source.strip()

    @staticmethod
    def getAttr(data,attrname):
        
        if 'level1' in data:
            target = 'level1/comap'
        else:
            target = 'comap'

        if attrname in data[target].attrs:
            attr = data[target].attrs[attrname]
            if not isinstance(attr,str):
                attr = attr.decode('utf-8')
        else:
            attr = ''
        return attr

    def setReadWrite(self,data):
        if not data.mode == 'r+':
            filename = data.filename
            data.close()
            data = h5py.File(filename,'r+')
        return data

    def getGroup(self,data,grp,grpname):
        fname = data.filename.split('/')[-1]
        if not grpname in grp:
            self.logger(f'{fname}:{self.name}: No {grpname} found.')
            raise KeyError
        output = grp[grpname]
        if isinstance(output,h5py.Dataset):
            return output[...]
        else:
            return output

    def checkAllowedSources(self, data, source, allowed_sources):
        """
        """
        fname = data.filename.split('/')[-1]
        if not source in allowed_sources:
            allowed_source_str = ' '.join(allowed_sources)
            self.logger(f'{fname}:{self.name}:Error: {source} not in allowed source ({allowed_source_str})')
            return True
        else:
            return False
