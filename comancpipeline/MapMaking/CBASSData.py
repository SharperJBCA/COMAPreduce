import numpy as np
import h5py
from astropy.io import fits
from tqdm import tqdm
import healpy as hp
from comancpipeline.Tools.median_filter import medfilt
from comancpipeline.Tools import Coordinates 
import os

import sys
import psutil 
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
import gc 
from dataclasses import dataclass, field

CBASS_LON = -118+17/60. 
CBASS_LAT = 37+14/60.

from comancpipeline.Tools import pysla
def pa(ra, dec, mjd, lon ,lat, degrees=True):
    """
    Calculate parallactic angle
    
    args:
    ra - arraylike, right ascension
    dec- arraylike, declination
    mjd- arraylike
    lon- double, longitude
    lat- double, latitude

    """
    
    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    p = pysla.pa(ra*c, dec*c, mjd, lon*c, lat*c,0)
    return p/c
    

@dataclass
class CBASSData(object):

    filename : str = None # Filename of the data
    stats_hdu : int = 2 # HDU containing the stats
    nside : int = 512 
    obsid : int = -1 
    offset_length : int = 100 

    @staticmethod
    def get_size(filename, offset_length=100):
        """ Get the size of the data """
        hdu = fits.open(filename,memmap=False)
        nsize = int(hdu[1].data['I1'].size//offset_length*offset_length)
        hdu.close()
        return nsize

    def __post_init__(self):
        """ Load the data """
        self.extra_data = {} 

        self.hdu = fits.open(self.filename,memmap=False)

        self.load_data() 
        self.load_weights() 
        self.flag_data() 
        self.calc_parallactic_angle() 
        self.hdu.close() 

    def __del__(self):
        self.hdu.close() 
        try:
            self.clear_memory() 
        except AttributeError:
            pass

    def clear_memory(self):
        del self.I1 
        del self.I2 
        del self.Q1
        del self.Q2
        del self.U1
        del self.U2
        del self.wI1
        del self.wI2
        del self.wQ1
        del self.wQ2
        del self.wU1
        del self.wU2
        del self.dayflag
        del self.sundist

    @property
    def nsize(self):
        if hasattr(self,'I1'):
            return int((self.I1.size//self.offset_length)*self.offset_length)
        else:
            return int((self.hdu[1].data['I1'].size//self.offset_length)*self.offset_length)


    def load_data(self):
        self.I1 = self.hdu[1].data['I1'][:self.nsize]
        self.I2 = self.hdu[1].data['I2'][:self.nsize]
        self.Q1 = self.hdu[1].data['Q1'][:self.nsize]
        self.Q2 = self.hdu[1].data['Q2'][:self.nsize]
        self.U1 = self.hdu[1].data['U1'][:self.nsize]
        self.U2 = self.hdu[1].data['U2'][:self.nsize]
        self.flag = self.hdu[1].data['FLAG'][:self.nsize]
        self.dayflag = self.hdu[1].data['DAYFLAG'][:self.nsize]
        self.sundist = self.hdu[1].data['SUNDIST'][:self.nsize]
        self.mjd = self.hdu[1].data['MJD'][:self.nsize]
        self.az = self.hdu[1].data['AZ'][:self.nsize]
        self.el = self.hdu[1].data['EL'][:self.nsize]
        self.ra = self.hdu[1].data['RA'][:self.nsize]
        self.dec = self.hdu[1].data['DEC'][:self.nsize]

    def load_weights(self):
        self.wI1 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['I1_sigma'][0]**2 
        self.wI2 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['I2_sigma'][0]**2
        self.wQ1 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['Q1_sigma'][0]**2
        self.wQ2 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['Q2_sigma'][0]**2
        self.wU1 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['U1_sigma'][0]**2
        self.wU2 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['U2_sigma'][0]**2

    def flag_data(self):
        """ Flag the data based on the dayflag and sundist """
        self.wI1[self.flag != 0] = 0
        self.wI2[self.flag != 0] = 0
        self.wQ1[self.flag != 0] = 0
        self.wQ2[self.flag != 0] = 0
        self.wU1[self.flag != 0] = 0
        self.wU2[self.flag != 0] = 0

    def calc_parallactic_angle(self):
        self.pa = pa(np.mod(self.ra[::100]*180./np.pi,360),self.dec[::100]*180./np.pi,self.mjd[::100],CBASS_LON,CBASS_LAT)
        #self.pa[self.pa < 0] += 360
        self.pa = np.interp(self.mjd,self.mjd[::100],self.pa, period=180)*np.pi/180.
        #self.pa[self.pa > np.pi] -= 2*np.pi

    @staticmethod
    def calc_empty_offsets(flag, offset_length=100):
        """ Find where offsets are completely masked and remove them """
        flag_steps = flag.reshape(-1,offset_length)
        flag_steps = np.repeat(np.sum(flag_steps,axis=1), offset_length) 

        good_offsets = (flag_steps < offset_length*0.1)
        return good_offsets

    def save_to_hdf5(self, filename):  
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if os.path.exists(filename):
            os.remove(filename)

        h = h5py.File(filename,'w')
        # h.create_dataset('I1',data=self.I1)
        # h.create_dataset('I2',data=self.I2)
        # h.create_dataset('Q1',data=self.Q1)
        # h.create_dataset('Q2',data=self.Q2)
        # h.create_dataset('U1',data=self.U1)
        # h.create_dataset('U2',data=self.U2)
        # h.create_dataset('wI1',data=self.wI1)
        # h.create_dataset('wI2',data=self.wI2)
        # h.create_dataset('wQ1',data=self.wQ1)
        # h.create_dataset('wQ2',data=self.wQ2)
        # h.create_dataset('wU1',data=self.wU1)
        # h.create_dataset('wU2',data=self.wU2)
        # h.create_dataset('flag',data=self.flag)
        # h.create_dataset('dayflag',data=self.dayflag)
        # h.create_dataset('sundist',data=self.sundist)
        h.create_dataset('mjd',data=self.mjd)
        h.create_dataset('az',data=self.az)
        h.create_dataset('el',data=self.el)
        h.create_dataset('ra',data=self.ra)
        h.create_dataset('dec',data=self.dec)

        for k,v in self.extra_data.items():
            h.create_dataset(k,data=v)

        h.close()

    @property
    def tod(self):
        tod = np.zeros(self.nsize) 
        mask = self.flag == 0
        tod[mask] = (self.I1[mask]*self.wI1[mask] + self.I2[mask]*self.wI2[mask])/(self.wI1[mask] + self.wI2[mask])
        tod[mask] -= np.nanmedian(tod[mask]) 
        return tod
    @property
    def tod_iqu(self):
        tod = np.zeros((3,self.nsize))
        mask = (self.flag == 0)
        tod[0,mask] = (self.I1[mask]*self.wI1[mask] + self.I2[mask]*self.wI2[mask])/(self.wI1[mask] + self.wI2[mask])
        tod[1,mask] =  self.Q1[mask]#*self.wQ1[mask] + self.Q2[mask]*self.wQ2[mask])/(self.wQ1[mask] + self.wQ2[mask])
        tod[2,mask] =  self.U1[mask]#*self.wU1[mask] + self.U2[mask]*self.wU2[mask])/(self.wU1[mask] + self.wU2[mask])
        #tod -= np.nanmedian(tod,axis=1)[:,None]
        tod_offsets = np.reshape(tod,(3,-1,self.offset_length)) 
        tod_offsets -= np.nanmedian(tod_offsets,axis=2)[:,:,None]
        del tod
        tod = tod_offsets.flatten()
        del tod_offsets 
        return tod

    @property
    def weights(self):
        weights = self.wI1[:self.nsize] + self.wI2[:self.nsize]
        return weights
    
    @property
    def weights_iqu(self):
        weights = np.zeros((3,self.nsize))
        weights[0,:] = self.wI1[:self.nsize] + self.wI2[:self.nsize]
        weights[1,:] = self.wQ1[:self.nsize] #+ self.wQ2[:self.nsize]
        weights[2,:] = self.wU1[:self.nsize] #+ self.wU2[:self.nsize]

        weights = weights.flatten()

        return weights

    @property
    def pointing(self):
        pointing = hp.ang2pix(self.nside,np.pi/2 - self.dec[:self.nsize],self.ra[:self.nsize]).astype(int)
        return pointing
    
    @property
    def pointing_iqu(self):
        pointing = np.zeros((3,self.nsize))
        pointing[:,:] = (hp.ang2pix(self.nside,np.pi/2 - self.dec[:self.nsize],self.ra[:self.nsize]).astype(int))[None,:]
        pointing[1,:] += 12*self.nside**2
        pointing[2,:] += 2*12*self.nside**2
        return pointing.flatten()
    
    @property
    def obsid_array(self):
        obsids = np.ones(self.nsize)*self.obsid
        return obsids
    
    @property
    def obsid_array_iqu(self):
        obsids = np.ones((3,self.nsize))*self.obsid
        return obsids.flatten()
    
    @property
    def special_weights(self):
        """ This is where we store the parallactic angle weights for destriping"""
        special_weights = np.ones(self.nsize)
        return special_weights

    @property
    def special_weights_iqu(self):
        """ This is where we store the parallactic angle weights for destriping"""
        pa = self.pa
        special_weights = np.ones((3,self.nsize))
        special_weights[1,:] =  np.cos(2*pa)
        special_weights[2,:] = -np.sin(2*pa)
        special_weights_rot = np.ones((3,self.nsize))
        special_weights_rot[1,:] = np.sin(2*pa)
        special_weights_rot[2,:] = np.cos(2*pa)
        special_weights_pa = np.ones((3,self.nsize)) 
        special_weights_pa[1,:] = 2
        special_weights_pa[2,:] = 3
        return special_weights.flatten(),special_weights_rot.flatten(),special_weights_pa.flatten() 
    
@dataclass
class CBASSDataSim(CBASSData):

    filename : str = None # Filename of the data
    stats_hdu : int = 2 # HDU containing the stats
    nside : int = 512 
    obsid : int = -1 
    offset_length : int = 100 

    cbass_map : np.ndarray = field(default_factory=lambda: np.zeros((3,12*512**2)))

    @property
    def tod_iqu(self):
        tod = np.zeros((3,self.nsize))
        mask = (self.flag == 0)
        pixels = hp.ang2pix(hp.npix2nside(self.cbass_map.shape[1]),np.pi/2 - self.dec[:self.nsize],self.ra[:self.nsize]).astype(int)
        pixels = pixels[mask] 
        I = self.cbass_map[0,pixels]
        I[np.abs(I) > 1e10] = 0
        Q = self.cbass_map[1,pixels]
        Q[np.abs(Q) > 1e10] = 0
        U = self.cbass_map[2,pixels]
        U[np.abs(U) > 1e10] = 0
        tod[0,mask] = I #+ np.random.normal(size=I.size,scale=1/np.sqrt(self.wI1[mask] + self.wI2[mask]))
        tod[1,mask] =  Q*np.cos(2*self.pa[mask]) + U*np.sin(2*self.pa[mask])  #+ np.random.normal(size=I.size,scale=1/np.sqrt(self.wI1[mask] + self.wI2[mask]))
        tod[2,mask] = -Q*np.sin(2*self.pa[mask]) + U*np.cos(2*self.pa[mask])  #+ np.random.normal(size=I.size,scale=1/np.sqrt(self.wI1[mask] + self.wI2[mask]))


        return tod.flatten()

def  read_comap_data(_filelist, offset_length=100, ifile_start=0, ifile_end=None): 
    if isinstance(ifile_end, type(None)):
        ifile_end = len(_filelist)

    if rank == 0:
        filelist = tqdm(_filelist[ifile_start:ifile_end])
    else:
        filelist = _filelist[ifile_start:ifile_end]

    ncount = 0 
    for filename in filelist:
        ncount += CBASSData.get_size(filename, offset_length=offset_length)

    # Create the arrays
    tod = np.zeros(ncount,dtype=np.float64)
    weights = np.zeros(ncount,dtype=np.float64)
    pointing = np.zeros(ncount,dtype=np.int64)
    obsid = np.zeros(ncount,dtype=np.int64)
    flags = np.zeros(ncount,dtype=np.int64)
    special_weights = np.zeros(ncount,dtype=np.float64)

    # Loop over the files
    nstart = 0

    all_cbass_data = [] 
    for i,filename in enumerate(filelist):
        cbass_data = CBASSData(filename,obsid=i+ifile_start,offset_length=offset_length)
        nend = nstart + cbass_data.nsize

        tod[nstart:nend] = cbass_data.tod
        weights[nstart:nend] = cbass_data.weights 
        pointing[nstart:nend] = cbass_data.pointing
        obsid[nstart:nend] = cbass_data.obsid_array
        flags[nstart:nend] = cbass_data.flag[:cbass_data.nsize]
        special_weights[nstart:nend] = cbass_data.special_weights

        all_cbass_data += [cbass_data]
        nstart = nend

    good_offsets = CBASSData.calc_empty_offsets(flags, offset_length=offset_length)
    return tod[good_offsets], weights[good_offsets], pointing[good_offsets], obsid[good_offsets], special_weights[good_offsets], all_cbass_data

def  read_comap_data_iqu(_filelist, offset_length=100, ifile_start=0, ifile_end=None,nside=512): 
    if isinstance(ifile_end, type(None)):
        ifile_end = len(_filelist)

    if rank == 0:
        filelist = tqdm(_filelist[ifile_start:ifile_end])
    else:
        filelist = _filelist[ifile_start:ifile_end]

    ncount = 0 
    for filename in filelist:
        ncount += CBASSData.get_size(filename, offset_length=offset_length)*3

    # Create the arrays
    tod = np.random.normal(size=ncount)# np.zeros(ncount,dtype=np.float64)
    weights = np.zeros(ncount,dtype=np.float64)
    pointing = np.zeros(ncount,dtype=np.int64)
    obsid = np.zeros(ncount,dtype=np.int64)
    flags = np.zeros(ncount,dtype=np.int64)
    special_weights = np.zeros(ncount,dtype=np.float64)
    special_weights_rot = np.zeros(ncount,dtype=np.float64)
    special_weights_pa = np.zeros(ncount,dtype=np.float64)

    # Loop over the files
    nstart = 0

    cbass_map = hp.read_map('/scratch/nas_cbassarc/cbass_data/Reductions/v34m3_mcal1/NIGHTMERID20/AWR1/rawest_map/AWR1_xND12_xAS14_1024_NM20S3M1_C_Offmap.fits',field=[0,1,2])
    cbass_map = hp.ud_grade(hp.read_map('/scratch/nas_cbassarc/cbass_data/Reductions/v34m3_mcal1/NIGHTMERID20/AWR1/calibrated_map/AWR1_xND12_xAS14_1024_NM20S3M1_C_Offmap.fits',field=[0,1,2]),64)

    print('BEFORE LOOP',psutil.virtual_memory())

    all_cbass_data = [] 
    for i,filename in enumerate(filelist):
        cbass_data = CBASSData(filename,obsid=i+ifile_start,offset_length=offset_length,nside=nside)#,cbass_map=cbass_map)
        _nend = nstart + cbass_data.nsize*3
        nend = _nend 

        tod[nstart:nend] = cbass_data.tod_iqu
        weights[nstart:nend] = cbass_data.weights_iqu
        pointing[nstart:nend] = cbass_data.pointing_iqu
        obsid[nstart:nend] = cbass_data.obsid_array_iqu
        flags[nstart:nend] = np.tile(cbass_data.flag[:cbass_data.nsize],3)
        special_weights[nstart:nend],special_weights_rot[nstart:nend],special_weights_pa[nstart:nend] = cbass_data.special_weights_iqu

        cbass_data.clear_memory() 
        if rank == 0:
            print('AFTER',psutil.virtual_memory())
        all_cbass_data += [cbass_data]
        nstart = _nend
        

    good_offsets = CBASSData.calc_empty_offsets(flags, offset_length=offset_length)
    return tod[good_offsets], weights[good_offsets], pointing[good_offsets], obsid[good_offsets], special_weights[good_offsets], special_weights_rot[good_offsets], special_weights_pa[good_offsets], all_cbass_data