
import astropy.wcs as wcs
from astropy.io import fits
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import path
from itertools import compress

from mapext.core.processClass import Process
from mapext.core.mapClass import astroMap
from mapext.core.srcClass import astroSrc

class return_objects():

    def __init__(self,catalogues,coord_query,frame='galactic',return_type = 'srcClass'):

        if catalogues=='all':
            catalogues=[
                        'hiiregions',
                        'westerhout',
                        'yso',
                        'uchii',
                        'snr',
                        'densemoleculargas',
                        'condenseddust',
                        'planetarynebulae',
                        'ic_ngc'
                        ]

        coord_query = SkyCoord(coord_query[:,0],coord_query[:,1],unit = u.degree,frame=frame)
        coord_polygon_query = path.Path(np.array([coord_query.l.degree,coord_query.b.degree]).T)

        #call coordinates of objects
        results = {}
        for cat in tqdm(catalogues):
            callfunc = getattr(self, 'query_'+cat)
            r, name = callfunc(coord_polygon_query)
            result = [astroSrc(coord=SkyCoord(_['Glon']*u.degree,
                                              _['Glat']*u.degree,
                                              frame='galactic'),
                               name=_['Name']) for _ in r]
            results[name] = result
        self.result = results

    def query_hiiregions(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/hii_regions-paladini.fits')[1].data)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)
        catnam = ['G'+_['Gname'].decode() for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'Hii Regions'

    def query_westerhout(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/complexes-westerhout.fits')[1].data)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)
        catnam = ['W'+_['W'].decode()+_['m_W'].decode() for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'Westerhout'

    def query_yso(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/YSO_AKARI_Toth+2014.fit')[1].data)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)
        catnam = [_['AKARI'] for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        print(outlist)
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'YSOs'

    def query_uchii(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/UCHII_CORNISH_Kalcheva+2018.fit')[1].data)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)
        catnam = [_['CORNISH'].decode() for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'UCHii'

    def query_snr(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/SNR_Green2019.fit')[1].data)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)
        catnam = [_['SNR'].decode() for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'SNR'

    def query_densemoleculargas(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/DenseMolGas_BGPS_Shirley+2013.fit')[1].data)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)
        catnam = [_['BGPS'].decode() for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'DMG'

    def query_condenseddust(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/CondensDust_ATLASGAL_Csengeri+2014.fit')[1].data)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)
        catnam = [_['Name'].decode() for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'Condensed Dust'

    def query_planetarynebulae(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/GALACTIC_PNe.fit')[1].data)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)
        catnam = [_['Name'].decode() for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'Planetary Nebulae'

    def query_ic_ngc(self,coord_polygon_query):
        # import data
        f = np.array(fits.open('mapext/catalogues/IC_NGC.fit')[1].data)
        print(f.dtype.names)
        glat = f['_Glat'].astype(float)
        glon = f['_Glon'].astype(float)

        def rtnCat(_):
            if _ == 'I':
                return 'IC'
            elif _ == 'N':
                return 'NGC'
            else:
                return ''

        catnam = [rtnCat(_['Cat'])+str(_['NGC_IC']) for _ in f]

        # work out which are in the query area
        isin = list(coord_polygon_query.contains_points(np.array([glon,glat]).T))
        # output the list
        outlist = list([list(compress(glon,isin)),
                        list(compress(glat,isin)),
                        list(compress(catnam,isin))])
        outlist = [tuple([row[i] for row in outlist]) for i in range(len(outlist[0]))]
        outlist = np.array(outlist,
                dtype=[('Glon',float),('Glat',float),('Name','S50')])
        return outlist, 'IC & NGC Objects'
