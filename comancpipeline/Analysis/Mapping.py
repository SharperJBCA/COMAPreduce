from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types

from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS
import h5py
import numpy as np
import matplotlib.pyplot as plt   
from astropy.io import fits
import sys
from astropy.wcs import WCS
import os
import argparse
import configparser
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot

class SimpleMap(object):
    
    def __init__(self, xpix=100, ypix=100, cdeltx=1, cdelty=1, ctypex='RA---TAN', ctypey='DEC--TAN', x0=0, y0=0, filtertod=False, lon=0, lat=0):
        self.xpix = int(xpix)
        self.ypix = int(ypix)
        self.cdeltx = float(cdeltx)/60.
        self.cdelty = float(cdelty)/60.
        self.x0 = int(x0)
        self.y0 = int(y0)

        self.ctype=[ctypex, ctypey]
        self.naxis = [self.xpix, self.ypix]
        self.cdelt = [self.cdeltx, self.cdelty]
        self.crval = [self.x0, self.y0]

        self.lon = lon
        self.lat = lat 


        self.filtertod = filtertod

    def __call__(self, container, filename=''):
        self.run(container)
        self.filename = filename

    def run(self,container):
        tod = container.getdset('spectrometer/tod')
        ra  = container.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = container.getdset('spectrometer/pixel_pointing/pixel_dec')
        mjd = container.getdset('spectrometer/MJD')
        el  = container.getdset('spectrometer/pixel_pointing/pixel_el')

        #create wcs
        self.crval = [np.nanmedian(ra), np.nanmedian(dec)]
        self.wcs,_,_ = DefineWCS(naxis=self.naxis, 
                                 cdelt=self.cdelt, 
                                 crval=self.crval,
                                 ctype=self.ctype)

        maps = self.MakeMap(tod, ra, dec, mjd, el)
        noisemap = self.MakeNoiseMap(tod, ra, dec, mjd, el)
        
        #save wcs info
        #container.setExtrasData('Mapping/WCS',
        #                        self.wcs,
        #                        [Types._OTHER_])
        #save map
        container.setExtrasData('Mapping/SimpleMaps', 
                                maps,
                                [Types._HORNS_, 
                                 Types._SIDEBANDS_, 
                                 Types._FREQUENCY_,
                                 Types._OTHER_, Types._OTHER_])
        #save noise map
        container.setExtrasData('Mapping/NoiseMaps', 
                                noisemap,
                                [Types._HORNS_, 
                                 Types._SIDEBANDS_, 
                                 Types._FREQUENCY_,
                                 Types._OTHER_, Types._OTHER_])

    def initialPeak(self,tod, x, y):
        
        rms  = Filtering.calcRMS(tod)
        r = np.sqrt((x)**2 + (y)**2)
        close = (r < 10)                      
        tod -= Filtering.estimateBackground(tod, rms, close)

        dx, dy = 1./60., 1./60.
        Dx, Dy = 1., 1.
        npix = int(Dx/dx)
        xpix, ypix = np.arange(npix+1), np.arange(npix+1)
        xpix = xpix*dx - Dx/2.
        ypix = ypix*dy - Dy/2.
        m = np.histogram2d(x, y, xpix, weights=tod)[0]/np.histogram2d(x, y, xpix)[0]
        m = median_filter(m, 3)
        xmax,ymax = np.unravel_index(np.nanargmax(m),m.shape)
        return xpix[xmax], ypix[ymax]


    def MakeMap(self, tod, ra, dec, mjd, el):
        #takes a 1D tod array and makes a simple map

        #produce arrays for mapping
        npix = self.naxis[0]*self.naxis[1]

        pixbins = np.arange(0, npix+1).astype(int)
        
        nHorns, nSBs, nChans, nSamples = tod.shape
        rms  = Filtering.calcRMS(tod)

        maps = np.zeros((nHorns, nSBs, nChans, self.naxis[0], self.naxis[1]))
        for i in range(nHorns):

            good = (np.isnan(ra[i,:]) == False) & (np.isnan(tod[i,0,0]) == False) 
            pa = Coordinates.pa(ra[i,good], dec[i,good], mjd[good], self.lon, self.lat)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.crval[0], self.crval[1], -pa)

            nbins = 10
            xbins = np.linspace(np.min(x),np.max(x), nbins+1)
            xmids = (xbins[1:] + xbins[:-1])/2.
            xbw, _ = np.histogram(x,xbins)
            ybw, _ = np.histogram(y,xbins)

            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)
            fitx0, fity0 = self.initialPeak(todAvg[good], x, y)
            r = np.sqrt((x-fitx0)**2 + (y-fity0)**2)
            close = (r < 6./60.)

            pix = ang2pixWCS(self.wcs, ra[i,good], dec[i,good]).astype('int')
            mask = np.where((pix != -1))[0]

            h, b = np.histogram(pix, pixbins, weights=(pix != -1).astype(float))
            self.hits = np.reshape(h, (self.naxis[0], self.naxis[1]))

            for j in range(nSBs):
                for k in range(nChans):
                    todmap = tod[i,j,k,good]

                    if self.filtertod:
                        txbw, _ = np.histogram(x,xbins, weights=todmap)
                        tybw, _ = np.histogram(y,xbins, weights=todmap)
                        fb = txbw/xbw
                        gd = np.isfinite(fb)
                        pmdl = np.poly1d(np.polyfit(xmids[gd],fb[gd],1))
                        todmap -= pmdl(x)
                        fb = tybw/ybw
                        gd = np.isfinite(fb)
                        pmdl = np.poly1d(np.polyfit(xmids[gd],fb[gd],1))
                        todmap -= pmdl(y)

                    w, b = np.histogram(pix[mask], pixbins, weights=todmap[mask])
#                    w, b = np.histogram(pix[:], pixbins, weights=tod[i,j,k,:])
                    m = np.reshape(w, (self.naxis[0], self.naxis[1]))
                    maps[i,j,k,...] = m/self.hits
            pyplot.subplot(projection=self.wcs)
            pyplot.imshow(maps[0,0,0,:,:])
            pyplot.show()
        return maps

    def MakeNoiseMap(self, tod, ra, dec, mjd, el):
        #takes a 1D tod array and makes a simple noise map

        #produce arrays for mapping
        npix = self.naxis[0]*self.naxis[1]

        pixbins = np.arange(0, npix+1).astype(int)       
        rms  = Filtering.calcRMS(tod)

        #get noise rms and associated ra and dec
        noise, ranew, dnew, mjdnew = Filtering.noiseProperties(tod,ra,dec,mjd)

        nHorns, nSBs, nChans, nSamples = noise.shape
        maps = np.zeros((nHorns, nSBs, nChans, self.naxis[0], self.naxis[1]))
        for i in range(nHorns):

            good = (np.isnan(ranew[i,:]) == False) & (np.isnan(noise[i,0,0]) == False) 
            pa = Coordinates.pa(ranew[i,good], dnew[i,good], mjdnew[good], self.lon, self.lat)
            x, y = Coordinates.Rotate(ranew[i,good], dnew[i,good], self.crval[0], self.crval[1], -pa)
            nbins = 10
            xbins = np.linspace(np.min(x),np.max(x), nbins+1)
            xmids = (xbins[1:] + xbins[:-1])/2.
            xbw, _ = np.histogram(x,xbins)
            ybw, _ = np.histogram(y,xbins)

            noiseAvg = np.nanmean(np.nanmean(noise[i,...],axis=0),axis=0)
            fitx0, fity0 = self.initialPeak(noiseAvg[good], x, y)
            r = np.sqrt((x-fitx0)**2 + (y-fity0)**2)
            close = (r < 6./60.)

            pix = ang2pixWCS(self.wcs, ranew[i,good], dnew[i,good]).astype('int')
            mask = np.where((pix != -1))[0]

            h, b = np.histogram(pix, pixbins, weights=(pix != -1).astype(float))
            self.hits = np.reshape(h, (self.naxis[0], self.naxis[1]))

            for j in range(nSBs):
                for k in range(nChans):
                    noisemap = noise[i,j,k,good]

                    if self.filtertod:
                        txbw, _ = np.histogram(x,xbins, weights=noisemap)
                        tybw, _ = np.histogram(y,xbins, weights=noisemap)
                        fb = txbw/xbw
                        gd = np.isfinite(fb)
                        pmdl = np.poly1d(np.polyfit(xmids[gd],fb[gd],1))
                        noisemap -= pmdl(x)
                        fb = tybw/ybw
                        gd = np.isfinite(fb)
                        pmdl = np.poly1d(np.polyfit(xmids[gd],fb[gd],1))
                        noisemap -= pmdl(y)

                    w, b = np.histogram(pix[mask], pixbins, weights=noisemap[mask])
#                    w, b = np.histogram(pix[:], pixbins, weights=noise[i,j,k,:])
                    m = np.reshape(w, (self.naxis[0], self.naxis[1]))
                    maps[i,j,k,...] = m/self.hits

        return maps



    def WriteMain(self, finalMaps, obsid):
        #writes map data to .fits file along with wcs for scaling
        header = self.wcs.to_header()
        hdu = fits.PrimaryHDU(finalMaps, header=header)
        hdu2 = fits.ImageHDU(self.hits)
        hdul = fits.HDUList()
        hdul.append(hdu)
        hdul.append(hdu2)
        hdul.writeto(obsid + '/map.fits')

    def WriteNoise(self, noisemap, obsid):
        header = self.wcs.to_header()
        hdu = fits.PrimaryHDU(noisemap, header=header)
        hdu2 = fits.ImageHDU(self.hits)
        hdul = fits.HDUList()
        hdul.append(hdu)
        hdul.append(hdu2)
        hdul.writeto(obsid + '/noisemap.fits')

    def Quicklook(self, finalMaps):
        #make a plot on maps calculated in this file
        plt.imshow(finalMaps, origin='lower', cmap=plt.cm.viridis)
        plt.show()

    def PlotMapFromFile(self, filename=''):
        #read in .fits file and plots image
        self.filename = filename
        hdu = fits.open(filename)
        wcs = WCS(hdu[0].header)

        fig = plt.figure()
        fig.add_subplot(111, projection=wcs)
        plt.imshow(hdu[0].data, origin = 'lower', cmap=plt.cm.viridis)
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.colorbar()
        plt.show()


class SimpleMapCentred(SimpleMap):

    def getJupiter(self, data):
        mjd = data.getdset('spectrometer/MJD')
        self.x0, self.y0, self.dist = Coordinates.getPlanetPosition('Jupiter', self.lon, self.lat, mjd)
        return self.x0, self.y0, self.dist


    def run(self,container):
        tod = container.getdset('spectrometer/tod')
        ra  = container.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = container.getdset('spectrometer/pixel_pointing/pixel_dec')
        mjd = container.getdset('spectrometer/MJD')
        el  = container.getdset('spectrometer/pixel_pointing/pixel_el')

        #create wcs
        self.x0, self.y0, self.dist = self.getJupiter(container)

        self.crval = [0,0]
        self.wcs,_,_ = DefineWCS(naxis=self.naxis, 
                                 cdelt=self.cdelt, 
                                 crval=self.crval,
                                 ctype=self.ctype)

        maps = self.MakeMap(tod, ra, dec, mjd, el)
        container.setExtrasData('Mapping/SimpleMaps', 
                                maps,
                                [Types._HORNS_, 
                                 Types._SIDEBANDS_, 
                                 Types._FREQUENCY_,
                                 Types._OTHER_, Types._OTHER_])

    def MakeMap(self, tod, ra, dec, mjd, el):
        #takes a 1D tod array and makes a simple map

        #produce arrays for mapping
        npix = self.naxis[0]*self.naxis[1]

        pixbins = np.arange(0, npix+1).astype(int)
        
        nHorns, nSBs, nChans, nSamples = tod.shape
        rms  = Filtering.calcRMS(tod)

        maps = np.zeros((nHorns, nSBs, nChans, self.naxis[0], self.naxis[1]))
        for i in range(nHorns):

            good = (np.isnan(ra[i,:]) == False) & (np.isnan(tod[i,0,0]) == False) 
            pa = Coordinates.pa(ra[i,good], dec[i,good], mjd[good], self.lon, self.lat)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, -pa)

            nbins = 10
            xbins = np.linspace(np.min(x),np.max(x), nbins+1)
            xmids = (xbins[1:] + xbins[:-1])/2.
            xbw, _ = np.histogram(x,xbins)
            ybw, _ = np.histogram(y,xbins)

            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)
            fitx0, fity0 = self.initialPeak(todAvg[good], x, y)
            r = np.sqrt((x-fitx0)**2 + (y-fity0)**2)
            close = (r < 6./60.)

            pix = ang2pixWCS(self.wcs, x, y).astype('int')
            mask = np.where((pix != -1))[0]


            h, b = np.histogram(pix, pixbins, weights=(pix != -1).astype(float))
            self.hits = np.reshape(h, (self.naxis[0], self.naxis[1]))

            for j in range(nSBs):
                for k in range(1):#nChans):
                    todmap = tod[i,j,k,good]

                    if self.filtertod:
                        txbw, _ = np.histogram(x,xbins, weights=todmap)
                        tybw, _ = np.histogram(y,xbins, weights=todmap)
                        fb = txbw/xbw
                        gd = np.isfinite(fb)
                        pmdl = np.poly1d(np.polyfit(xmids[gd],fb[gd],1))
                        todmap -= pmdl(x)
                        fb = tybw/ybw
                        gd = np.isfinite(fb)
                        pmdl = np.poly1d(np.polyfit(xmids[gd],fb[gd],1))
                        todmap -= pmdl(y)

                    w, b = np.histogram(pix[mask], pixbins, weights=todmap[mask])
#                    w, b = np.histogram(pix[:], pixbins, weights=tod[i,j,k,:])
                    m = np.reshape(w, (self.naxis[0], self.naxis[1]))
                    maps[i,j,k,...] = m/self.hits

        return maps
