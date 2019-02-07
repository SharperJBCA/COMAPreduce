# Classes describing the focal plane.
# Interfaces with Pointing classes (e.g., FillPointing)
import numpy as np
import sys
import os

class FocalPlane(object):

    def __init__(self):
        
        self.p = 0.1853 # arcmin mm^-1, inverse of effective focal length

        self.azoff = 0
        self.eloff = 0

        self.theta = np.pi/2.*1.
        self.Rot   = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                               [-np.sin(self.theta),-np.cos(self.theta)]])

        # Which pixels and sidebands?
        self.pixels = np.arange(19).astype(int) + 1
        feedpositions = np.loadtxt('COMAP_FEEDS.dat', ndmin=1)
        self.offsets = []

        for k, f in enumerate(feedpositions):
            self.offsets += [(self.Rot.dot(f[1:,np.newaxis])).flatten()/60.*self.p]
