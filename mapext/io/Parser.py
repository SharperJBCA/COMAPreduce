import numpy as np

class Parser():

    def __init__(self, filename):

        with open(filename,'r') as f:
            lines = f.readlines()
        lines = [_.strip() for _ in lines]

        self.obj = []

        for _line, line in enumerate(lines):
            if _line <= 1:
                pass
            elif _line == 2:
                self.coordsys = line
            else:
                l = self.parse_ds9_obj(line)
                self.obj.append(tuple(l))
        self.obj = np.asarray(self.obj, dtype = [('type','S10'),('x',float),('y',float),('r',float),('fstring','S100')])

        xpoints = list(self.obj['x'])
        ypoints = list(self.obj['y'])
        coords = np.zeros([len(xpoints)+1,2])
        coords[:-1,0] = xpoints
        coords[-1,0] = xpoints[0]
        coords[:-1,1] = ypoints
        coords[-1,1] = ypoints[0]
        self.coords = coords
        del coords

    def parse_ds9_obj(self,line):
        if line.startswith('point'):
            type='point'
            x = line.split('(')[1].split(')')[0].split(',')[0]
            y = line.split('(')[1].split(')')[0].split(',')[1]
            r = np.nan
            fstring = line.split('#')[1][1:]
            return [type,x,y,r,fstring]
