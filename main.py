import math
import random
import sys
import imageio
import numpy
from progress.bar import Bar
import traceback
import argparse

class Colors:
    def __init__(self):
        self.colorMap = set()
        self.colorToValue = {}
        self.C = len(self.colorMap)

    def addColor(self, color):
        self.colorMap.add(str(color))
        if ( len(self.colorMap) > self.C ):
            self.colorToValue[str(color)] = self.C
        self.C = len(self.colorMap)

    def getColorValue(self, color):
        return self.colorToValue[str(color)]

    def getColorFromValue(self, value):
        keylen = 3
        for key in self.colorToValue:
            keylen = len(key.split(":"))
            if ( self.colorToValue[key] == value ):
                return key
        if ( keylen == 4 ):
            return "255:0:0:255"
        return "255:0:0"

    def getNumberOfColors(self):
        return self.C


class OverlappingModel:

    def __init__(self, colors, outX, outY, world, N=2, symmetry=8, periodic=False, keepup=False, keepdown=False):
        self.N = N
        self.periodic = periodic
        self.colors = colors
        self.C = colors.getNumberOfColors()
        self.world = world
        self.inX = len(world)
        self.inY = len(world[0])
        self.outX = outX
        self.outY = outY
        self.keepup = keepup
        self.keepdown = keepdown
    
        print("N: {}, Symmetry: {}".format(N,symmetry))
        self.patterns = []

        def pattern(N, f):
            p = []
            for x in range(N):
                p.append([])
                for y in range(N):
                    p[x].append(f(x,y))
            return p

        pattern_from_world = lambda x, y: pattern(self.N, lambda dx, dy: self.world[(x+dx) if (x+dx) < self.inX else self.inX - 1][(y+dy) if (y+dy) < self.inY else self.inY-1] ) 
        rotate = lambda array: pattern(self.N, lambda x, y: array[self.N - y - 1][x])
        reflect = lambda array: pattern(self.N, lambda x, y: array[x][self.N - y - 1])

        self.weights = {}
        for x in range(0, self.inX - self.N + 1 ):
            for y in range(0, self.inY - self.N + 1):
                patterns = []
                    
                patterns.append(pattern_from_world(x,y))
                patterns.append(reflect(patterns[0]))
                patterns.append(rotate(patterns[0]))
                patterns.append(reflect(patterns[2]))
                patterns.append(rotate(patterns[2]))
                patterns.append(reflect(patterns[4]))
                patterns.append(rotate(patterns[4]))
                patterns.append(reflect(patterns[6]))
                
                for i in range(symmetry):
                    idx = self.patternToIndex(patterns[i])
                    key = str(idx)
                    if ( key in self.weights ):
                        self.weights[key] += 1
                    else:
                        self.weights[key] = 1

        if keepdown or keepup:
            self.bottom_pattern = pattern_from_world(self.inX-N+1, 0)
            idx = self.patternToIndex(self.bottom_pattern)
            key = idx
            if ( not key in self.weights ):    
                self.weights[key] = 1
           
        #self.print_weights()
        #print(self.weights)

        self.patterns = []
        for key in self.weights:
            idx = key
            p = self.indexToPattern(idx)
            self.patterns.append(p)

        self.patternIndexWeight = {}
        self.patternIndexLogWeights = {}
        self.weightSumAllPatterns = 0.0
        self.logWeightSumAllPatterns = 0.0
        for i in range(len(self.patterns)):
            k = str(i)
            k2 = str(self.patternToIndex(self.patterns[i]))
            self.patternIndexWeight[k] = self.weights[k2]
            self.weightSumAllPatterns += self.weights[k2]
            self.patternIndexLogWeights[k] = math.log(self.weights[k2]) * self.weights[k2]
            self.logWeightSumAllPatterns += math.log(self.weights[k2]) * self.weights[k2]

        self.startingEntropy = math.log(self.weightSumAllPatterns) - self.logWeightSumAllPatterns / self.weightSumAllPatterns

        self.world_entropy = []
        self.world_options = []
        self.T = len(self.weights)

        self.bar = Bar('Collapsing', max=(outX*outY*self.T))
        self.D = 4 # down, right, up, left; the four directions
        self.dx = [-1, 0, 1, 0]
        self.dy = [0, 1, 0, -1]
        self.opposite = [2, 3, 0, 1]
        self.propagator = [] # Get patterns that match a combo: p1, p2, direction

        for d in range(self.D):
            self.propagator.append([])
            for t1 in range(self.T):
                self.propagator[d].append([])
                for t2 in range(self.T):
                    p1 = self.patterns[t1]
                    p2 = self.patterns[t2]
                    if self.patternAgrees(p1, p2, d):
                        self.propagator[d][t1].append(t2)
        self.wave = []
        self.compatible = []
        self.optionSpace = []
        self.totalOptionSpace = 0
        self.weightSum = []
        self.logWeightSum = []
        self.entropies = []
        for i in range(self.outX*self.outY):
            self.wave.append([])
            self.compatible.append([])
            self.optionSpace.append(self.T)
            self.totalOptionSpace += self.T
            self.weightSum.append(self.weightSumAllPatterns)
            self.logWeightSum.append(self.logWeightSumAllPatterns)
            self.entropies.append(self.startingEntropy)
            for t in range(self.T):
                self.wave[i].append(True)
                self.compatible[i].append([])
                for d in range(self.D):
                    self.compatible[i][t].append(len(self.propagator[self.opposite[d]][t]))

        self.currentOptionSpace = self.totalOptionSpace

        self.stack = []

        self.debugCount = 0

        if keepup or keepdown:
            p_up = pattern_from_world(0, 0)
            p_down = self.bottom_pattern

            i_p_up = self.patternToIndex(p_up)
            i_p_down = self.patternToIndex(p_down)
    
            t_p_up = None
            t_p_down = None

            for p in range(len(self.patterns)):
                pi = self.patternToIndex(self.patterns[p])
                if ( i_p_up == pi ):
                    t_p_up = p
                if ( i_p_down == pi ):
                    t_p_down = p

            # Constrain away all patterns from up/down that doesn't match up/down
            if keepup:
                x = 0
                while ( x < self.outX):
                    y = 0
                    while ( y < self.outY):
                        i = x + self.outX * y
                        if x == 0:
                            for t in range(len(self.patterns)):
                                if t != t_p_up:
                                    self.constrain(i, t)
                        y += self.N
                    x += self.N
                self.propagate()

            if keepdown:
                x = 0
                while ( x < self.outX):
                    y = 0
                    while ( y < self.outY):
                        i = x + self.outX * y
                        if x == (self.outX-self.N) and (y == 0 or y == self.outY-self.N):
                            for t in range(len(self.patterns)):
                                if t != t_p_down:
                                    self.constrain(i, t)
                        y += self.N
                    x += self.N
                self.propagate()


    def printPattern(self, p):
        s = ""
        for xx in range(self.N):
            for yy in range(self.N):
                s += str(p[xx][yy]) + " "
            s += "\n"
        print(s)



    def printWeights(self):

        for p in self.weights:
            value = self.weights[p]
            pattern = self.indexToPattern(p)
            s = ""
            for x in range(len(pattern)):
                s += "\n"
                for y in range(len(pattern[0])):
                    s += str(pattern[x][y]) + " "
            print("========================")
            print(s)
            print("\n")
            print("value: {}".format(value))
    
    def createWorld(self):

        # create world
        while ( not self.isCollapsed() ):
            index = self.getNextPoint()
            if ( index < 0 ):
                break
            self.observe(index)
            if not self.propagate():
                break
        self.bar.finish()

        # get world
        out_world = []
        for x in range(self.outX):
            out_world.append([])
            for y in range(self.outY):
                out_world[x].append(None)

        x = 0
        while ( x < self.outX):
            y = 0
            while ( y < self.outY):
                i = x + self.outX * y
                p = None # By default...
                for t in range(self.T):
                    s = sum([ 1 if x else 0 for x in self.wave[i] ])

                    if ( s >= 1 and self.wave[i][t] ):
                        p = self.patterns[t]
                    if ( s == 0):
                        raise Exception("Collapse failed, storing a debugging image: debug.png")
      
                for xx in range(self.N):
                    for yy in range(self.N):
                        if ( x + xx > self.outX - 1):
                            continue
                        if ( y + yy > self.outY - 1):
                            continue
                        out_world[x+xx][y+yy] = -1
                        if ( p != None ):
                            out_world[x+xx][y+yy] = p[xx][yy]
                        else:
                            raise Exception("Collapse failed")
                y += self.N
            x += self.N
        
        return out_world

    def observe(self, index):
        options = self.wave[index].copy()
        distribution = []
        for i in range(self.T):
            distribution.append( 0.0 if options[i] == False else self.patternIndexWeight[str(i)] ) 
        S = sum(distribution)
        if S > 0:
            for i in range(self.T):
                distribution[i] /= S
        r = numpy.random.choice(numpy.arange(0, self.T), p=distribution)
        for t in range(self.T):
            if ( options[t] != (t == r) ):
                self.constrain(index, t)

    def constrain(self, i, t):
        
        if ( self.optionSpace[i] == 1 ):
            return
        for d in range(self.D):
            self.compatible[i][t][d] = 0

        self.wave[i][t] = False
        self.stack.append((i,t))

        self.currentOptionSpace -= 1
        self.optionSpace[i] -= 1
        self.weightSum[i] -= self.patternIndexWeight[str(t)]
        self.logWeightSum[i] -= self.patternIndexLogWeights[str(t)]
        s = self.weightSum[i]
        self.entropies[i] = math.log(s) - self.logWeightSum[i] / s
        self.bar.next()

    def getUncertainWorld(self):
        # Get uncertain world
        out_world = []
        for x in range(self.outX):
            out_world.append([])
            for y in range(self.outY):
                out_world[x].append(0)

        x = 0
        while ( x < self.outX):
            y = 0
            while ( y < self.outY):
                i = x + self.outX * y
                p = None # By default...
                for t in range(self.T):
                    s = sum([ 1.0 if x else 0.0 for x in self.wave[i] ])
                    if ( self.wave[i][t] ):
                        p = self.patterns[t]

                for xx in range(self.N):
                    for yy in range(self.N):
                        if ( x + xx > self.outX - 1):
                            continue
                        if ( y + yy > self.outY - 1):
                            continue
                        out_world[x+xx][y+yy] = -1
                        if ( p != None ):
                            out_world[x+xx][y+yy] = p[xx][yy]
                y += self.N
            x += self.N

        return out_world 

    def isCollapsed(self):
        return self.currentOptionSpace <= (self.outX * self.outY) + 1 

    def getNextPoint(self):
        minIndex = 0
        minEntropy = 10000
        noneFound = True
        for i in range(len(self.wave)):
            if ( self.optionSpace[i] == 1 ):
                continue
            e = self.entropies[i] 
            if ( self.optionSpace[i] > 1 and e < minEntropy ):
                n = 0 # 0.000001 * random.random()
                if ( e + n < minEntropy ):
                    minEntropy = e
                    minIndex = i
                    noneFound = False
        if noneFound:
            return -1
        return minIndex

    def isWithing(self, maxX, maxY, i):
        x = i % maxX
        y = int(i/maxX)

        if x < 0 or x >= (maxX):
            return False
        if y < 0 or y >= (maxY):
            return False
        return True

    def getCurrentProgress(self):
        total = self.totalOptionSpace
        progNow = self.currentOptionSpace
        goal = self.outX * self.outY

        distanceToGo = total - goal
        soFar = total - progNow

        return ((100.0 / distanceToGo) * soFar, progNow, total, goal)

    def propagate(self):

        while ( len(self.stack) > 0 and not self.isCollapsed() ):
            i1, t1 = self.stack.pop()

            x1 = i1 % self.outX
            y1 = int(i1 / self.outX)

            for d in range(self.D):
                x2 = x1 + self.dx[d]
                y2 = y1 + self.dy[d]

                if (not self.periodic) and (x2 < 0 or y2 < 0 or x2 + self.N > self.outX or y2 + self.N > self.outY):
                    continue

                if (x2 < 0):
                    x2 += self.outX
                elif (x2 >= self.outX):
                    x2 -= self.outX

                if (y2 < 0):
                    y2 += self.outY
                elif (y2 >= self.outY):
                    y2 -= self.outY

                i2 = x2 + y2 * self.outX

                possibles = self.propagator[d][t1]
                compatibilities = self.compatible[i2]

                for p in range(len(possibles)):
                    t2 = possibles[p]
                    comp = compatibilities[t2]

                    comp[d] -= 1
                    if ( comp[d] == 0 ):
                        self.constrain(i2, t2)

                    #p = self.getCurrentProgress()
                    #print("\rProgress {:.2f}%  ({}, {}, {})".format(p[0], p[1], p[2], p[3]), end="")

        return self.optionSpace[0] > 0

    def patternAgrees(self, p1, p2, direction):
        # Make keys, don't check more than once
        dx = self.dx[direction]
        dy = self.dy[direction]

        xstart = (0 if dx < 0 else dx)
        xstop = (self.N + dx if dx < 0 else self.N)
        ystart = (dy if dy >= 0 else 0)
        ystop = (self.N if dy >= 0 else self.N + dy)

        for x in range(xstart, xstop):
            for y in range(ystart, ystop):
                if ( p1[x][y] != p2[x-dx][y-dy] ):
                    return False
        return True

    def patternToIndex(self, p):
        # 0 < x < N
        # 0 < y < N
        s = ""
        for x in range(0, self.N):
            for y in range(0, self.N):
                if y > 0:
                    s += ","
                s += str(p[x][y])
            s += ";"
        return s 

    def indexToPattern(self, idx):
        patternArray = []
        for x in idx.split(";"):
            if ( len(x) > 0) :
                patternArray.append([int(i) for i in x.split(",")])        
        return patternArray

class ImageHandler:

    def getWorldFromImage(self, image_path, scale=1):
        self.colors = Colors()
        im = imageio.v2.imread(image_path)
        self.world = []
        dimensions = im.shape
        
        for x in range(dimensions[0]*scale):
            self.world.append([])
            for y in range(dimensions[1]*scale):
                self.world[x].append(0)

        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                value = ""
                for z in range(dimensions[2]):
                    if z > 0:
                        value += ":"    
                    value += str(im[x][y][z])
                self.colors.addColor(value)
                for xx in range(scale):
                    for yy in range(scale):
                        self.world[x*scale+xx][y*scale+yy] = self.colors.getColorValue(value)

        return self.world

    def getColors(self):
        return self.colors
    
    def createImageFromWorld(self, world, scale, out_path):
        dimensions = (len(world), len(world[0]), len(self.colors.getColorFromValue(world[0][0]).split(":")))
        outndarray = numpy.zeros((dimensions[0]*scale, dimensions[1]*scale, dimensions[2]),dtype=numpy.uint8)
        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                v = self.colors.getColorFromValue(world[x][y]).split(":")
                for z in range(dimensions[2]):
                    for xx in range(scale):
                        for yy in range(scale):
                            try:
                                outndarray[x*scale+xx][y*scale+yy][z] = v[z]
                            except IndexError as ie:
                                print("outndarray[{}][{}][{}], dims: {}, v: {}, z: {}".format(
                                    x*scale+xx, y*scale+yy, z, outndarray.shape, v, z))
                                raise ie
        imageio.v2.imwrite(out_path, outndarray)
    
    def createImageFromUncertainWorld(self, world, colors, out_path):
        dimensions = (len(world), len(world[0]), 3) 
        outndarray = numpy.zeros(dimensions,dtype=numpy.uint8)
        for x in range(dimensions[0]):
            for y in range(dimensions[1]):
                v = [255,0,0]
                if ( len(world[x][y]) == 1 ):

                    v = colors.getColorFromValue(world[x][y][0]).split(":")
                for z in range(len(v)):
                    outndarray[x][y][z] = v[z]
        imageio.v2.imwrite(out_path, outndarray)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate pixelimages')

    parser.add_argument('-n', help='Overlapping window size (feature generator)',
            type=int, default=3)
    parser.add_argument('-s','--symmetry', help='Symmetry value (1-8). 1: no symmetry, 8: all symmetries',
            choices=range(1,9),
            type=int, default=1)
    parser.add_argument('-o','--output', help='Output file (.png)',
            type=str, default='out.png')
    parser.add_argument('-d','--dimension', help='Output dimension WidthxHeight, e.g. 80x60',
            default='80x60', type=str)
    parser.add_argument('-i','--input', help='Input image, see the "samples" folder',
            default='samples/Flowers.png', type=str)
    parser.add_argument('--iscale', help='Input image scaling',
            default=1, type=int)
    parser.add_argument('--oscale', help='Output image scaling',
            default=50, type=int)
    parser.add_argument('--nostrip', help='Don\'t strip output size to factor of nearest feature window size (N) multiple', action='store_false')
    parser.add_argument('--keepup', help='Mirror upper parts of the image with the original', action='store_true')
    parser.add_argument('--keepdown', help='Mirror lower parts of the image with the original', action='store_true')

    args = parser.parse_args()
    strip = args.nostrip
    N = args.n
    S = args.symmetry
    outputSize = args.dimension
    outputFile = args.output
    inputFile = args.input
    inputScale = args.iscale
    outputScale = args.oscale
    keepup = args.keepup
    keepdown = args.keepdown

    if ( len(outputSize.split("x")) != 2):
        parser.print_help()
        sys.exit(1)

    outputX = int(outputSize.split("x")[1])
    outputY = int(outputSize.split("x")[0])
    
    if strip:
        outputX = int(outputX/N) * N
        outputY = int(outputY/N) * N

    if len(sys.argv) >= 2:
        ih = ImageHandler()
        world = ih.getWorldFromImage(inputFile, scale=inputScale)
        colors = ih.getColors()

        om = OverlappingModel(colors, outputX, outputY, world, N, S, False, keepup, keepdown)
        try:
            out_world = om.createWorld()
            ih.createImageFromWorld(out_world, outputScale, outputFile)
        except Exception as e:
            out_world_bad = om.getUncertainWorld()
            print(e)
            traceback.print_exc()
            ih.createImageFromWorld(out_world_bad, outputScale, "debug.png")

    
