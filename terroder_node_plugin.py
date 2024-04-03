import os

import maya.api.OpenMaya as om
import maya.cmds as cmds
import maya.mel as mm

import numpy as np
from PIL import Image

# USE PYTHON API 2.0
maya_useNewAPI = True

# Useful functions for declaring attributes as inputs or outputs.
def MAKE_INPUT(attr):
    attr.keyable = True
    attr.storable = True
    attr.readable = True
    attr.writable = True
    
def MAKE_OUTPUT(attr):
    attr.keyable = False
    attr.storable = False
    attr.readable = True
    attr.writable = False

# Give the node a unique ID. Make sure this ID is different from all of your
# other nodes!
nodeId = om.MTypeId(0x8878)

class TerroderSimulationParameters(object):
    # All parameters that can be changed by the user

    def __init__(self):
        self.cellSize = 0.1
        self.gridShape = (100, 100)
        self.numIterations = 5
        self.upliftMapFile = ""
        self.upliftScale = 0.05  # scale uplift from [0, 1] to [0, this]
        self.erosionScale = 0.01
        self.minHeight = 0.0
        self.maxHeight = 5.0
        self._cachedUpliftMap = None
        self._cachedUpliftMapFile = ""
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TerroderSimulationParameters):
            return False
        
        return self.upliftMapFile == other.upliftMapFile \
            and self.gridShape == other.gridShape \
            and self.numIterations == other.numIterations \
            and abs(self.upliftScale - other.upliftScale) <= 0.0001 \
            and abs(self.erosionScale - other.erosionScale) <= 0.0001 \
            and abs(self.minHeight - other.minHeight) <= 0.001 \
            and abs(self.maxHeight - other.maxHeight) <= 0.001 
    
    @property
    def upliftMap(self):
        if self._cachedUpliftMap is not None and self._cachedUpliftMapFile == self.upliftMapFile:
            return self._cachedUpliftMapFile
        if len(self.upliftMapFile) == 0:
            return np.zeros(self.gridShape)

        self._cachedUpliftMapFile = ""
        self._cachedUpliftMap = None

        # Open the file and read in the uplift map
        try:
            with Image.open(self.upliftMapFile) as im:
                self._cachedUpliftMapFile = self.upliftMapFile
                self._cachedUpliftMap = np.zeros(self.gridShape)
                step = (float(im.size[0]) / float(self.gridShape[0]), float(im.size[1]) / float(self.gridShape[1]))
                for i in range(self.gridShape[0]):
                    for k in range(self.gridShape[1]):
                        # Interpolate (i, k)
                        rx, ry = step[0] * i, step[1] * k
                        self._cachedUpliftMap[i][k] = TerroderSimulationParameters.readInterpolatedUplift(im, (rx, ry))
        except FileNotFoundError:
            return np.zeros(self.gridShape)
        
        return self._cachedUpliftMap
    
    @staticmethod
    def readInterpolatedUplift(image: Image, coords):
        x, y = coords
        fx, fy = int(np.floor(x)), int(np.floor(y))
        uplifts = []
        weights = []
        for dx in range(0, 2):
            for dy in range(0, 2):
                if not (0 <= fx + dx < image.size[0] and 0 <= fy + dy < image.size[1]):
                    continue

                weight = max(0, (1 - abs(fx + dx - x)) * (1 - abs(fy + dy - y)))
                weights.append(weight)
                color = image.getpixel((fx + dx, fy + dy))
                uplifts.append(max(color[0], color[1], color[2]))
        
        totalWeight = sum(weights)
        if totalWeight <= 0:
            return 0
        
        return sum([weights[i] * uplifts[i] / totalWeight for i in range(len(uplifts))])
    

class TerroderNode(om.MPxNode):
    # constants
    TYPE_NAME = "terroder"
    NEIGHBOR_ORDER = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # SLOPE_NORM_EXPONENT = 4.0
    STEEPEST_SLOPE_EXPONENT = 2.0
    DRAIN_AREA_EXPONENT = 1.0

    # Declare class variables:
    # input
    upliftMapFile = om.MObject()
    iterations = om.MObject()
    cellSize = om.MObject()
    gridSizeX = om.MObject()
    gridSizeZ = om.MObject()

    # output
    outputMesh = om.MObject()

    def __init__(self):
        om.MPxNode.__init__(self)

        self.simParams = None
        # parmaeters fixed for now (but maybe not later)
        self.initialHeight = 1.0

        # variables used during the command execution
        self.numIterationsDone = 0
        self.heightMap = None
    
    def cellInBounds(self, cell) -> bool:
        return 0 <= cell[0] < self.simParams.gridShape[0] and 0 <= cell[1] < self.simParams.gridShape[1]
    
    def getNeighborCells(self, cell):
        neighborCells = []
        i, k = cell
        for di, dk in TerroderNode.NEIGHBOR_ORDER:
            ni, nk = i + di, k + dk
            if self.cellInBounds((ni, nk)):
                neighborCells.append((ni, nk))
        return neighborCells
    
    def compute(self, plug, dataBlock: om.MDataBlock):
        # get the input data
        self.simParams = TerroderSimulationParameters()
        self.simParams.upliftMapFile = dataBlock.inputValue(TerroderNode.upliftMapFile).asString()
        self.simParams.cellSize = dataBlock.inputValue(TerroderNode.cellSize).asDouble()
        self.simParams.gridShape = (dataBlock.inputValue(TerroderNode.gridSizeX).asInt(), dataBlock.inputValue(TerroderNode.gridSizeZ).asInt())
        self.simParams.numIterations = dataBlock.inputValue(TerroderNode.iterations).asInt()

        self.heightMap = self.makeInitialHeightMap()
        for _ in range(self.simParams.numIterations):
            pass

        # Set the output data
        outputMeshHandle: om.MDataHandle = dataBlock.outputValue(TerroderNode.outputMesh)

        self.runSimulation()
        # self.createOutputMesh()
        outputMeshHandle.setMObject(self.createOutputMesh())
        outputMeshHandle.setClean()
        dataBlock.setClean(plug)
        # Node doesn't have a setResult method
        # self.setResult("[DEBUG] Executed command")
    
    def makeInitialHeightMap(self):
        shape = self.simParams.gridShape
        flatMid = np.full(shape, self.initialHeight)
        randomness = (2. * np.random.random_sample(shape) - 1.) * (self.cellSize)
        return flatMid + randomness

    def runIteration(self):
        # Compute steepest slope to a lower neighbor
        steepestSlope = np.zeros(self.simParams.gridShape)  # 0 if no lower neighbor
        for i in range(self.simParams.gridShape[0]):
            for k in range(self.simParams.gridShape[1]):
                height = self.heightMap[i][k]
                for ni, nk in self.getNeighborCells((i, k)):
                    neighborHeight = self.heightMap[ni][nk]
                    if neighborHeight >= height:
                        continue

                    xDist = (ni - i) * self.simParams.cellSize
                    zDist = (nk - k) * self.simParams.cellSize
                    neighborDist = np.sqrt(xDist * xDist + zDist * zDist)
                    slope = (height - neighborHeight) / neighborDist
                    if slope > steepestSlope[i][k]:
                        steepestSlope[i][k] = slope

        self.drainageAreaMap = self.makeDrainageAreaMap()

        # Equals 1 at steepest slope 1 and drain area 1
        erosion = np.power(steepestSlope, TerroderNode.STEEPEST_SLOPE_EXPONENT) * np.power(self.drainageAreaMap, TerroderNode.DRAIN_AREA_EXPONENT)

        self.heightMap += self.simParams.upliftMap * self.simParams.upliftScale
        self.heightMap -= erosion * self.simParams.erosionScale
        self.heightMap = np.clip(self.heightMap, self.simParams.minHeight, self.simParams.maxHeight)  # clip height map
        self.numIterationsDone += 1

    # populates self.drainageArea
    def makeDrainageAreaMap(self) -> np.ndarray:
         # sort by descending height
        cellHeights = []
        for i in range(self.gridShape[0]):
            for k in range(self.gridShape[1]):
                cellHeights.append((i, k, self.heightMap[i][k]))
        cellHeights.sort(key = lambda ch: -ch[2]) 

        drainageArea = np.ones(self.gridShape)
        for i, k, h in cellHeights:
            neighborCells = self.getNeighborCells((i, k))
            relFlows = {}
            totalRelFlow = 0.0
            for ni, nk in neighborCells:
                nh = self.heightMap[ni][nk]
                if nh >= h:
                    continue
                
                di, dk = ni - i, nk - k
                relFlow = pow((h - nh) / np.sqrt(di * di + dk * dk), 4) # only need to compute proportionally due to normalization later
                relFlows[(ni, nk)] = relFlow
                totalRelFlow += 1

            if len(relFlows) == 0 or totalRelFlow < 0.001:
                continue

            for ni, nk in relFlows:
                drainageArea[ni][nk] += drainageArea[i][k] * relFlows[(ni, nk)] / totalRelFlow

        return drainageArea
    
    def runSimulation(self):
        if self.upliftMap is None:
            raise RuntimeError("The uplift map hasn't been created.")
        self.heightMap = self.makeInitialHeightMap()
        self.drainageAreaMap = np.ones(self.upliftMap.shape)
        for _ in range(self.numIters):
            self.runIteration()

    def createOutputMesh(self):
        if self.heightMap is None:
            raise RuntimeError("The height map hasn't been created.")

        # outputPoints will have the (x, y, z) point for each cell in self.heightMap
        outputPoints = np.empty((self.heightMap.shape[0], self.heightMap.shape[1], 3))
        for i in range(self.heightMap.shape[0]):
            x = self.interpolateX(float(i) / float(self.heightMap.shape[0] - 1))
            for k in range(self.heightMap.shape[1]):
                z = self.interpolateZ(float(k) / float(self.heightMap.shape[1] - 1))
                outputPoints[i][k][0] = x
                outputPoints[i][k][1] = self.heightMap[i][k]
                outputPoints[i][k][2] = z

        mergeTolerance = self.cellSize / 2.1
        outputMesh = om.MFnMesh()
        for i in range(outputPoints.shape[0] - 1):
            for k in range(outputPoints.shape[1] - 1):
                a = om.MPoint([outputPoints[i][k][d] for d in range(3)])
                b = om.MPoint([outputPoints[i+1][k][d] for d in range(3)])
                c = om.MPoint([outputPoints[i+1][k+1][d] for d in range(3)])
                d = om.MPoint([outputPoints[i][k+1][d] for d in range(3)])
                outputMesh.addPolygon([a, b, c, d], True, mergeTolerance)
        
        return outputMesh

    # ROUGHLY from https://stackoverflow.com/a/76686523
    # TODO: inspect for correctness
    def makeHeightMapLaplacian(self):
        grad_x, grad_z = np.gradient(self.heightMap, self.xStep, self.zStep)
        grad_xx = np.gradient(grad_x, self.xStep, axis=0)
        grad_zz = np.gradient(grad_z, self.zStep, axis=1)
        return grad_xx + grad_zz

    @staticmethod
    def initializer():
        nAttr = om.MFnNumericAttribute()
        tAttr = om.MFnTypedAttribute()
        
        # input
        TerroderNode.upliftMapFile = tAttr.create("upliftMapFile", "uf", om.MFnNumericData.kString)
        MAKE_INPUT(tAttr)
        tAttr.usedAsFilename = True
        
        TerroderNode.iterations = nAttr.create("iterations", "i", om.MFnNumericData.kInt, 5)
        MAKE_INPUT(nAttr)

        TerroderNode.cellSize = nAttr.create("cellSize", "cs", om.MFnNumericData.kFloat, 0.1)
        MAKE_INPUT(nAttr)

        TerroderNode.gridSizeX = nAttr.create("gridSizeX", "gsx", om.MFnNumericData.kInt, 100)
        MAKE_INPUT(nAttr)

        TerroderNode.gridSizeZ = nAttr.create("gridSizeZ", "gsz", om.MFnNumericData.kInt, 100)
        MAKE_INPUT(nAttr)

        # output
        TerroderNode.outputMesh = tAttr.create("outputMesh", "om", om.MFnData.kMesh)
        MAKE_OUTPUT(tAttr)

        # Add the attributes to the node and set up the
        #         attributeAffects (addAttribute, and attributeAffects)
        # Don't do try/except here, otherwise the error message won't be printed
        om.MPxNode.addAttribute(TerroderNode.upliftMapFile)
        om.MPxNode.addAttribute(TerroderNode.iterations)
        om.MPxNode.addAttribute(TerroderNode.cellSize)
        om.MPxNode.addAttribute(TerroderNode.gridSizeX)
        om.MPxNode.addAttribute(TerroderNode.gridSizeZ)
        om.MPxNode.addAttribute(TerroderNode.outputMesh)

        om.MPxNode.attributeAffects(TerroderNode.upliftMapFile, TerroderNode.outputMesh)
        om.MPxNode.attributeAffects(TerroderNode.iterations, TerroderNode.outputMesh)
        om.MPxNode.attributeAffects(TerroderNode.cellSize, TerroderNode.outputMesh)
        om.MPxNode.attributeAffects(TerroderNode.gridSizeX, TerroderNode.outputMesh)
        om.MPxNode.attributeAffects(TerroderNode.gridSizeZ, TerroderNode.outputMesh)

        print("TerroderNode initialised.\n")
    
    @staticmethod
    def creator():
        return TerroderNode()

# initialize the script plug-in
def initializePlugin(mobject):
    mplugin = om.MFnPlugin(mobject, "Company", "1.0", "Any")
    # Don't use try except bc otherwise the error message won't be printed
    mplugin.registerNode(TerroderNode.TYPE_NAME, nodeId, TerroderNode.creator, TerroderNode.initializer, om.MPxNode.kDependNode)

    melPath = f"{mplugin.loadPath()}/TerroderMenu.mel"
    if os.path.exists(melPath):
        mm.eval(f"source \"{melPath}\";")
        print(f"Loaded the script {melPath}.")
    else:
        print(f"Could not find the script in {melPath}.")

# uninitialize the script plug-in
def uninitializePlugin(mobject):
    mplugin = om.MFnPlugin(mobject)
    # Don't use try except bc otherwise the error message won't be printed
    mplugin.deregisterNode( nodeId )

    # Make sure menuName matches what written in TerroderMenu.mel
    menuName = "TerroderMenu"
    mm.eval(f"deleteUI ${menuName};")
