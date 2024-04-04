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
class TerroderSimulationParameters(object):
    # All parameters that can be changed by the user
    # Does NOT include time; if time alone changes, we might not need to redo the entire simulation

    def __init__(self):
        self.cellSize = 0.1
        self.gridShape = (50, 50)
        self.upliftScale = 0.01  # scale uplift from [0, 1] to [0, this]
        self.erosionScale = 0.1
        self.minHeight = 0.0
        self.maxHeight = 3.0
        self.averageHeight = 1.0  # will be enforced
        self._upliftMapFile = ""
        self._cachedUpliftMap = None
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TerroderSimulationParameters):
            return False
        
        return self.upliftMapFile == other.upliftMapFile \
            and self.gridShape == other.gridShape \
            and abs(self.upliftScale - other.upliftScale) <= 0.0001 \
            and abs(self.erosionScale - other.erosionScale) <= 0.0001 \
            and abs(self.minHeight - other.minHeight) <= 0.001 \
            and abs(self.maxHeight - other.maxHeight) <= 0.001 \
            and abs(self.averageHeight - other.averageHeight) <= 0.001
    
    @property
    def upliftMapFile(self):
        return self._upliftMapFile
    
    @upliftMapFile.setter
    def upliftMapFile(self, value):
        if self._upliftMapFile == value:
            return
        
        self._upliftMapFile = value
        self._cachedUpliftMap = None
        try:
            with Image.open(self.upliftMapFile) as image:
                gsImage = image.convert('L')  # grayscale
                step = (float(gsImage.size[0]) / float(self.gridShape[0]), float(gsImage.size[1]) / float(self.gridShape[1]))
                
                self._cachedUpliftMap = np.empty(self.gridShape)
                for i in range(self.gridShape[0]):
                    for k in range(self.gridShape[1]):
                        # Interpolate (i, k)
                        rx, ry = step[0] * i, step[1] * k
                        self._cachedUpliftMap[i][k] = TerroderSimulationParameters._readInterpolatedUplift(gsImage, (rx, ry))
            
            self._cachedUpliftMap = np.clip(self._cachedUpliftMap, 0., 1.)
        except FileNotFoundError:
            om.MGlobal.displayWarning(f'File "{value}" not found.')
    
    @property
    def upliftMap(self):
        return self._cachedUpliftMap if self._cachedUpliftMap is not None else np.full(self.gridShape, 0.5)
    
    @staticmethod
    def _readInterpolatedUplift(grayscaleImage: Image, coords):
        x, y = coords
        fx, fy = int(np.floor(x)), int(np.floor(y))
        uplifts = []
        weights = []
        for dx in range(0, 2):
            for dy in range(0, 2):
                if not (0 <= fx + dx < grayscaleImage.size[0] and 0 <= fy + dy < grayscaleImage.size[1]):
                    continue

                weight = max(0, (1 - abs(fx + dx - x)) * (1 - abs(fy + dy - y)))
                weights.append(weight)
                uplifts.append(grayscaleImage.getpixel((fx + dx, fy + dy)))
        
        totalWeight = sum(weights)
        if totalWeight <= 0:
            return 0
        
        return sum([weights[i] * uplifts[i] / totalWeight for i in range(len(uplifts))])
    

class TerroderNode(om.MPxNode):
    # constants
    TYPE_NAME = "TerroderNode"
    ID = om.MTypeId(0x0008099a)
    NEIGHBOR_ORDER = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # SLOPE_NORM_EXPONENT = 4.0
    STEEPEST_SLOPE_EXPONENT = 2.0
    DRAIN_AREA_EXPONENT = 1.0

    TIME_ATTR_LONG_NAME = "time"
    TIME_ATTR_SHORT_NAME = "t"
    OUTPUT_MESH_ATTR_LONG_NAME = "outputMesh"
    OUTPUT_MESH_ATTR_SHORT_NAME = "om"

    # Attributes
    # input
    aTime = None
    aUpliftMapFile = None
    aCellSize = None
    aGridSizeX = None
    aGridSizeZ = None

    # output
    aOutputMesh = None

    def __init__(self):
        om.MPxNode.__init__(self)

        self.simParams = None
        self.heightMapTs = []  # element i is the heightmap after i iterations
    
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
    
    def compute(self, plug: om.MPlug, dataBlock: om.MDataBlock):
        if (plug != TerroderNode.aOutputMesh) and (plug.parent() != TerroderNode.aOutputMesh):
            return None
        
        rawTime = dataBlock.inputValue(TerroderNode.aTime).asFloat()
        numIterations = max(0, int(rawTime) - 1)

        # get the input data
        newSimParams = TerroderSimulationParameters()
        newSimParams.upliftMapFile = dataBlock.inputValue(TerroderNode.aUpliftMapFile).asString()
        newSimParams.cellSize = dataBlock.inputValue(TerroderNode.aCellSize).asFloat()
        newSimParams.gridShape = (dataBlock.inputValue(TerroderNode.aGridSizeX).asInt(), dataBlock.inputValue(TerroderNode.aGridSizeZ).asInt())
        if self.simParams is None or newSimParams != self.simParams:
            # Current sim params are out of date; reset
            om.MGlobal.displayInfo("[DEBUG] Using new sim params and resetting simulation")
            self.simParams = newSimParams
            self.heightMapTs = []

        # Simulate until we have at least numIters iterations
        while numIterations >= len(self.heightMapTs):
            self.simulateNextIteration()

        # Set the output data
        outputMeshHandle: om.MDataHandle = dataBlock.outputValue(TerroderNode.aOutputMesh)
        dataCreator: om.MFnMeshData = om.MFnMeshData()
        outputData: om.MObject = dataCreator.create()
        self.createOutputMesh(outputData, numIterations)
        outputMeshHandle.setMObject(outputData)
        outputMeshHandle.setClean()
        
        # successfully computed
        return self

    def simulateNextIteration(self) -> None:
        if len(self.heightMapTs) == 0:
            self.heightMapTs.append(self.makeInitialHeightMap())
            return
        
        # Compute steepest slope to a lower neighbor
        curHeightMap = self.heightMapTs[-1]
        steepestSlope = np.zeros(self.simParams.gridShape)  # 0 if no lower neighbor
        for i in range(self.simParams.gridShape[0]):
            for k in range(self.simParams.gridShape[1]):
                height = curHeightMap[i][k]
                for ni, nk in self.getNeighborCells((i, k)):
                    neighborHeight = curHeightMap[ni][nk]
                    if neighborHeight >= height:
                        continue

                    xDist = (ni - i) * self.simParams.cellSize
                    zDist = (nk - k) * self.simParams.cellSize
                    neighborDist = np.sqrt(xDist * xDist + zDist * zDist)
                    slope = (height - neighborHeight) / neighborDist
                    if slope > steepestSlope[i][k]:
                        steepestSlope[i][k] = slope

        drainageAreaMap = self.computeDrainageAreaMap(curHeightMap)

        # Equals 1 at steepest slope 1 and drain area 1
        erosion = np.power(steepestSlope, TerroderNode.STEEPEST_SLOPE_EXPONENT) * np.power(drainageAreaMap, TerroderNode.DRAIN_AREA_EXPONENT)

        nextHeightMap = np.copy(curHeightMap)
        nextHeightMap += self.simParams.upliftMap * self.simParams.upliftScale
        nextHeightMap -= erosion * self.simParams.erosionScale
        nextHeightMap = self.applyHeightConstraints(nextHeightMap)
        self.heightMapTs.append(nextHeightMap)
    
    def makeInitialHeightMap(self) -> np.ndarray:
        shape = self.simParams.gridShape
        flatMid = np.full(shape, self.simParams.averageHeight)
        randomness = (2. * np.random.random_sample(shape) - 1.) * self.simParams.cellSize
        heightMap = flatMid + randomness
        return self.applyHeightConstraints(heightMap)

    def applyHeightConstraints(self, heightMap: np.ndarray) -> np.ndarray:
        newHeightMap = heightMap - np.mean(heightMap) + self.simParams.averageHeight
        newHeightMap = np.clip(newHeightMap, self.simParams.minHeight, self.simParams.maxHeight)
        return newHeightMap

    # populates self.drainageArea
    def computeDrainageAreaMap(self, heightMap) -> np.ndarray:
         # sort by descending height
        cellHeights = []
        for i in range(self.simParams.gridShape[0]):
            for k in range(self.simParams.gridShape[1]):
                cellHeights.append((i, k, heightMap[i][k]))
        cellHeights.sort(key = lambda ch: -ch[2]) 

        drainageAreaMap = np.ones(self.simParams.gridShape) * self.simParams.cellSize
        for i, k, h in cellHeights:
            neighborCells = self.getNeighborCells((i, k))
            relFlows = {}
            totalRelFlow = 0.0
            for ni, nk in neighborCells:
                nh = heightMap[ni][nk]
                if nh >= h:
                    continue
                
                di, dk = ni - i, nk - k
                relFlow = pow((h - nh) / np.sqrt(di * di + dk * dk), 4) # only need to compute proportionally due to normalization later
                relFlows[(ni, nk)] = relFlow
                totalRelFlow += 1

            if len(relFlows) == 0 or totalRelFlow < 0.001:
                continue

            for ni, nk in relFlows:
                drainageAreaMap[ni][nk] += drainageAreaMap[i][k] * relFlows[(ni, nk)] / totalRelFlow

        return drainageAreaMap

    def createOutputMesh(self, outputData: om.MObject, numIterations: int):
        if numIterations >= len(self.heightMapTs):
            raise RuntimeError(f"No height map corresponding to {numIterations} iterations.")

        vertices = []
        polygonCounts = []
        polygonConnects = []

        heightMap: np.ndarray = self.heightMapTs[numIterations]
        # center at (0, 0)
        xMin = -0.5 * self.simParams.cellSize * (heightMap.shape[0] - 1)
        zMin = -0.5 * self.simParams.cellSize * (heightMap.shape[1] - 1)

        # outputPoints will have the (x, y, z) point for each cell heightMap
        indexMap = {}
        for i in range(heightMap.shape[0]):
            x = xMin + self.simParams.cellSize * i
            for k in range(heightMap.shape[1]):
                z = zMin + self.simParams.cellSize * k
                y = heightMap[i][k]
                vertices.append(om.MPoint(x, y, z))
                indexMap[(i, k)] = len(vertices) - 1

        for i in range(heightMap.shape[0] - 1):
            for k in range(heightMap.shape[1] - 1):
                # Create the quad with lower corner at grid point (i, k); this order lets the top be shaded
                polygonCounts.append(4)
                polygonConnects.append(indexMap[(i, k)])
                polygonConnects.append(indexMap[(i, k + 1)])
                polygonConnects.append(indexMap[(i + 1, k + 1)])
                polygonConnects.append(indexMap[(i + 1, k)])

        fnMesh = om.MFnMesh()
        meshObj: om.MObject = fnMesh.create(vertices, polygonCounts, polygonConnects, parent=outputData)
        return meshObj

    # ROUGHLY from https://stackoverflow.com/a/76686523
    # TODO: inspect for correctness
    def makeHeightMapLaplacian(self, heightMap):
        grad_x, grad_z = np.gradient(heightMap, self.xStep, self.zStep)
        grad_xx = np.gradient(grad_x, self.xStep, axis=0)
        grad_zz = np.gradient(grad_z, self.zStep, axis=1)
        return grad_xx + grad_zz

    @staticmethod
    def initialize():
        nAttr = om.MFnNumericAttribute()
        tAttr = om.MFnTypedAttribute()

        TerroderNode.aTime = nAttr.create("time", "t", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 1.0

        # input
        TerroderNode.aUpliftMapFile = tAttr.create("upliftMapFile", "uf", om.MFnNumericData.kString)
        MAKE_INPUT(tAttr)
        tAttr.usedAsFilename = True

        TerroderNode.aCellSize = nAttr.create("cellSize", "cs", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 0.1

        TerroderNode.aGridSizeX = nAttr.create("gridSizeX", "gsx", om.MFnNumericData.kInt)
        MAKE_INPUT(nAttr)
        nAttr.default = 50

        TerroderNode.aGridSizeZ = nAttr.create("gridSizeZ", "gsz", om.MFnNumericData.kInt)
        MAKE_INPUT(nAttr)
        nAttr.default = 50

        # output
        TerroderNode.aOutputMesh = tAttr.create(TerroderNode.OUTPUT_MESH_ATTR_LONG_NAME, TerroderNode.OUTPUT_MESH_ATTR_SHORT_NAME, om.MFnData.kMesh)
        MAKE_OUTPUT(tAttr)

        # Add the attributes to the node and set up the
        #         attributeAffects (addAttribute, and attributeAffects)
        # Don't do try/except here, otherwise the error message won't be printed
        om.MPxNode.addAttribute(TerroderNode.aTime)
        om.MPxNode.addAttribute(TerroderNode.aUpliftMapFile)
        om.MPxNode.addAttribute(TerroderNode.aCellSize)
        om.MPxNode.addAttribute(TerroderNode.aGridSizeX)
        om.MPxNode.addAttribute(TerroderNode.aGridSizeZ)
        om.MPxNode.addAttribute(TerroderNode.aOutputMesh)

        om.MPxNode.attributeAffects(TerroderNode.aTime, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aUpliftMapFile, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aCellSize, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aGridSizeX, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aGridSizeZ, TerroderNode.aOutputMesh)

        print("[DEBUG] TerroderNode initialised.\n")
    
    @staticmethod
    def create():
        return TerroderNode()
    
class TerroderUI(object):
    """
    Static class to organize UI info
    """
    createdMenuName = ""

    @staticmethod
    def createMenu():
        mainWindowName = mm.eval('string $temp = $gMainWindow;')
        TerroderUI.createdMenuName = cmds.menu(l="Terroder", p=mainWindowName)
        invokeMenuItemName = cmds.menuItem(l="Create Terroder Mesh", p=TerroderUI.createdMenuName, c=TerroderUI._invokeCommand)

    @staticmethod
    def destroyMenu():
        cmds.deleteUI(TerroderUI.createdMenuName)
        createdMenuName = ""
    
    @staticmethod
    def _invokeCommand(*args) -> None:
        transformNodeName = cmds.createNode("transform")
        visibleMeshNodeName = cmds.createNode("mesh", parent=transformNodeName)
        cmds.sets(visibleMeshNodeName, add="initialShadingGroup")
        terroderNodeName = cmds.createNode(TerroderNode.TYPE_NAME)
        cmds.connectAttr("time1.outTime", f"{terroderNodeName}.{TerroderNode.TIME_ATTR_LONG_NAME}")
        cmds.connectAttr(f"{terroderNodeName}.{TerroderNode.OUTPUT_MESH_ATTR_LONG_NAME}", f"{visibleMeshNodeName}.inMesh")

        """
        BELOW: Try to select the transform and switch to the TerroderNode window in the attribute editor
        doesn't work yet (perhaps too quick? last line does the job if done manually in script editor)
        
        cmds.select(transformNodeName, r=True)
        attrEdTabLayoutName = mm.eval('string $temp = $gAETabLayoutName;')
        tabNames = cmds.tabLayout(attrEdTabLayoutName, q=True, tli=True)
        for i in range(len(tabNames)):
            if tabNames[i] == terroderNodeName:
                cmds.tabLayout(attrEdTabLayoutName, e=True, sti=i+1)
        """

# initialize the script plug-in
def initializePlugin(mObject):
    mPlugin = om.MFnPlugin(mObject, "Company", "1.0", "Any")
    # Don't use try except bc otherwise the error message won't be printed
    mPlugin.registerNode(TerroderNode.TYPE_NAME, TerroderNode.ID, TerroderNode.create, TerroderNode.initialize, om.MPxNode.kDependNode)

    TerroderUI.createMenu()

# uninitialize the script plug-in
def uninitializePlugin(mObject):
    TerroderUI.destroyMenu()

    mplugin = om.MFnPlugin(mObject)
    # Don't use try except bc otherwise the error message won't be printed
    mplugin.deregisterNode(TerroderNode.ID)
