import os

import math
import maya.api.OpenMaya as om
import maya.cmds as cmds
import maya.mel as mm

import numpy as np
from PIL import Image
from typing import Tuple

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

class TerroderSimulationParameters(object):
    # Intended to be read-only. If you need to change values, you need a new TerroderSimulationParameters object.
    # All parameters that can be changed by the user
    # Does NOT include time; if time alone changes, we might not need to redo the entire simulation

    def __init__(self, cellSize: float, targetGridScale: Tuple[float, float], upliftMapFile: str, 
                 minUpliftRatio: float, relUpliftScale: float, relErosionScale: float, waterHalfRetentionDist: float):
        self._cellSize: float = cellSize
        self._targetGridScale: Tuple[float, float] = targetGridScale
        self._upliftMapFile: str = upliftMapFile
        self._minUpliftRatio: float = minUpliftRatio
        self._relUpliftScale: float = relUpliftScale
        self._relErosionScale: float = relErosionScale
        self._waterHalfRetentionDist: float = waterHalfRetentionDist
        self._upliftMap = None

        gridShapeX = max(int(math.ceil(self._targetGridScale[0] / self.cellSize)), 4)
        gridShapeZ = max(int(math.ceil(self._targetGridScale[1] / self.cellSize)), 4)
        self.gridShape: Tuple[int, int] = (gridShapeX, gridShapeZ)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TerroderSimulationParameters):
            return False
        
        return math.isclose(self.cellSize, other.cellSize) \
            and self.gridShape == other.gridShape \
            and self.upliftMapFile == other.upliftMapFile \
            and math.isclose(self.minUpliftRatio, other.minUpliftRatio) \
            and math.isclose(self.relUpliftScale, other.relUpliftScale) \
            and math.isclose(self.relErosionScale, other.relErosionScale) \
            and math.isclose(self.waterHalfRetentionDist, other.waterHalfRetentionDist)

    @property
    def cellSize(self) -> float:
        return self._cellSize
    
    @property
    def upliftMapFile(self) -> str:
        return self._upliftMapFile

    @property
    def minUpliftRatio(self) -> float:
        return self._minUpliftRatio

    @property
    def relUpliftScale(self) -> float:
        return self._relUpliftScale
    
    @property
    def relErosionScale(self) -> float:
        return self._relErosionScale
    
    @property
    def waterHalfRetentionDist(self) -> float:
        return self._waterHalfRetentionDist

    @property
    def upliftMap(self) -> np.ndarray:
        if self._upliftMap is None:
            self._upliftMap = np.zeros(self.gridShape)
            if len(self._upliftMapFile) > 0:
                try:
                    with Image.open(self.upliftMapFile) as image:
                        gsImage = image.convert('L')  # grayscale
                        step = (float(gsImage.size[0]) / float(self.gridShape[0]), float(gsImage.size[1]) / float(self.gridShape[1]))
                        
                        for i in range(self.gridShape[0]):
                            for k in range(self.gridShape[1]):
                                # Interpolate (i, k)
                                rx, ry = step[0] * i, step[1] * k
                                self._upliftMap[i][k] = TerroderSimulationParameters._readInterpolatedUplift(gsImage, (rx, ry))

                    self._upliftMap = self.minUpliftRatio + (1.0 - self.minUpliftRatio) * self._upliftMap
                    self._upliftMap = np.clip(self._upliftMap, 0., 1.)
                except FileNotFoundError:
                    om.MGlobal.displayWarning(f'File "{self.upliftMapFile}" not found.')
        
        return self._upliftMap
    
    @staticmethod
    def _readInterpolatedUplift(grayscaleImage: Image, coords: Tuple[float, float]):
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
                color = grayscaleImage.getpixel((fx + dx, fy + dy))
                uplifts.append(color / 255.0)
        
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
    MIN_HEIGHT = 0.0
    MAX_HEIGHT = 1.0
    UPLIFT_DEFAULT_SCALE = 0.01
    EROSION_DEFAULT_SCALE = 0.1

    TIME_ATTR_LONG_NAME = "time"
    TIME_ATTR_SHORT_NAME = "t"
    OUTPUT_MESH_ATTR_LONG_NAME = "outputMesh"
    OUTPUT_MESH_ATTR_SHORT_NAME = "om"

    # Attributes
    # input
    aTime = None
    aUpliftMapFile = None
    aMinUpliftRatio = None
    aCellSize = None
    aGridScaleX = None
    aGridScaleZ = None
    aUpliftRelScale = None
    aErosionRelScale = None
    aWaterHalfRetentionDistance = None

    simulateFromTimestamp = None

    # output
    aOutputMesh = None

    def __init__(self):
        om.MPxNode.__init__(self)

        self.simParams: TerroderSimulationParameters = None
        self.heightMapTs: list[np.ndarray] = []  # element i is the heightmap after i iterations

    @staticmethod
    def create():
        return TerroderNode()

    @staticmethod
    def initialize() -> None:
        nAttr = om.MFnNumericAttribute()
        tAttr = om.MFnTypedAttribute()

        TerroderNode.aTime = nAttr.create("time", "t", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 1.0

        # input
        TerroderNode.aUpliftMapFile = tAttr.create("upliftMapFile", "uf", om.MFnNumericData.kString)
        MAKE_INPUT(tAttr)
        tAttr.usedAsFilename = True

        TerroderNode.aMinUpliftRatio = nAttr.create("minUpliftRatio", "mur", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 0.1
        nAttr.setMin(0.0)
        nAttr.setMax(0.999)

        TerroderNode.aCellSize = nAttr.create("cellSize", "cs", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 0.1
        nAttr.setMin(0.001)

        TerroderNode.aGridScaleX = nAttr.create("gridScaleX", "sx", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 5.0
        nAttr.setMin(0.01)

        TerroderNode.aGridScaleZ = nAttr.create("gridScaleZ", "sz", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 5.0
        nAttr.setMin(0.01)

        TerroderNode.aUpliftRelScale = nAttr.create("upliftRelativeScale", "urs", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 1.0
        nAttr.setMin(0.0)

        TerroderNode.aErosionRelScale = nAttr.create("erosionRelativeScale", "ers", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 1.0
        nAttr.setMin(0.0)

        TerroderNode.aWaterHalfRetentionDistance = nAttr.create("waterHalfRetentionDistance", "whr", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 1.0
        nAttr.setMin(0.001)

        TerroderNode.simulateFromTimestamp = nAttr.create("simulateFromTimestamp", "sft", om.MFnNumericData.kBoolean)
        MAKE_INPUT(nAttr)
        nAttr.default = False

        # output
        TerroderNode.aOutputMesh = tAttr.create(TerroderNode.OUTPUT_MESH_ATTR_LONG_NAME, TerroderNode.OUTPUT_MESH_ATTR_SHORT_NAME, om.MFnData.kMesh)
        MAKE_OUTPUT(tAttr)

        # Add the attributes to the node and set up the
        #         attributeAffects (addAttribute, and attributeAffects)
        # Don't do try/except here, otherwise the error message won't be printed
        om.MPxNode.addAttribute(TerroderNode.aTime)
        om.MPxNode.addAttribute(TerroderNode.aUpliftMapFile)
        om.MPxNode.addAttribute(TerroderNode.aMinUpliftRatio)
        om.MPxNode.addAttribute(TerroderNode.aCellSize)
        om.MPxNode.addAttribute(TerroderNode.aGridScaleX)
        om.MPxNode.addAttribute(TerroderNode.aGridScaleZ)
        om.MPxNode.addAttribute(TerroderNode.aOutputMesh)
        om.MPxNode.addAttribute(TerroderNode.aUpliftRelScale)
        om.MPxNode.addAttribute(TerroderNode.aErosionRelScale)
        om.MPxNode.addAttribute(TerroderNode.aWaterHalfRetentionDistance)
        om.MPxNode.addAttribute(TerroderNode.simulateFromTimestamp)

        om.MPxNode.attributeAffects(TerroderNode.aTime, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aUpliftMapFile, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aMinUpliftRatio, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aCellSize, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aGridScaleX, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aGridScaleZ, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aUpliftRelScale, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aErosionRelScale, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aWaterHalfRetentionDistance, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.simulateFromTimestamp, TerroderNode.aOutputMesh)

        print("[DEBUG] TerroderNode initialised.\n")
    
    def compute(self, plug: om.MPlug, dataBlock: om.MDataBlock):
        if (plug != TerroderNode.aOutputMesh) and (plug.parent() != TerroderNode.aOutputMesh):
            return None
        
        rawTime = dataBlock.inputValue(TerroderNode.aTime).asFloat()
        numIterations = max(0, int(rawTime) - 1)

        # get the input data
        avUpliftMapFile = dataBlock.inputValue(TerroderNode.aUpliftMapFile).asString()
        avMinUpliftRatio = dataBlock.inputValue(TerroderNode.aMinUpliftRatio).asFloat()
        avCellSize = dataBlock.inputValue(TerroderNode.aCellSize).asFloat()
        avGridScale = (dataBlock.inputValue(TerroderNode.aGridScaleX).asFloat(), dataBlock.inputValue(TerroderNode.aGridScaleZ).asFloat())
        avRelUpliftScale = dataBlock.inputValue(TerroderNode.aUpliftRelScale).asFloat()
        avRelErosionScale = dataBlock.inputValue(TerroderNode.aErosionRelScale).asFloat()
        avWaterHalfRetentionDist = dataBlock.inputValue(TerroderNode.aWaterHalfRetentionDistance).asFloat()
        newSimParams = TerroderSimulationParameters(avCellSize, avGridScale, avUpliftMapFile, avMinUpliftRatio, 
                                                    avRelUpliftScale, avRelErosionScale, avWaterHalfRetentionDist)
        startFromTimestamp = dataBlock.inputValue(TerroderNode.simulateFromTimestamp).asBool()

        if self.simParams is None or newSimParams != self.simParams:
            # Current sim params are out of date; reset
            om.MGlobal.displayInfo("[DEBUG] Using new sim params and resetting simulation")
            self.simParams = newSimParams
            if startFromTimestamp:
                self.heightMapTs = [self.heightMapTs[int(rawTime)]]
            else:
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
        steepestSlopeMap = self.computeSteepestSlopeMap(curHeightMap)

        drainageAreaMap = self.computeDrainageAreaMap(curHeightMap)

        # Equals 1 at steepest slope 1 and drain area 1
        erosion = np.power(steepestSlopeMap, TerroderNode.STEEPEST_SLOPE_EXPONENT) * np.power(drainageAreaMap, TerroderNode.DRAIN_AREA_EXPONENT)

        nextHeightMap = np.copy(curHeightMap)
        nextHeightMap += self.simParams.upliftMap * self.simParams.relUpliftScale * TerroderNode.UPLIFT_DEFAULT_SCALE
        nextHeightMap -= erosion * self.simParams.relErosionScale * TerroderNode.EROSION_DEFAULT_SCALE
        nextHeightMap = np.clip(nextHeightMap, TerroderNode.MIN_HEIGHT, TerroderNode.MAX_HEIGHT)
        self.heightMapTs.append(nextHeightMap)

    def makeInitialHeightMap(self) -> np.ndarray:
        return np.zeros(self.simParams.gridShape)
    
    def computeSteepestSlopeMap(self, heightMap: np.ndarray) -> np.ndarray:
        steepestSlopeMap = np.zeros(self.simParams.gridShape)  # 0 if no lower neighbor
        for i in range(self.simParams.gridShape[0]):
            for k in range(self.simParams.gridShape[1]):
                height = heightMap[i][k]
                for ni, nk in self.getNeighborCells((i, k)):
                    neighborHeight = heightMap[ni][nk] if self.cellInBounds((ni, nk)) else 0.0
                    if neighborHeight >= height:
                        continue

                    xDist = (ni - i) * self.simParams.cellSize
                    zDist = (nk - k) * self.simParams.cellSize
                    neighborDist = np.sqrt(xDist * xDist + zDist * zDist)
                    slope = (height - neighborHeight) / neighborDist
                    if slope > steepestSlopeMap[i][k]:
                        steepestSlopeMap[i][k] = slope
        
        return steepestSlopeMap

    # populates self.drainageArea
    def computeDrainageAreaMap(self, heightMap: np.ndarray) -> np.ndarray:
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
                nh = heightMap[ni][nk] if self.cellInBounds((ni, nk)) else 0.0
                if nh >= h:
                    continue
                
                di, dk = ni - i, nk - k
                relFlow = pow((h - nh) / np.sqrt(di * di + dk * dk), 4) # only need to compute proportionally due to normalization later
                relFlows[(ni, nk)] = relFlow
                totalRelFlow += relFlow

            if len(relFlows) == 0:
                continue

            for ni, nk in relFlows:
                if self.cellInBounds((ni, nk)):
                    xDist = (ni - i) * self.simParams.cellSize
                    zDist = (nk - k) * self.simParams.cellSize
                    neighborDist = np.sqrt(xDist * xDist + zDist * zDist)
                    retention = np.power(0.5, neighborDist / self.simParams.waterHalfRetentionDist)
                    drainageAreaMap[ni][nk] += drainageAreaMap[i][k] * relFlows[(ni, nk)] / totalRelFlow * retention

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
    
    def cellInBounds(self, cell: Tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.simParams.gridShape[0] and 0 <= cell[1] < self.simParams.gridShape[1]
    
    def getNeighborCells(self, cell: Tuple[int, int]) -> list:
        return [(cell[0] + di, cell[1] + dk) for (di, dk) in TerroderNode.NEIGHBOR_ORDER]

    
class TerroderUI(object):
    """
    Static class to organize UI info
    """
    createdMenuName = ""

    @staticmethod
    def createMenu():
        mainWindowName = mm.eval('string $temp = $gMainWindow;')
        TerroderUI.createdMenuName = cmds.menu(l="Terroder", p=mainWindowName)
        cmds.menuItem(l="Create Terroder Mesh", p=TerroderUI.createdMenuName, c=TerroderUI._createNode)

    @staticmethod
    def destroyMenu():
        cmds.deleteUI(TerroderUI.createdMenuName)
        createdMenuName = ""
    
    @staticmethod
    def _createNode(*args) -> None:
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
