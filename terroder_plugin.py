import sys

# Uses Maya's Python API 2.0
# reference: https://help.autodesk.com/view/MAYAUL/2022/ENU/?guid=Maya_SDK_py_ref_index_html
import maya.api.OpenMaya as om;
import maya.cmds as cmds;
import numpy as np;
import math;

# Import the Python wrappers for MEL commands

# The name of the command. 

class TerroderCommand(om.MPxCommand):
    NAME = "terroder"
    CELL_SIZE_FLAG = "-cs"
    CELL_SIZE_LONG_FLAG = "-cellSize"
    NUM_ITERS_FLAG = "-i"
    NUM_ITERS_LONG_FLAG = "-iterations"
    SQRT_2 = math.sqrt(2)
    NEIGHBOR_ORDER = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    SLOPE_NORM_EXPONENT = 4.0
    STEEPEST_SLOPE_EXPONENT = 1.0
    DRAIN_AREA_EXPONENT = 0.5

    def __init__(self):
        om.MPxCommand.__init__(self)
        # control parameters; initialized to defaults
        self.cellSize = 0.05
        self.numIters = 5

        # variables used during the command execution
        self.xMin = 0.0
        self.xMax = 0.0
        self.zMin = 0.0
        self.zMax = 0.0
        self.xStep = 0.0
        self.zStep = 0.0
        self.upliftMap = None
        self.heightMap = None
    
    @property
    def xRange(self) -> float:
        return self.xMax - self.xMin
    
    @property
    def zRange(self) -> float:
        return self.zMax - self.zMin
    
    def interpolateX(self, fraction: float) -> float:
        return self.xMin * (1 - fraction) + self.xMax * fraction
    
    def interpolateZ(self, fraction: float) -> float:
        return self.zMin * (1 - fraction) + self.zMax * fraction
    
    def cellInBounds(self, i: int, k: int) -> bool:
        return 0 <= i < self.heightMap.shape[0] and 0 <= k < self.heightMap.shape[1]
    
    def doIt(self, args):
        argDb = om.MArgDatabase(self.syntax(), args)

        # Read arguments from flags
        if argDb.isFlagSet(TerroderCommand.CELL_SIZE_FLAG):
            self.cellSize = argDb.flagArgumentDouble(TerroderCommand.CELL_SIZE_FLAG, 0)
        if argDb.isFlagSet(TerroderCommand.NUM_ITERS_FLAG):
            self.numIters = argDb.flagArgumentInt(TerroderCommand.NUM_ITERS_FLAG, 0)

        om.MGlobal.displayInfo(f"[DEBUG] cell size: {self.cellSize}, numIters: {self.numIters}")

        # We expect exactly one mesh to be selected; we will read uplift from it
        selectedObjNames = cmds.ls(selection = True)
        om.MGlobal.displayInfo(f"[DEBUG] selected object names: [{', '.join(selectedObjNames)}]")
        if len(selectedObjNames) != 1:
            om.MGlobal.displayError("There must be exactly one object selected.")
            self.setResult("Did not execute command due to an error.")
            return
        
        selectedMesh = TerroderCommand.nameToMesh(selectedObjNames[0])

        # Read x- and z- bounds from the bounding box of the Mesh
        # TODO: if the input mesh doesn't include anything at a particular (x, z), neither should the output mesh
        bb = cmds.exactWorldBoundingBox(selectedObjNames[0])
        self.xMin, self.xMax = bb[0], bb[3]
        self.zMin, self.zMax = bb[2], bb[5]
        om.MGlobal.displayInfo(f"[DEBUG] selected object bounding box: {bb[0:3]} to {bb[3:6]}")
        if self.xRange <= 0.01 or self.zRange <= 0.01:
            om.MGlobal.displayError(f"(x, z) range ({self.xRange}, {self.zRange}) is too small in some dimension.")
            self.setResult("Did not execute command due to an error.")
            return
        # Ensures (xMin, zMin) and (xMax, zMax) are within the bounding box of the input mesh
        self.xMin += 0.001
        self.zMin += 0.001
        self.xMax -= 0.001
        self.zMax -= 0.001

        # Set the grid size
        xCellDim = max(int(math.ceil(self.xRange / self.cellSize + 0.01)), 2)
        zCellDim = max(int(math.ceil(self.zRange / self.cellSize + 0.01)), 2)
        gridShape = (xCellDim + 1, zCellDim + 1)
        om.MGlobal.displayInfo(f"[DEBUG] grid shape: {gridShape}")
        self.xStep, self.zStep = self.xRange / xCellDim, self.zRange / zCellDim
        self.upliftMap = np.zeros(gridShape)


        # Begin raycasting from above the uplift mesh to read the y-value at xz lattice points

        raycastY = bb[4] + 0.1  # 0.1 "slack distance"
        rayDirection = om.MFloatVector(0, -1, 0)
        # max distance is 0.2 + yRange since raycastY is only 0.1 above the bounding box
        raycastDistance = 0.2 + (bb[4] - bb[1])
        for i in range(gridShape[0]):
            x = self.interpolateX(float(i) / float(xCellDim))
            for k in range(gridShape[1]):
                z = self.interpolateZ(float(k) / float(zCellDim))
                rayOrigin = om.MFloatPoint(x, raycastY, z)
                intersectionResult = selectedMesh.closestIntersection(rayOrigin, rayDirection, om.MSpace.kWorld, raycastDistance, False)
                if intersectionResult is not None:
                    hitPoint = intersectionResult[0]
                    self.upliftMap[i][k] = hitPoint.y
        
        # Normalize upliftMap so that uplifts appear in the range -1 to 1
        minUplift = np.min(self.upliftMap) - 0.01
        maxUplift = np.max(self.upliftMap) + 0.01
        self.upliftMap = -1 + 2 * np.divide(np.subtract(self.upliftMap, minUplift), np.subtract(maxUplift, minUplift))

        self.runSimulation()
        self.createOutputMesh()

        self.setResult("[DEBUG] Executed command")
    
    def runSimulation(self):
        if self.upliftMap is None:
            raise RuntimeError("The uplift map hasn't been created.")
        self.heightMap = np.random.random_sample(self.upliftMap.shape)
        for _ in range(self.numIters):
            self.runIteration()
    
    def runIteration(self):
        gridShape = self.heightMap.shape
        averageHeight = np.average(self.heightMap)

        # drainProportion[i][k][d] is the proportion of water at (i, k) that flows in the direction NEIGHBOR_ORDER[d]
        drainProportion = np.zeros((gridShape[0], gridShape[1], 8))
        steepestSlope = np.zeros(gridShape)  # 0 means this is a pit
        for i in range(gridShape[0]):
            for k in range(gridShape[1]):
                height = self.heightMap[i][k]
                for d in range(8):
                    di, dk = TerroderCommand.NEIGHBOR_ORDER[d]
                    ni, nk = i + di, k + dk
                    neighborHeight = self.heightMap[ni][nk] if self.cellInBounds(ni, nk) else averageHeight
                    if neighborHeight < height:
                        dist = math.sqrt(math.pow(di * self.xStep, 2) + math.pow(dk * self.zStep, 2))
                        slope = (height - neighborHeight) / dist
                        steepestSlope[i][k] = max(slope, steepestSlope[i][k])
                        drainProportion[i][k][d] = math.pow(slope, TerroderCommand.SLOPE_NORM_EXPONENT)
                s = np.sum(drainProportion[i][k][:])
                if s > 0:
                    drainProportion[i][k][:] /= s
        
        # Approximate (relative) drainage Area
        drainArea = np.ones(gridShape)
        for t in range(10):
            waterDelta = np.ones(gridShape)
            for i in range(gridShape[0]):
                for k in range(gridShape[1]):
                    totalDrainAmount = 0
                    for d in range(drainProportion.shape[2]):
                        if drainProportion[i][k][d] <= 0.0:
                            continue

                        drainAmount = drainProportion[i][k][d] * drainArea[i][k]
                        waterDelta[i][k] -= drainAmount
                        
                        # add water to neighbor if possible
                        di, dk = TerroderCommand.NEIGHBOR_ORDER[d]
                        ni, nk = i + di, k + dk
                        if self.cellInBounds(ni, nk):
                            waterDelta[ni][nk] += drainAmount
            drainArea += waterDelta
        drainArea = np.clip(drainArea, 0, None)
        
        slopeTerm = np.power(steepestSlope, TerroderCommand.STEEPEST_SLOPE_EXPONENT)
        drainAreaTerm = np.power(drainArea, TerroderCommand.DRAIN_AREA_EXPONENT)
        self.heightMap += self.upliftMap - np.multiply(slopeTerm, drainAreaTerm)
        

    def createOutputMesh(self):
        if self.heightMap is None:
            raise RuntimeError("The height map hasn't been created.")
        minHeight = np.min(self.heightMap) - 0.01
        maxHeight = np.max(self.heightMap) + 0.01

        # outputPoints will have the (x, y, z) point for each cell in self.heightMap
        outputPoints = np.empty((self.heightMap.shape[0], self.heightMap.shape[1], 3))
        for i in range(self.heightMap.shape[0]):
            x = self.interpolateX(float(i) / float(self.heightMap.shape[0] - 1))
            for k in range(self.heightMap.shape[1]):
                z = self.interpolateZ(float(k) / float(self.heightMap.shape[1] - 1))
                y = (self.heightMap[i][k] - minHeight) / (maxHeight - minHeight)
                outputPoints[i][k][0] = x
                outputPoints[i][k][1] = y
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
    
    @staticmethod
    def nameToMesh(name):
        selectedMesh = None
        selectionList = om.MSelectionList()
        selectionList.add(name)

        om.MGlobal.displayInfo(f"[DEBUG] Object '{name}' added to the selection list.")

        dagPath = selectionList.getDagPath(0)
        if dagPath.hasFn(om.MFn.kMesh):
            # Initialize the MFnMesh function set with the dagPath
            selectedMesh = om.MFnMesh(dagPath)
        else:
            raise TypeError(f"Object '{name}' is not a mesh.")

        if not selectedMesh:
            raise RuntimeError(f"Couldn't find a mesh named '{name}'.")

        return selectedMesh
    
    # Required to parse arguments
    # Needed to add this to avoid crashing
    @staticmethod
    def createSyntax():
        syntax = om.MSyntax()
        syntax.addFlag(TerroderCommand.CELL_SIZE_FLAG, TerroderCommand.CELL_SIZE_LONG_FLAG, om.MSyntax.kDouble)
        syntax.addFlag(TerroderCommand.NUM_ITERS_FLAG, TerroderCommand.NUM_ITERS_LONG_FLAG, om.MSyntax.kLong)
        return syntax
    
    @staticmethod
    def createCommand():
        return TerroderCommand()

def maya_useNewAPI():
    pass

# Initialize the plugin
def initializePlugin(plugin):
    pluginFn = om.MFnPlugin(plugin)
        # Must register syntaxCreator as well
    pluginFn.registerCommand(TerroderCommand.NAME, TerroderCommand.createCommand, TerroderCommand.createSyntax)

# Uninitialize the plugin
def uninitializePlugin(plugin):
    mplugin = om.MFnPlugin(plugin)
    mplugin.deregisterCommand(TerroderCommand.NAME)
