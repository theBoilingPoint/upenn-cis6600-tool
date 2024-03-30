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
    OUTPUT_PERIOD_FLAG = "-op"
    OUTPUT_PERIOD_LONG_FLAG = "-outputPeriod"

    NEIGHBOR_ORDER = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    SLOPE_NORM_EXPONENT = 4.0
    STEEPEST_SLOPE_EXPONENT = 2.0
    DRAIN_AREA_EXPONENT = 1.0

    def __init__(self):
        om.MPxCommand.__init__(self)
        # control parameters; initialized to defaults
        self.cellSize = 0.2
        self.numIters = 5
        # as of yet not exposed
        self.minUplift = 0.0
        self.maxUplift = 0.05
        self.erosionScale = 0.01
        self.minHeight = 0.0
        self.initialHeight = 1.0
        self.maxHeight = 5.0  # TODO: enforce by constraint
        # TODO: laplacianScale (and add Laplacian term)

        # variables used during the command execution
        self.xMin = 0.0
        self.xMax = 0.0
        self.zMin = 0.0
        self.zMax = 0.0
        self.xStep = 0.0
        self.zStep = 0.0
        self.gridShape = None
        self.cellUsed = None  # TODO: incorporate
        self.upliftMap = None
        self.heightMap = None
        self.drainageAreaMap = None
        self.numIterationsRun = 0

    @property
    def upliftRange(self) -> float:
        return self.maxUplift - self.minUplift
    
    @property
    def heightRange(self) -> float:
        return self.maxHeight - self.minHeight
    
    @property
    def xRange(self) -> float:
        return self.xMax - self.xMin
    
    @property
    def zRange(self) -> float:
        return self.zMax - self.zMin
    
    @property
    def cellArea(self) -> float:
        return self.xStep * self.zStep
    
    
    def interpolateX(self, fraction: float) -> float:
        return self.xMin * (1 - fraction) + self.xMax * fraction
    
    def interpolateZ(self, fraction: float) -> float:
        return self.zMin * (1 - fraction) + self.zMax * fraction
    
    def cellInBounds(self, cell) -> bool:
        i, k = cell
        return 0 <= i < self.heightMap.shape[0] and 0 <= k < self.heightMap.shape[1]
    
    def getNeighborCells(self, cell):
        neighborCells = []
        i, k = cell
        for di, dk in TerroderCommand.NEIGHBOR_ORDER:
            ni, nk = i + di, k + dk
            if self.cellInBounds((ni, nk)):
                neighborCells.append((ni, nk))
        return neighborCells
    
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
        self.gridShape = (xCellDim + 1, zCellDim + 1)
        om.MGlobal.displayInfo(f"[DEBUG] grid shape: {self.gridShape}")
        self.xStep, self.zStep = self.xRange / xCellDim, self.zRange / zCellDim

        upliftInput = np.zeros(self.gridShape)
        self.cellUsed = np.full(self.gridShape, False)

        # Begin raycasting from above the uplift mesh to read the y-value at xz lattice points

        raycastY = bb[4] + 0.1  # 0.1 "slack distance"
        rayDirection = om.MFloatVector(0, -1, 0)
        # max distance is 0.2 + yRange since raycastY is only 0.1 above the bounding box
        raycastDistance = 0.2 + (bb[4] - bb[1])
        for i in range(self.gridShape[0]):
            x = self.interpolateX(float(i) / float(xCellDim))
            for k in range(self.gridShape[1]):
                z = self.interpolateZ(float(k) / float(zCellDim))
                rayOrigin = om.MFloatPoint(x, raycastY, z)
                intersectionResult = selectedMesh.closestIntersection(rayOrigin, rayDirection, om.MSpace.kWorld, raycastDistance, False)
                if intersectionResult is not None:
                    hitPoint = intersectionResult[0]
                    upliftInput[i][k] = hitPoint.y
                    self.cellUsed[i][k] = True
        
        # Linearly remap upliftMap so that it fits within the uplift range
        minUpliftInput = np.min(upliftInput) - 0.001
        maxUpliftInput = np.max(upliftInput) + 0.001
        normalizedUpliftInput = (upliftInput - minUpliftInput) / (maxUpliftInput - minUpliftInput)  # to [0, 1]
        self.upliftMap = self.minUplift + self.upliftRange * normalizedUpliftInput

        self.runSimulation()
        self.createOutputMesh()

        self.setResult("[DEBUG] Executed command")
    
    def runSimulation(self):
        if self.upliftMap is None:
            raise RuntimeError("The uplift map hasn't been created.")
        self.heightMap = self.makeInitialHeightMap()
        self.drainageAreaMap = np.ones(self.upliftMap.shape)
        for _ in range(self.numIters):
            self.runIteration()
    
    def makeInitialHeightMap(self):
        flatMid = np.full(self.gridShape, self.initialHeight)
        randomness = (2. * np.random.random_sample(self.gridShape) - 1.) * (self.xStep + self.zStep)
        return flatMid + randomness

    def runIteration(self):
        # Compute steepest slope to a lower neighbor
        steepestSlope = np.zeros(self.gridShape)  # 0 if no lower neighbor
        for i in range(self.gridShape[0]):
            for k in range(self.gridShape[1]):
                height = self.heightMap[i][k]

                for ni, nk in self.getNeighborCells((i, k)):
                    neighborHeight = self.heightMap[ni][nk]
                    if neighborHeight >= height:
                        continue

                    xDist = (ni - i) * self.xStep
                    zDist = (nk - k) * self.zStep
                    neighborDist = np.sqrt(xDist * xDist + zDist * zDist)
                    slope = (height - neighborHeight) / neighborDist
                    if slope > steepestSlope[i][k]:
                        steepestSlope[i][k] = slope

        self.drainageAreaMap = self.makeDrainageAreaMap()

        # Equals 1 at steepest slope 1 and drain area 1
        erosion = np.power(steepestSlope, TerroderCommand.STEEPEST_SLOPE_EXPONENT) * np.power(self.drainageAreaMap, TerroderCommand.DRAIN_AREA_EXPONENT)

        self.heightMap += self.upliftMap - self.erosionScale * erosion
        self.heightMap = np.clip(self.heightMap, self.minHeight, self.maxHeight)  # clip height map

        self.numIterationsRun += 1
    
    # populates self.drainageArea
    def makeDrainageAreaMap(self) -> np.ndarray:
        cellHeights = []
        for i in range(self.gridShape[0]):
            for k in range(self.gridShape[0]):
                cellHeights.append((i, k, self.heightMap[i][k]))
        cellHeights.sort(key = lambda ch: -ch[2])  # sort by descending height

        globalMaxDiff = 0
        drainageArea = np.ones(self.gridShape)
        for i, k, h in cellHeights:
            neighborCells = self.getNeighborCells((i, k))
            relFlow = {}
            for ni, nk in neighborCells:
                nh = self.heightMap[ni][nk]
                if nh >= h:
                    continue
                
                di, dk = ni - i, nk - k
                relSlope = (h - nh) / math.sqrt(di * di + dk * dk) # only need to compute proportionally due to normalization later
                relFlow[(ni, nk)] = pow(relSlope, 4)
                globalMaxDiff = max(globalMaxDiff, h - nh)
            
            if len(relFlow) == 0:
                continue

            totalRelFlow = sum(relFlow.values())
            if totalRelFlow < 0.001:
                continue
            for ni, nk in relFlow:
                flowFraction = relFlow[(ni, nk)] / totalRelFlow
                drainageArea[ni][nk] += drainageArea[i][k] * flowFraction

        if self.numIterationsRun % 10 == 9:
            om.MGlobal.displayInfo(f"[DEBUG] iteration {self.numIterationsRun}, xstep: {self.xStep}, avg drainage area: {np.mean(drainageArea)}, max diff: {globalMaxDiff}")
        return drainageArea
    
    # ROUGHLY from https://stackoverflow.com/a/76686523
    # TODO: inspect for correctness
    def makeHeightMapLaplacian(self):
        grad_x, grad_z = np.gradient(self.heightMap, self.xStep, self.zStep)
        grad_xx = np.gradient(grad_x, self.xStep, axis=0)
        grad_zz = np.gradient(grad_z, self.zStep, axis=1)
        return grad_xx + grad_zz    

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
