import sys
import maya.api.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import maya.cmds as cmds;
import numpy as np;

# Useful functions for declaring attributes as inputs or outputs.
def MAKE_INPUT(attr):
    attr.setKeyable(True)
    attr.setStorable(True)
    attr.setReadable(True)
    attr.setWritable(True)
    
def MAKE_OUTPUT(attr):
    attr.setKeyable(False)
    attr.setStorable(False)
    attr.setReadable(True)
    attr.setWritable(False)

# Define the name of the node
kPluginNodeTypeName = "terroder"

# Give the node a unique ID. Make sure this ID is different from all of your
# other nodes!
nodeId = OpenMaya.MTypeId(0x8888)

class TerroderNode(OpenMayaMPx.MPxNode):
    # Declare class variables:
    # input
    iterations = OpenMaya.MObject()
    cellSize = OpenMaya.MObject()

    # output
    output_mesh = OpenMaya.MObject()

    # constants
    MENU_NAME = "Terroder"

    NEIGHBOR_ORDER = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # SLOPE_NORM_EXPONENT = 4.0
    STEEPEST_SLOPE_EXPONENT = 2.0
    DRAIN_AREA_EXPONENT = 1.0

    def __init__(self):
        OpenMayaMPx.MPxNode.__init__(self)

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
        for di, dk in TerroderNode.NEIGHBOR_ORDER:
            ni, nk = i + di, k + dk
            if self.cellInBounds((ni, nk)):
                neighborCells.append((ni, nk))
        return neighborCells
    
    @staticmethod
    def nameToMesh(name):
        selectedMesh = None
        selectionList = OpenMaya.MSelectionList()
        selectionList.add(name)

        OpenMaya.MGlobal.displayInfo(f"[DEBUG] Object '{name}' added to the selection list.")

        dagPath = selectionList.getDagPath(0)
        if dagPath.hasFn(OpenMaya.MFn.kMesh):
            # Initialize the MFnMesh function set with the dagPath
            selectedMesh = OpenMaya.MFnMesh(dagPath)
        else:
            raise TypeError(f"Object '{name}' is not a mesh.")

        if not selectedMesh:
            raise RuntimeError(f"Couldn't find a mesh named '{name}'.")

        return selectedMesh
    
    def makeInitialHeightMap(self):
        flatMid = np.full(self.gridShape, self.initialHeight)
        randomness = (2. * np.random.random_sample(self.gridShape) - 1.) * (self.xStep + self.zStep)
        return flatMid + randomness
    
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
        erosion = np.power(steepestSlope, TerroderNode.STEEPEST_SLOPE_EXPONENT) * np.power(self.drainageAreaMap, TerroderNode.DRAIN_AREA_EXPONENT)

        self.heightMap += self.upliftMap - self.erosionScale * erosion
        self.heightMap = np.clip(self.heightMap, self.minHeight, self.maxHeight)  # clip height map

        self.numIterationsRun += 1
    
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
        outputMesh = OpenMaya.MFnMesh()
        for i in range(outputPoints.shape[0] - 1):
            for k in range(outputPoints.shape[1] - 1):
                a = OpenMaya.MPoint([outputPoints[i][k][d] for d in range(3)])
                b = OpenMaya.MPoint([outputPoints[i+1][k][d] for d in range(3)])
                c = OpenMaya.MPoint([outputPoints[i+1][k+1][d] for d in range(3)])
                d = OpenMaya.MPoint([outputPoints[i][k+1][d] for d in range(3)])
                outputMesh.addPolygon([a, b, c, d], True, mergeTolerance)
        
        return outputMesh

    # ROUGHLY from https://stackoverflow.com/a/76686523
    # TODO: inspect for correctness
    def makeHeightMapLaplacian(self):
        grad_x, grad_z = np.gradient(self.heightMap, self.xStep, self.zStep)
        grad_xx = np.gradient(grad_x, self.xStep, axis=0)
        grad_zz = np.gradient(grad_z, self.zStep, axis=1)
        return grad_xx + grad_zz    

    def compute(self, plug, dataBlock):
        # get the input data
        self.numIters = dataBlock.inputValue(TerroderNode.iterations).asInt()
        self.cellSize = dataBlock.inputValue(TerroderNode.cellSize).asFloat()
        
        OpenMaya.MGlobal.displayInfo(f"[DEBUG] cell size: {self.cellSize}, numIters: {self.numIters}")

        # Set the output data
        outputMeshData = dataBlock.outputValue(TerroderNode.output_mesh)

        # We expect exactly one mesh to be selected; we will read uplift from it
        selectedObjNames = cmds.ls(selection = True)
        OpenMaya.MGlobal.displayInfo(f"[DEBUG] selected object names: [{', '.join(selectedObjNames)}]")
        if len(selectedObjNames) != 1:
            OpenMaya.MGlobal.displayError("There must be exactly one object selected.")
            # Node doesn't have a setResult method
            # self.setResult("Did not execute command due to an error.")
            return
        
        selectedMesh = TerroderNode.nameToMesh(selectedObjNames[0])

        # Read x- and z- bounds from the bounding box of the Mesh
        # TODO: if the input mesh doesn't include anything at a particular (x, z), neither should the output mesh
        bb = cmds.exactWorldBoundingBox(selectedObjNames[0])
        self.xMin, self.xMax = bb[0], bb[3]
        self.zMin, self.zMax = bb[2], bb[5]
        TerroderNode.MGlobal.displayInfo(f"[DEBUG] selected object bounding box: {bb[0:3]} to {bb[3:6]}")
        if self.xRange <= 0.01 or self.zRange <= 0.01:
            TerroderNode.MGlobal.displayError(f"(x, z) range ({self.xRange}, {self.zRange}) is too small in some dimension.")
            # Node doesn't have a setResult method
            # self.setResult("Did not execute command due to an error.")
            return
        # Ensures (xMin, zMin) and (xMax, zMax) are within the bounding box of the input mesh
        self.xMin += 0.001
        self.zMin += 0.001
        self.xMax -= 0.001
        self.zMax -= 0.001

        # Set the grid size
        xCellDim = max(int(np.ceil(self.xRange / self.cellSize + 0.01)), 2)
        zCellDim = max(int(np.ceil(self.zRange / self.cellSize + 0.01)), 2)
        self.gridShape = (xCellDim + 1, zCellDim + 1)
        OpenMaya.MGlobal.displayInfo(f"[DEBUG] grid shape: {self.gridShape}")
        self.xStep, self.zStep = self.xRange / xCellDim, self.zRange / zCellDim

        upliftInput = np.zeros(self.gridShape)
        self.cellUsed = np.full(self.gridShape, False)

        # Begin raycasting from above the uplift mesh to read the y-value at xz lattice points
        raycastY = bb[4] + 0.1  # 0.1 "slack distance"
        rayDirection = OpenMaya.MFloatVector(0, -1, 0)
        # max distance is 0.2 + yRange since raycastY is only 0.1 above the bounding box
        raycastDistance = 0.2 + (bb[4] - bb[1])
        for i in range(self.gridShape[0]):
            x = self.interpolateX(float(i) / float(xCellDim))
            for k in range(self.gridShape[1]):
                z = self.interpolateZ(float(k) / float(zCellDim))
                rayOrigin = OpenMaya.MFloatPoint(x, raycastY, z)
                intersectionResult = selectedMesh.closestIntersection(rayOrigin, rayDirection, OpenMaya.MSpace.kWorld, raycastDistance, False)
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
        # self.createOutputMesh()
        outputMeshData.setMObject(self.createOutputMesh())

        outputMeshData.setClean()
        dataBlock.setClean(plug)
        # Node doesn't have a setResult method
        # self.setResult("[DEBUG] Executed command")

# initializer
def nodeInitializer():
    nAttr = OpenMaya.MFnNumericAttribute()
    tAttr = OpenMaya.MFnTypedAttribute()
    
    # input
    TerroderNode.iterations = nAttr.create("iterations", "i", OpenMaya.MFnNumericData.kInt, 5)
    MAKE_INPUT(nAttr)

    TerroderNode.cellSize = nAttr.create("cellSize", "cs", OpenMaya.MFnNumericData.kFloat, 0.2)
    MAKE_INPUT(nAttr)

    # output
    TerroderNode.output_mesh = tAttr.create("outputMesh", "om", OpenMaya.MFnData.kMesh)
    MAKE_OUTPUT(tAttr)

    # Add the attributes to the node and set up the
    #         attributeAffects (addAttribute, and attributeAffects)
    # Don't do try/except here, otherwise the error message won't be printed
    OpenMayaMPx.MPxNode.addAttribute(TerroderNode.iterations)
    OpenMayaMPx.MPxNode.addAttribute(TerroderNode.cellSize)
    OpenMayaMPx.MPxNode.addAttribute(TerroderNode.output_mesh)

    OpenMayaMPx.MPxNode.attributeAffects(TerroderNode.iterations, TerroderNode.output_mesh)
    OpenMayaMPx.MPxNode.attributeAffects(TerroderNode.cellSize, TerroderNode.output_mesh)

    print("TerroderNode initialised.\n")

# creator
def nodeCreator():
    return OpenMayaMPx.asMPxPtr( TerroderNode() )

# initialize the script plug-in
def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    # Don't use try except bc otherwise the error message won't be printed
    mplugin.registerNode( kPluginNodeTypeName, nodeId, nodeCreator, nodeInitializer )


# uninitialize the script plug-in
def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    # Don't use try except bc otherwise the error message won't be printed
    mplugin.deregisterNode( nodeId )
