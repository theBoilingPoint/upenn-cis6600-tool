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

    def __init__(self):
        om.MPxCommand.__init__(self)
        # control parameters; initialized to defaults
        self.cellSize = 0.05

        # variables used during the command execution
        self.xMin = 0.0
        self.xMax = 0.0
        self.zMin = 0.0
        self.zMax = 0.0
        self.minUplift = 0.0
        self.maxUplift = 0.0
        self.uplift = None
        self.heightmap = None
    
    @property
    def xRange(self) -> float:
        return self.xMax - self.xMin
    
    @property
    def zRange(self) -> float:
        return self.zMax - self.zMin
    
    @property
    def upliftRange(self) -> float:
        return self.maxUplift - self.minUplift
    
    def interpolateX(self, fraction: float) -> float:
        return self.xMin * (1 - fraction) + self.xMax * fraction
    
    def interpolateZ(self, fraction: float) -> float:
        return self.zMin * (1 - fraction) + self.zMax * fraction
    
    def doIt(self, args):
        argDb = om.MArgDatabase(self.syntax(), args)

        if argDb.isFlagSet(TerroderCommand.CELL_SIZE_FLAG):
            self.cellSize = argDb.flagArgumentDouble(TerroderCommand.CELL_SIZE_FLAG, 0)

        om.MGlobal.displayInfo(f"[DEBUG] cell size: {self.cellSize}")

        selectedObjNames = cmds.ls(selection = True)
        om.MGlobal.displayInfo(f"[DEBUG] selected object names: [{', '.join(selectedObjNames)}]")
        
        if len(selectedObjNames) != 1:
            om.MGlobal.displayError("There must be exactly one object selected.")
            self.setResult("Did not execute command due to an error.")
            return
        
        selectedMesh = TerroderCommand.nameToMesh(selectedObjNames[0])

        bb = cmds.exactWorldBoundingBox(selectedObjNames[0])
        self.xMin = bb[0]
        self.xMax = bb[3]
        self.zMin = bb[2]
        self.zMax = bb[5]
        self.minUplift = bb[1]
        self.maxUplift = bb[4]
        om.MGlobal.displayInfo(f"[DEBUG] selected object bounding box: {bb[0:3]} to {bb[3:6]}")

        raycastY = self.maxUplift + 0.1  # 0.1 "slack distance"
        if self.xRange <= 0.01:
            om.MGlobal.displayError("Range of x values is too small.")
            self.setResult("Did not execute command due to an error.")
            return
        if self.zRange <= 0.01:
            om.MGlobal.displayError("Range of z values is too small.")
            self.setResult("Did not execute command due to an error.")
            return
        
        # Ensure (xMin, zMin) and (xMax, zMax) are within the bounding box
        self.xMin += 0.001
        self.zMin += 0.001
        self.xMax -= 0.001
        self.zMax -= 0.001

        xCellDim = max(int(math.ceil(self.xRange / self.cellSize + 0.01)), 2)
        zCellDim = max(int(math.ceil(self.zRange / self.cellSize + 0.01)), 2)
        gridShape = (xCellDim + 1, zCellDim + 1)

        om.MGlobal.displayInfo(f"[DEBUG] grid shape: {gridShape}")
        self.uplift = np.zeros(gridShape)

        rayDirection = om.MFloatVector(0, -1, 0)
        # max distance is 0.2 + yRange since raycastY is only 0.1 above the bounding box
        raycastDistance = 0.2 + self.upliftRange
        for i in range(gridShape[0]):
            x = self.interpolateX(float(i) / float(xCellDim))
            for k in range(gridShape[1]):
                z = self.interpolateZ(float(k) / float(zCellDim))
                rayOrigin = om.MFloatPoint(x, raycastY, z)

                intersectionResult = selectedMesh.closestIntersection(rayOrigin, rayDirection, om.MSpace.kWorld, raycastDistance, False)
                if intersectionResult is not None:
                    hitPoint = intersectionResult[0]
                    self.uplift[i][k] = hitPoint.y
        
        # Currently uplift is populated with y coordinates of a grid
        om.MGlobal.displayInfo(f"[DEBUG] uplift: {self.uplift}")

        self.runSimulation()
        self.createOutputMesh()

        self.setResult("[DEBUG] Executed command")
    
    def runSimulation(self):
        self.heightmap = np.copy(self.uplift) + 0.5 * np.random.random_sample(self.uplift.shape)
    
    def createOutputMesh(self):
        minHeight = np.min(self.heightmap) - 0.01
        maxHeight = np.max(self.heightmap) + 0.01

        outputPoints = np.empty((self.heightmap.shape[0], self.heightmap.shape[1], 3))
        for i in range(self.heightmap.shape[0]):
            x = self.interpolateX(float(i) / float(self.heightmap.shape[0] - 1))
            for k in range(self.heightmap.shape[1]):
                z = self.interpolateZ(float(k) / float(self.heightmap.shape[1] - 1))
                y = (self.heightmap[i][k] - minHeight) / (maxHeight - minHeight)
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
