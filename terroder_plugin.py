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

    def doIt(self, args):
        argDb = om.MArgDatabase(self.syntax(), args)

        cellSizeVal = 0.05
        if argDb.isFlagSet(TerroderCommand.CELL_SIZE_FLAG):
            cellSizeVal = argDb.flagArgumentDouble(TerroderCommand.CELL_SIZE_FLAG, 0)

        om.MGlobal.displayInfo(f"[DEBUG] grid size: {cellSizeVal}")

        selectedObjNames = cmds.ls(selection = True)
        om.MGlobal.displayInfo(f"[DEBUG] selected object names: [{', '.join(selectedObjNames)}]")
        
        if len(selectedObjNames) != 1:
            om.MGlobal.displayError("There must be exactly one object selected.")
            self.setResult("Did not execute command due to an error.")
            return
        
        selectedMesh = TerroderCommand.nameToMesh(selectedObjNames[0])

        bb = cmds.exactWorldBoundingBox(selectedObjNames[0])
        xMin = bb[0]
        yMin = bb[1]
        zMin = bb[2]
        xMax = bb[3]
        yMax = bb[4]
        zMax = bb[5]
        om.MGlobal.displayInfo(f"[DEBUG] selected object bounding box: {bb[0:3]} to {bb[3:6]}")

        raycastY = yMax + 0.1  # 0.1 "slack distance"
        if xMax - xMin <= 0.01:
            om.MGlobal.displayError("Range of x values is too small.")
            self.setResult("Did not execute command due to an error.")
            return
        if zMax - zMin <= 0.01:
            om.MGlobal.displayError("Range of z values is too small.")
            self.setResult("Did not execute command due to an error.")
            return
        
        # Ensure (xMin, zMin) and (xMax, zMax) are within the bounding box
        xMin += 0.001
        zMin += 0.001
        xMax -= 0.001
        zMax -= 0.001
        xRange = xMax - xMin
        yRange = yMax - yMin
        zRange = zMax - zMin

        xCellDim = max(int(math.ceil(xRange / cellSizeVal + 0.01)), 2)
        zCellDim = max(int(math.ceil(zRange / cellSizeVal + 0.01)), 2)
        gridShape = (xCellDim + 1, zCellDim + 1)

        om.MGlobal.displayInfo(f"[DEBUG] grid shape: {gridShape}")
        uplift = np.zeros(gridShape)

        rayDirection = om.MFloatVector(0, -1, 0)
        # max distance is 0.2 + yRange since raycastY is only 0.1 above the bounding box
        raycastDistance = 0.2 + yRange
        for i in range(gridShape[0]):
            xFrac = float(i) / float(xCellDim)
            x = xMin * (1 - xFrac) + xMax * xFrac
            for k in range(gridShape[1]):
                zFrac = float(k) / float(zCellDim) 
                z = zMin * (1 - zFrac) + zMax * zFrac
                rayOrigin = om.MFloatPoint(x, raycastY, z)

                intersectionResult = selectedMesh.closestIntersection(rayOrigin, rayDirection, om.MSpace.kWorld, raycastDistance, False)
                if intersectionResult is not None:
                    hitPoint = intersectionResult[0]
                    uplift[i][k] = hitPoint.y
        
        # Currently uplift is populated with y coordinates of a grid
        om.MGlobal.displayInfo(f"[DEBUG] uplift: {uplift}")

        heightmap = self.runSimulation(uplift)
        self.createOutputMesh(heightmap, cellSizeVal, xMin, zMin, xMax, zMax)

        self.setResult("[DEBUG] Executed command")
    
    def runSimulation(self, uplift):
        heightmap = np.copy(uplift) + np.random.random_sample(uplift.shape)
        return heightmap
    
    def createOutputMesh(self, heightmap, cellSizeVal, xMin, zMin, xMax, zMax):
        minHeight = np.min(heightmap) - 0.01
        maxHeight = np.max(heightmap) + 0.01

        outputPoints = np.empty((heightmap.shape[0], heightmap.shape[1], 3))
        for i in range(heightmap.shape[0]):
            xFrac = float(i) / float(heightmap.shape[0] - 1)
            x = xMin * (1 - xFrac) + xMax * xFrac
            for k in range(heightmap.shape[1]):
                zFrac = float(k) / float(heightmap.shape[1] - 1)
                z = zMin * (1 - zFrac) + zMax * zFrac

                y = (heightmap[i][k] - minHeight) / (maxHeight - minHeight)
                outputPoints[i][k][0] = x
                outputPoints[i][k][1] = y
                outputPoints[i][k][2] = z

        mergeTolerance = cellSizeVal / 3.0
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
