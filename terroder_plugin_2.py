import sys

# Uses Maya's Python API 2.0
# reference: https://help.autodesk.com/view/MAYAUL/2022/ENU/?guid=Maya_SDK_py_ref_index_html
import maya.api.OpenMaya as om;
import maya.cmds as cmds;
import numpy as np;
import math;

# Import the Python wrappers for MEL commands

# The name of the command. 

class TerrodeCommand(om.MPxCommand):
    NAME = "terrode"
    CELL_SIZE_FLAG = "-cs"
    CELL_SIZE_LONG_FLAG = "-cellSize"

    def __init__(self):
        om.MPxCommand.__init__(self)

    def doIt(self, args):
        argDb = om.MArgDatabase(self.syntax(), args)

        cellSizeVal = 0.025
        if argDb.isFlagSet(TerrodeCommand.CELL_SIZE_FLAG):
            cellSizeVal = argDb.flagArgumentDouble(TerrodeCommand.CELL_SIZE_FLAG, 0)

        om.MGlobal.displayInfo(f"[DEBUG] grid size: {cellSizeVal}")

        selectedObjNames = cmds.ls(selection = True)
        om.MGlobal.displayInfo(f"[DEBUG] selected object names: [{', '.join(selectedObjNames)}]")
        
        if len(selectedObjNames) != 1:
            om.MGlobal.displayError("There must be exactly one object selected.")
            self.setResult("Did not execute command due to an error.")
            return
        
        selectionList = om.MSelectionList()
        selectionList.add(selectedObjNames[0])
        selectedMesh = om.MFnMesh(selectionList.getDagPath(0))

        bb = cmds.exactWorldBoundingBox(selectedObjNames[0])
        bbMin = bb[0:3]
        bbMax = bb[3:6]
        om.MGlobal.displayInfo(f"[DEBUG] selected object bounding box: {bbMin} to {bbMax}")

        raycastY = bbMax[1] + 0.1  # 0.1 "slack distance"
        xRange = bbMax[0] - bbMin[0]
        yRange = bbMax[1] - bbMin[1]
        zRange = bbMax[2] - bbMin[2]
        gridShape = (int(math.ceil(xRange / cellSizeVal + 0.01)), int(math.ceil(zRange / cellSizeVal + 0.01)))

        om.MGlobal.displayInfo(f"[DEBUG] grid shape: {gridShape}")
        uplift = np.zeros(gridShape)
        xStep = xRange / gridShape[0]
        zStep = zRange / gridShape[1]

        rayDirection = om.MFloatVector(0, -1, 0)
        # max distance is 0.2 + yRange since raycastY is only 0.1 above the bounding box
        raycastDistance = 0.2 + yRange
        for i in range(gridShape[0]):
            x = bbMin[0] + xRange * (i + 0.5) / gridShape[0]
            for k in range(gridShape[1]):
                z = bbMin[2] + zRange * (k + 0.5) / gridShape[1]
                rayOrigin = om.MFloatPoint(x, raycastY, z)

                intersectionResult = selectedMesh.closestIntersection(rayOrigin, rayDirection, om.MSpace.kWorld, raycastDistance, False)
                if intersectionResult is not None:
                    hitPoint = intersectionResult[0]
                    uplift[i][k] = hitPoint.y
        
        # Currently uplift is populated with y coordinates of a grid
        om.MGlobal.displayInfo(f"uplift: {uplift}")

        self.setResult("Executed command")
    
    # Required to parse arguments
    # Needed to add this to avoid crashing
    @staticmethod
    def createSyntax():
        syntax = om.MSyntax()
        syntax.addFlag(TerrodeCommand.CELL_SIZE_FLAG, TerrodeCommand.CELL_SIZE_LONG_FLAG, om.MSyntax.kDouble)
        return syntax
    
    @staticmethod
    def createCommand():
        return TerrodeCommand()

def maya_useNewAPI():
    pass

# Initialize the plugin
def initializePlugin(plugin):
    pluginFn = om.MFnPlugin(plugin)
        # Must register syntaxCreator as well
    pluginFn.registerCommand(TerrodeCommand.NAME, TerrodeCommand.createCommand, TerrodeCommand.createSyntax)

# Uninitialize the plugin
def uninitializePlugin(plugin):
    mplugin = om.MFnPlugin(plugin)
    mplugin.deregisterCommand(TerrodeCommand.NAME)