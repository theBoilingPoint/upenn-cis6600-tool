from datetime import datetime

import math
import cv2
from PIL import Image
import numpy as np
from typing import Tuple

import maya.api.OpenMaya as om
import maya.cmds as cmds
import maya.mel as mm

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

    def __init__(self, cellSize: float, targetGridScale: Tuple[float, float], upliftMapFile: str, minUpliftRatio: float, 
                 relUpliftScale: float, relErosionScale: float, waterHalfRetentionDist: float):
        self._cellSize: float = cellSize
        self._targetGridScale: Tuple[float, float] = targetGridScale
        self._upliftMapFile: str = upliftMapFile
        self._minUpliftRatio: float = minUpliftRatio
        self._relUpliftScale: float = relUpliftScale
        self._relErosionScale: float = relErosionScale
        self._waterHalfRetentionDist: float = waterHalfRetentionDist
        self._upliftMap = None
        self.gridShape: Tuple[int, int] = self.computeGridShape()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TerroderSimulationParameters):
            return False

        return (math.isclose(self.cellSize, other.cellSize)
            and self.gridShape == other.gridShape
            and self.upliftMapFile == other.upliftMapFile
            and math.isclose(self.minUpliftRatio, other.minUpliftRatio)
            and math.isclose(self.relUpliftScale, other.relUpliftScale)
            and math.isclose(self.relErosionScale, other.relErosionScale)
            and math.isclose(self.waterHalfRetentionDist, other.waterHalfRetentionDist))
        

    def computeGridShape(self) -> Tuple[int, int]:
        return (max(int(math.ceil(self._targetGridScale[0] / self.cellSize)), 4),
                max(int(math.ceil(self._targetGridScale[1] / self.cellSize)), 4))

    @property
    def targetGridScale(self) -> Tuple[float, float]:
        return self._targetGridScale

    def setTargetGridScale(self, targetGridScale: Tuple[float, float]) -> None:
        self._targetGridScale = targetGridScale
        self.gridShape = self.computeGridShape()

    @property
    def cellSize(self) -> float:
        return self._cellSize

    def setCellSize(self, cellSize: float) -> None:
        self._cellSize = cellSize
        self.gridShape = self.computeGridShape()

    @property
    def upliftMapFile(self) -> str:
        return self._upliftMapFile

    def setUpliftMapFile(self, upliftMapFile: str) -> None:
        self._upliftMapFile = upliftMapFile
        self._upliftMap = None

    @property
    def minUpliftRatio(self) -> float:
        return self._minUpliftRatio

    def setMinUpliftRatio(self, minUpliftRatio: float) -> None:
        self._minUpliftRatio = minUpliftRatio

    @property
    def relUpliftScale(self) -> float:
        return self._relUpliftScale

    def setRelUpliftScale(self, relUpliftScale: float) -> None:
        self._relUpliftScale = relUpliftScale

    @property
    def relErosionScale(self) -> float:
        return self._relErosionScale

    def setErosionScale(self, relErosionScale: float) -> None:
        self._relErosionScale = relErosionScale

    @property
    def waterHalfRetentionDist(self) -> float:
        return self._waterHalfRetentionDist

    def setWaterHalfRetentionDist(self, waterHalfRetentionDist: float) -> None:
        self._waterHalfRetentionDist = waterHalfRetentionDist

    @property
    def upliftMap(self) -> np.ndarray:
        if self._upliftMap is None:
            if len(self._upliftMapFile) > 0:
                try:
                    with Image.open(self.upliftMapFile) as image:
                        # Read pixels directly first
                        imagePixels = np.empty(image.size)
                        gsImage = image.convert("L")  # grayscale
                        for i in range(image.size[0]):
                            for k in range(image.size[1]):
                                imagePixels[i][k] = (gsImage.getpixel((i, k)) / 255.0)  # grayscale color seems to be 0-255

                        cv2shape = (self.gridShape[1], self.gridShape[0])  # because cv2 flips x/y compared to numpy
                        self._upliftMap = cv2.resize(imagePixels, cv2shape)  # resize/interpolate to fit gridShape
                        self._upliftMap = (self.minUpliftRatio + (1.0 - self.minUpliftRatio) * self._upliftMap)
                        self._upliftMap = np.clip(self._upliftMap, 0.0, 1.0)
                except FileNotFoundError:
                    om.MGlobal.displayWarning(f'File "{self.upliftMapFile}" not found.')
                    self._upliftMap = np.zeros(self.gridShape)
            else:
                self._upliftMap = np.zeros(self.gridShape)

        return self._upliftMap


class TerroderUI(object):
    """
    Static class to organize UI info
    """
    TYPE_NAME = "TerroderNode"
    TIME_ATTR_LONG_NAME = "time"
    OUTPUT_MESH_ATTR_LONG_NAME = "outputMesh"

    minWidth = 400
    createdMenuName = ""
    scrollListName = ""
    selectedListItem = None
    selectedNodeName = None
    terroderNodeName = ""

    @staticmethod
    def createMenu():
        mainWindowName = mm.eval("string $temp = $gMainWindow;")
        TerroderUI.createdMenuName = cmds.menu(l="Terroder", p=mainWindowName)
        cmds.menuItem(l="Create Terroder Mesh", p=TerroderUI.createdMenuName, c=TerroderUI._createNode)
        cmds.menuItem(l="Manage Timestamps", p=TerroderUI.createdMenuName, c=TerroderUI._createSavePointWindow)
        cmds.menuItem(l="Add River Source", p=TerroderUI.createdMenuName, c=TerroderNode._addRiverSources)
        cmds.menuItem(l="Help", p=TerroderUI.createdMenuName, c=TerroderUI._showHelpMenu)
        
    @staticmethod
    def destroyMenu():
        cmds.deleteUI(TerroderUI.createdMenuName)
        TerroderUI.createdMenuName = ""

    @staticmethod
    def _createNode(*args) -> None:
        terrainTransformNodeName = cmds.createNode("transform", n="terrainTransform#")
        terrainVisibleMeshNodeName = cmds.createNode("mesh", parent=terrainTransformNodeName, n="terrainMesh#")

        # Create a lambert shader to shade the output
        lambertShaderName = cmds.shadingNode("lambert", asShader=True, name="lambert_shader#")
        lambertShaderSet = cmds.sets(renderable=True, noSurfaceShader=True, empty=True)
        cmds.connectAttr(lambertShaderName + ".outColor", lambertShaderSet + ".surfaceShader", force=True)
        lambertShadingGroup = cmds.listConnections(lambertShaderName, type="shadingEngine")[-1]
        cmds.sets(terrainVisibleMeshNodeName, fe=lambertShadingGroup)

        # terroderNodeName = cmds.createNode(TerroderNode.TYPE_NAME)
        TerroderUI.terroderNodeName = cmds.createNode(TerroderNode.TYPE_NAME)

        cmds.setAttr(f"{TerroderUI.terroderNodeName}.toggleSaveTimestamp", lock=True)
        cmds.setAttr(f"{TerroderUI.terroderNodeName}.toggleLoadTimestamp", lock=True)
        cmds.setAttr(f"{TerroderUI.terroderNodeName}.toggleStartNewSimulation", lock=True)
        cmds.setAttr(f"{TerroderUI.terroderNodeName}.toggleDeleteTimestamp", lock=True)

        cmds.connectAttr("time1.outTime", f"{TerroderUI.terroderNodeName}.{TerroderNode.TIME_ATTR_LONG_NAME}")
        cmds.connectAttr(f"{TerroderUI.terroderNodeName}.{TerroderNode.OUTPUT_MESH_ATTR_LONG_NAME}", f"{terrainVisibleMeshNodeName}.inMesh")

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

    @staticmethod
    def _toggleValue(nodeName, attrName):
        cmds.setAttr(f"{nodeName}.{attrName}", lock=False)

        prevVal = cmds.getAttr(f"{nodeName}.{attrName}")
        cmds.setAttr(f"{nodeName}.{attrName}", not prevVal)

        cmds.setAttr(f"{nodeName}.{attrName}", lock=True)

    @staticmethod
    def _toggleNodeAttribute(attrName):
        selected = cmds.ls(type=["TerroderNode"]) or []

        if len(selected) == 0:
            print("No TerroderNode selected.")
            return

        if len(selected) != 1:
            print("Please only select one TerroderNode.")
            return

        TerroderUI.selectedNodeName = selected[0]

        if attrName == "saveTimestamp":
            TerroderUI._toggleValue(selected[0], "toggleSaveTimestamp")
        elif attrName == "loadTimestamp":
            TerroderUI.selectedListItem = cmds.iconTextScrollList(TerroderUI.scrollListName, q=1, si=1)
            TerroderUI._toggleValue(selected[0], "toggleLoadTimestamp")
        elif attrName == "startNewSimulation":
            TerroderUI._toggleValue(selected[0], "toggleStartNewSimulation")
        elif attrName == "deleteTimestamp":
            TerroderUI.selectedListItem = cmds.iconTextScrollList(TerroderUI.scrollListName, q=1, si=1)
            TerroderUI._toggleValue(selected[0], "toggleDeleteTimestamp")
        else:
            print(f"Node {selected[0]} doesn't have attribute {attrName}.")
            return

        cmds.iconTextScrollList(TerroderUI.getScrollListName(), edit=True, deselectAll=True)

    @staticmethod
    def _toggleSaveTimestamp(*args) -> None:
        TerroderUI._toggleNodeAttribute("saveTimestamp")

    @staticmethod
    def _toggleLoadTimestamp(*args) -> None:
        TerroderUI._toggleNodeAttribute("loadTimestamp")

    @staticmethod
    def _toggleDeleteTimestamp(*args) -> None:
        TerroderUI._toggleNodeAttribute("deleteTimestamp")

    @staticmethod
    def _toggleStartNewSimulation(*args) -> None:
        TerroderUI._toggleNodeAttribute("startNewSimulation")

    @staticmethod
    def _createSavePointWindow(*args) -> None:
        if cmds.window("savePointWindow", exists=True):
            # cmds.deleteUI("savePointWindow")
            cmds.showWindow("savePointWindow")
            return

        cmds.window("savePointWindow", title="Saved Terroder Meshes", width=TerroderUI.minWidth, retain=True)

        cmds.menuBarLayout()  # Need to call this first before adding a menu
        # parent = cmds.menu(label="Mesh List")
        # cmds.menuItem(label="New")

        cmds.paneLayout()
        TerroderUI.scrollListName = cmds.iconTextScrollList(numberOfRows=8, allowMultiSelection=False, doubleClickCommand=TerroderUI._toggleLoadTimestamp)
        cmds.setParent("..")

        cmds.columnLayout()
        cmds.button(label="Save Timestamp", command=TerroderUI._toggleSaveTimestamp, width=TerroderUI.minWidth)
        cmds.button(label="Load Timestamp", command=TerroderUI._toggleLoadTimestamp, width=TerroderUI.minWidth)
        
        cmds.button(label="Start New Simulation", command=TerroderUI._toggleStartNewSimulation, width=TerroderUI.minWidth)
        cmds.button(label="Delete Timestamp", command=TerroderUI._toggleDeleteTimestamp, width=TerroderUI.minWidth)

        # Get current time on timeline as float
        cmds.setParent("..")

        cmds.showWindow("savePointWindow")

    @staticmethod
    def _showHelpMenu(*args):
        defaultMsg = [
            "This is Terroder, a plugin for creating realistic terrain.",
            "The TerroderNode is a dependency graph node and can be accessed from the dependency graph or the attribute editor.",
            "Alter the properties in the attribute editor to change the appearance of the terrain.",
            "To see colored texturing: select the output mesh and use Mesh Display > Toggle Display Colors Attribute."
        ]

        manageTimestampsMsg = [
            "Save or load the mesh and parameters of the currently selected timestamp. A saved timestamp can also be deleted.",
            "Once a timestamp is loaded, it will be highlighted in green.",
            "All changes to the attributes will be applied on the loaded timestamp unless the Start New Simulation button is clicked.",
        ]

        if cmds.window("helpWindow", exists=True):
            cmds.deleteUI("helpWindow")

        cmds.window("helpWindow", title="Terroder Help", width=TerroderUI.minWidth, sizeable=False)

        mainTitleCol = [0.5, 0.5, 0.5]
        subTitleCol = [0.35, 0.35, 0.35]
        cmds.columnLayout(adjustableColumn=True)
        # Default message
        cmds.text(label="\n".join(defaultMsg), align="left")
        cmds.text(label="\n", align="left")
        # Menu bar message
        cmds.text(label="Menu Bar Items", align="left", font="boldLabelFont", backgroundColor=mainTitleCol)
        cmds.text(label="Create Terroder Mesh", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="Create a mesh terrain connected to a TerroderNode.", align="left")

        cmds.text(label="Manage Timestamps", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="\n".join(manageTimestampsMsg), align="left")

        cmds.text(label="Help", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="Invoke the help menu.", align="left")
        cmds.text(label="\n", align="left")
        # Attribute bar message
        cmds.text(label="Node Attributes", align="left", font="boldLabelFont", backgroundColor=mainTitleCol)
        cmds.text(label="Time", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="The currently selected timestamp.", align="left")

        cmds.text(label="Uplift Map File", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="The uplift map texture to guide the simulation.", align="left")

        cmds.text(label="Min Uplift Ratio", align="left", font="smallBoldLabelFont",backgroundColor=subTitleCol,)
        cmds.text(label="Rescales the uplift found in the texture. The texture is grayscale is normally in the range [0,1]. This attribute rescales the range to [min uplift ratio, 1].",
                  align="left")

        cmds.text(label="Cell Size", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="The size of a grid for the terrain mesh.", align="left")

        cmds.text(label="Grid Scale X", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="The width of the terrain mesh.", align="left")

        cmds.text(label="Grid Scale Z", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="The height of the terrain mesh.", align="left")

        cmds.text(label="Uplift Relative Scale", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="Multiply the uplift term in the simulation by the provided value.", align="left")

        cmds.text(label="Erosion Relative Scale", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="Multiply the erosion term in the simulation by the provided value.", align="left")

        cmds.text(label="Water Retention Constant", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="Affects the decay of water in the drainage area computation. The higher the value, the slower the decay.", align="left")

        cmds.text(label="Snow Height", align="left", font="smallBoldLabelFont", backgroundColor=subTitleCol)
        cmds.text(label="For texturing purposes only. Parts of the terrain higher the provided value will appear snowy.", align="left")

        cmds.showWindow("helpWindow")

    @staticmethod
    def getScrollListName():
        return TerroderUI.scrollListName

    @staticmethod
    def getSelectedListItem():
        return TerroderUI.selectedListItem

    @staticmethod
    def getSelectedNodeName():
        return TerroderUI.selectedNodeName

    @staticmethod
    def getDefaultTextColour():
        return [1.0, 1.0, 1.0]

    @staticmethod
    def getSelectedTextColour():
        return [0.0, 1.0, 0.0]

class TerroderNode(om.MPxNode):
    # constants
    TYPE_NAME = "TerroderNode"
    ID = om.MTypeId(0x0008099A)
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
    SEQUENCE_NUMBER_ATTR_LONG_NAME = "sequenceNumber"
    SEQUENCE_NUMBER_ATTR_SHORT_NAME = "sn"

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
    aSnowHeight = None
    aSequenceNumber = None

    toggleSaveTimestamp = None
    toggleLoadTimestamp = None
    toggleStartNewSimulation = None
    toggleDeleteTimestamp = None

    # output
    aOutputMesh = None

    # For loading
    savedHeightMaps = dict()
    retrievedHeightMap = None

    riverSourcePositions = []

    def __init__(self):
        om.MPxNode.__init__(self)

        self.simParams: TerroderSimulationParameters = None
        self.heightMapTs: list[np.ndarray] = [] # element i is the heightmap after i iterations

        self.prevTimestampToggleValue = False
        self.prevLoadTimestampToggleValue = False
        self.prevStartNewSimulationToggleValue = False
        self.prevDeleteTimestampToggleValue = False
        self.loadedKey = None

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
        nAttr.default = 0.0
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

        TerroderNode.aWaterHalfRetentionDistance = nAttr.create("waterRetentionConstant", "wrc", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 1.0
        nAttr.setMin(0.001)

        TerroderNode.aSnowHeight = nAttr.create("snowHeight", "sh", om.MFnNumericData.kFloat)
        MAKE_INPUT(nAttr)
        nAttr.default = 0.25
        nAttr.setMin(0.001)
        nAttr.setMax(1.0)

        TerroderNode.aSequenceNumber = nAttr.create(TerroderNode.SEQUENCE_NUMBER_ATTR_LONG_NAME, 
                                                    TerroderNode.SEQUENCE_NUMBER_ATTR_SHORT_NAME, 
                                                    om.MFnNumericData.kLong)
        MAKE_INPUT(nAttr)
        nAttr.default = 0
        nAttr.hidden = True

        TerroderNode.toggleSaveTimestamp = nAttr.create("toggleSaveTimestamp", "tst", om.MFnNumericData.kBoolean)
        MAKE_INPUT(nAttr)
        nAttr.default = False

        TerroderNode.toggleLoadTimestamp = nAttr.create("toggleLoadTimestamp", "tlt", om.MFnNumericData.kBoolean)
        MAKE_INPUT(nAttr)
        nAttr.default = False

        TerroderNode.toggleStartNewSimulation = nAttr.create("toggleStartNewSimulation", "tsns", om.MFnNumericData.kBoolean)
        MAKE_INPUT(nAttr)
        nAttr.default = False

        TerroderNode.toggleDeleteTimestamp = nAttr.create("toggleDeleteTimestamp", "tdt", om.MFnNumericData.kBoolean)
        MAKE_INPUT(nAttr)
        nAttr.default = False

        # output
        TerroderNode.aOutputMesh = tAttr.create(TerroderNode.OUTPUT_MESH_ATTR_LONG_NAME, 
                                                TerroderNode.OUTPUT_MESH_ATTR_SHORT_NAME, 
                                                om.MFnData.kMesh)
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
        om.MPxNode.addAttribute(TerroderNode.aSnowHeight)
        om.MPxNode.addAttribute(TerroderNode.aSequenceNumber)

        om.MPxNode.addAttribute(TerroderNode.toggleSaveTimestamp)
        om.MPxNode.addAttribute(TerroderNode.toggleLoadTimestamp)
        om.MPxNode.addAttribute(TerroderNode.toggleStartNewSimulation)
        om.MPxNode.addAttribute(TerroderNode.toggleDeleteTimestamp)

        om.MPxNode.attributeAffects(TerroderNode.aTime, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aUpliftMapFile, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aMinUpliftRatio, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aCellSize, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aGridScaleX, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aGridScaleZ, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aUpliftRelScale, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aErosionRelScale, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aWaterHalfRetentionDistance, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aSnowHeight, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.aSequenceNumber, TerroderNode.aOutputMesh)

        om.MPxNode.attributeAffects(TerroderNode.toggleSaveTimestamp, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.toggleLoadTimestamp, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.toggleStartNewSimulation, TerroderNode.aOutputMesh)
        om.MPxNode.attributeAffects(TerroderNode.toggleDeleteTimestamp, TerroderNode.aOutputMesh)

        print("[DEBUG] TerroderNode initialised.\n")

    def loadSavedAttributes(self, newSimParams, key):
        upliftMapFile = TerroderNode.savedHeightMaps[key][1].upliftMapFile
        minUpliftRatio = TerroderNode.savedHeightMaps[key][1].minUpliftRatio
        cellSize = TerroderNode.savedHeightMaps[key][1].cellSize
        gridScale = TerroderNode.savedHeightMaps[key][1].targetGridScale
        relUpliftScale = TerroderNode.savedHeightMaps[key][1].relUpliftScale
        relErosionScale = TerroderNode.savedHeightMaps[key][1].relErosionScale
        waterHalfRetentionDist = TerroderNode.savedHeightMaps[key][1].waterHalfRetentionDist

        newSimParams.setUpliftMapFile(upliftMapFile)
        newSimParams.setMinUpliftRatio(minUpliftRatio)
        newSimParams.setCellSize(cellSize)
        newSimParams.setTargetGridScale(gridScale)
        newSimParams.setRelUpliftScale(relUpliftScale)
        newSimParams.setErosionScale(relErosionScale)
        newSimParams.setWaterHalfRetentionDist(waterHalfRetentionDist)

        cmds.setAttr(f"{TerroderUI.getSelectedNodeName()}.upliftMapFile", upliftMapFile, type="string")
        cmds.setAttr(f"{TerroderUI.getSelectedNodeName()}.minUpliftRatio", minUpliftRatio)
        cmds.setAttr(f"{TerroderUI.getSelectedNodeName()}.cellSize", cellSize)
        cmds.setAttr(f"{TerroderUI.getSelectedNodeName()}.gridScaleX", gridScale[0])
        cmds.setAttr(f"{TerroderUI.getSelectedNodeName()}.gridScaleZ", gridScale[1])
        cmds.setAttr(f"{TerroderUI.getSelectedNodeName()}.upliftRelativeScale", relUpliftScale)
        cmds.setAttr(f"{TerroderUI.getSelectedNodeName()}.erosionRelativeScale", relErosionScale)
        cmds.setAttr(f"{TerroderUI.getSelectedNodeName()}.waterRetentionConstant", waterHalfRetentionDist)

    def compute(self, plug: om.MPlug, dataBlock: om.MDataBlock):
        if (plug != TerroderNode.aOutputMesh) and (plug.parent() != TerroderNode.aOutputMesh):
            return None

        rawTime = dataBlock.inputValue(TerroderNode.aTime).asFloat()
        numIterations = max(0, int(rawTime) - 1)

        # Params for simulation, editable in attribute editor
        avUpliftMapFile = dataBlock.inputValue(TerroderNode.aUpliftMapFile).asString()
        avMinUpliftRatio = dataBlock.inputValue(TerroderNode.aMinUpliftRatio).asFloat()
        avCellSize = dataBlock.inputValue(TerroderNode.aCellSize).asFloat()
        avGridScale = (dataBlock.inputValue(TerroderNode.aGridScaleX).asFloat(), dataBlock.inputValue(TerroderNode.aGridScaleZ).asFloat())
        avRelUpliftScale = dataBlock.inputValue(TerroderNode.aUpliftRelScale).asFloat()
        avRelErosionScale = dataBlock.inputValue(TerroderNode.aErosionRelScale).asFloat()
        avWaterHalfRetentionDist = dataBlock.inputValue(TerroderNode.aWaterHalfRetentionDistance).asFloat()
        avSnowHeight = dataBlock.inputValue(TerroderNode.aSnowHeight).asFloat()
        newSimParams = TerroderSimulationParameters(avCellSize,
                                                    avGridScale,
                                                    avUpliftMapFile,
                                                    avMinUpliftRatio,
                                                    avRelUpliftScale,
                                                    avRelErosionScale,
                                                    avWaterHalfRetentionDist)
        
        # Params for menu button control, not editable in attribute editor
        # TODO: Ideally these attributes should be hidden from the attribute editor
        saveTimestampVal = dataBlock.inputValue(TerroderNode.toggleSaveTimestamp).asBool()
        loadTimestampVal = dataBlock.inputValue(TerroderNode.toggleLoadTimestamp).asBool()
        startNewSimulationVal = dataBlock.inputValue(TerroderNode.toggleStartNewSimulation).asBool()
        deleteTimestampVal = dataBlock.inputValue(TerroderNode.toggleDeleteTimestamp).asBool()

        if saveTimestampVal != self.prevTimestampToggleValue:
            # If we enter this if statement, it means the save timestamp button was pressed
            self.prevTimestampToggleValue = saveTimestampVal
            key = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            time = cmds.currentTime(query=True)  # Get currently selected time on the time line
            TerroderNode.savedHeightMaps[key] = (self.heightMapTs[int(time)], newSimParams)
            cmds.iconTextScrollList(TerroderUI.getScrollListName(), edit=True, append={key})
            cmds.iconTextScrollList(TerroderUI.getScrollListName(), 
                                    edit=True, 
                                    itemTextColor=[TerroderNode.getKeyIndex(key)] + TerroderUI.getDefaultTextColour())

        if loadTimestampVal != self.prevLoadTimestampToggleValue:
            # If we enter this if statement, it means the load timestamp button was pressed
            self.prevLoadTimestampToggleValue = loadTimestampVal
            selectedItemList = TerroderUI.getSelectedListItem()
            if selectedItemList:
                key = selectedItemList[0]
                TerroderNode.retrievedHeightMap = TerroderNode.savedHeightMaps[key][0]
                self.loadSavedAttributes(newSimParams, key)
                self.heightMapTs = [TerroderNode.retrievedHeightMap]

                if self.loadedKey is not None:
                    cmds.iconTextScrollList(TerroderUI.getScrollListName(),
                                            edit=True,
                                            itemTextColor=[TerroderNode.getKeyIndex(self.loadedKey)] + TerroderUI.getDefaultTextColour())

                self.loadedKey = key

                cmds.iconTextScrollList(TerroderUI.getScrollListName(),
                                        edit=True,
                                        itemTextColor=[TerroderNode.getKeyIndex(self.loadedKey)] + TerroderUI.getSelectedTextColour())

                print("Loaded heightmap with key: ", key)
            else:
                print(" No timestamp is selected from the list.")

        if startNewSimulationVal != self.prevStartNewSimulationToggleValue:
            # If we enter this if statement, it means the start new simulation button was pressed
            self.prevStartNewSimulationToggleValue = startNewSimulationVal
            TerroderNode.retrievedHeightMap = None
            self.heightMapTs = []
            if self.loadedKey:
                cmds.iconTextScrollList(TerroderUI.getScrollListName(), 
                                        edit=True,
                                        itemTextColor=[TerroderNode.getKeyIndex(self.loadedKey)] + TerroderUI.getDefaultTextColour())
            self.loadedKey = None
                    
        if deleteTimestampVal != self.prevDeleteTimestampToggleValue:
            self.prevDeleteTimestampToggleValue = deleteTimestampVal
            selectedItemList = TerroderUI.getSelectedListItem()
            if selectedItemList:
                key = selectedItemList[0]
                del TerroderNode.savedHeightMaps[key]
                cmds.iconTextScrollList(TerroderUI.getScrollListName(), edit=True, removeAll=True)
                TerroderNode._loadAllSavedHeightMaps()

                if key != self.loadedKey and self.loadedKey is not None:
                    cmds.iconTextScrollList(TerroderUI.getScrollListName(),
                                            edit=True,
                                            itemTextColor=[TerroderNode.getKeyIndex(self.loadedKey)] + TerroderUI.getSelectedTextColour())
                else:
                    TerroderNode.retrievedHeightMap = None
                    self.loadedKey = None

                print("Deleted heightmap with key: ", key)
            else:
                print("No timestamp is selected from the list.")

        if self.simParams is None or newSimParams != self.simParams:
            # Current sim params are out of date; reset
            om.MGlobal.displayInfo("[DEBUG] Using new sim params and resetting simulation")
            self.simParams = newSimParams
            if TerroderNode.retrievedHeightMap is not None:
                cv2shape = (self.simParams.gridShape[1], self.simParams.gridShape[0])  # because cv2 flips x/y compared to numpy
                hm = cv2.resize(TerroderNode.retrievedHeightMap, cv2shape)
                self.heightMapTs = [hm]
                TerroderNode.riverSourcePositions.clear()
            else:
                self.heightMapTs = []
                TerroderNode.riverSourcePositions.clear()

        # Simulate until we have at least numIters iterations
        while numIterations >= len(self.heightMapTs):
            self.simulateNextIteration()

        # Set the output data
        outputMeshHandle: om.MDataHandle = dataBlock.outputValue(TerroderNode.aOutputMesh)
        dataCreator: om.MFnMeshData = om.MFnMeshData()
        outputData: om.MObject = dataCreator.create()
        self.createOutputMesh(outputData, numIterations, avWaterHalfRetentionDist, avSnowHeight)
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
                    neighborHeight = heightMap[ni][nk] if self.cellInBounds((ni, nk)) else TerroderNode.MIN_HEIGHT
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
        cellHeights.sort(key=lambda ch: -ch[2])

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
                relFlow = pow((h - nh) / np.sqrt(di * di + dk * dk), 4)  # only need to compute proportionally due to normalization later
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

    def createOutputMesh(self, outputData: om.MObject, numIterations: int, waterDistanceConstant: float, snowHeight: float):
        if numIterations >= len(self.heightMapTs):
            raise RuntimeError(f"No height map corresponding to {numIterations} iterations.")

        vertices = []
        polygonCounts = []
        polygonConnects = []

        heightMap: np.ndarray = self.heightMapTs[numIterations]
        gridColors: np.ndarray = self.getGridColors(heightMap, waterDistanceConstant, snowHeight)

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
                vertexIndices = [indexMap[(i, k)], indexMap[(i, k + 1)], indexMap[(i + 1, k + 1)], indexMap[(i + 1, k)]]
                for vertexIndex in vertexIndices:
                    polygonConnects.append(vertexIndex)

        fnMesh = om.MFnMesh()
        meshObj: om.MObject = fnMesh.create(vertices, polygonCounts, polygonConnects, parent=outputData)

        for (i, k), vi in indexMap.items():
            color = om.MColor(gridColors[i][k])
            fnMesh.setVertexColor(color, vi)

        return meshObj

    def getGridColors(self, heightMap: np.ndarray, waterDistanceConstant: float, snowHeight: float) -> np.ndarray:
        # Nothing originally water
        isWater = np.full(heightMap.shape, False, dtype=np.bool8)
        xMin = -0.5 * self.simParams.cellSize * (heightMap.shape[0] - 1)
        zMin = -0.5 * self.simParams.cellSize * (heightMap.shape[1] - 1)
        riverSourceIndices = set()
        for x, _, z in TerroderNode.riverSourcePositions:
            i = int(round((x - xMin) / self.simParams.cellSize))
            k = int(round((z - zMin) / self.simParams.cellSize))
            riverSourceIndices.add((i, k))

        # Trace to lower neighbors
        for si, sk in riverSourceIndices:
            i, k = si, sk

            while not isWater[i][k]:
                isWater[i][k] = True

                minHeight = heightMap[i][k]
                minNeighbor = None
                for ni, nk in [c for c in self.getNeighborCells((i, k)) if self.cellInBounds(c)]:
                    if heightMap[ni][nk] <= minHeight:
                        minHeight = heightMap[ni][nk]
                        minNeighbor = ni, nk
                
                if minNeighbor is None:
                    break

                i, k = minNeighbor

        # Then mark each cell with its distance from water to measure its wetness
        distanceFromWater = np.full(heightMap.shape, -1, dtype=np.int32)
        frontier = []
        for i in range(isWater.shape[0]):
            for k in range(isWater.shape[1]):
                if isWater[i][k]:
                    distanceFromWater[i][k] = 0
                    frontier.append((i, k))
        curDistance = 0
        while len(frontier) > 0:
            curDistance += 1
            next_frontier = []
            for i, k in frontier:
                for ni, nk in [c for c in self.getNeighborCells((i, k)) if self.cellInBounds(c)]:
                    if distanceFromWater[ni][nk] >= 0:
                        continue  # already reached here
                    distanceFromWater[ni][nk] = curDistance
                    next_frontier.append((ni, nk))

            frontier = next_frontier

        waterColor = np.array([65.0, 107.0, 223.0]) / 255.0
        desertColor = np.array([238.0, 206.0, 130.0]) / 255.0
        wetlandColor = np.array([76.0, 187.0, 23.0]) / 255.0
        mountainColor = np.array([0.91, 0.89, 0.86])
        snowColor = np.array([243.0, 252.0, 255.0]) / 255.0

        # Water cells are colored with water; snow cells are colored with snow
        # For other cells, interpolate between wetland/desert colors based on wetness, then interpolate with mountain color based on height
        colors: np.ndarray = np.empty((heightMap.shape[0], heightMap.shape[1], 3))
        for i in range(colors.shape[0]):
            for k in range(colors.shape[1]):
                if isWater[i][k]:
                    colors[i][k][:] = waterColor
                    continue
                if heightMap[i][k] > snowHeight:
                    colors[i][k][:] = snowColor
                    continue

                relDistance = distanceFromWater[i][k] * self.simParams.cellSize / waterDistanceConstant
                wetness = max(1.0 - (relDistance / 8), 0)

                color = (1.0 - wetness) * desertColor + wetness * wetlandColor  # wetness interpolation
                heightFraction = heightMap[i][k] / snowHeight
                color = (1.0 - heightFraction) * color + heightFraction * mountainColor  # height/mountain-ness interpolation
                colors[i][k][:] = np.array(color)

        return colors

    def cellInBounds(self, cell: Tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.simParams.gridShape[0] and 0 <= cell[1] < self.simParams.gridShape[1]
        

    def getNeighborCells(self, cell: Tuple[int, int]) -> list:
        return [(cell[0] + di, cell[1] + dk) for (di, dk) in TerroderNode.NEIGHBOR_ORDER]

    @staticmethod
    def getSavedHeightMaps():
        return TerroderNode.savedHeightMaps

    @staticmethod
    def getKeyIndex(key):
        # In Python 3, all dicts are ordered by default, so we can just get the index of the key
        # The assumption is that the order of the items in the scroll list is the same as the order of the keys in the dict
        return list(TerroderNode.savedHeightMaps.keys()).index(key) + 1
    
    @staticmethod
    def _loadAllSavedHeightMaps(*args) -> None:
        for key in TerroderNode.getSavedHeightMaps().keys():
            cmds.iconTextScrollList(TerroderUI.scrollListName, edit=True, append={key})
            idx = TerroderNode.getKeyIndex(key)
            cmds.iconTextScrollList(TerroderUI.scrollListName,
                                    edit=True,
                                    itemTextColor=[idx] + TerroderUI.getDefaultTextColour())
    
    
    @staticmethod
    def _addRiverSources(*args) -> None:
        selVertices = cmds.filterExpand(sm=31)

        if len(selVertices) == 0:
            cmds.confirmDialog(t="Add River Source", b="OK", db="OK", cb="OK", m="At least one vertex of the terrain mesh need to be selected.")
            return
        
        for v in selVertices:
            TerroderNode.riverSourcePositions.append(cmds.pointPosition(v, w=True))
        
        
        sn = cmds.getAttr(f"{TerroderUI.terroderNodeName}.{TerroderNode.SEQUENCE_NUMBER_ATTR_LONG_NAME}")
        cmds.setAttr(f"{TerroderUI.terroderNodeName}.{TerroderNode.SEQUENCE_NUMBER_ATTR_LONG_NAME}", sn + 1)                  

# initialize the script plug-in
def initializePlugin(mObject):
    mPlugin = om.MFnPlugin(mObject, "Company", "1.0", "Any")
    # Don't use try except bc otherwise the error message won't be printed
    mPlugin.registerNode(TerroderNode.TYPE_NAME, 
                         TerroderNode.ID,
                         TerroderNode.create,
                         TerroderNode.initialize,
                         om.MPxNode.kDependNode)

    TerroderUI.createMenu()


# uninitialize the script plug-in
def uninitializePlugin(mObject):
    TerroderUI.destroyMenu()

    mplugin = om.MFnPlugin(mObject)
    # Don't use try except bc otherwise the error message won't be printed
    mplugin.deregisterNode(TerroderNode.ID)
