import maya.cmds as cmds
import maya.mel as mm

TYPE_NAME = "TerroderNode"
TIME_ATTR_LONG_NAME = "time"
OUTPUT_MESH_ATTR_LONG_NAME = "outputMesh"

class TerroderUI(object):
    """
    Static class to organize UI info
    """

    minWidth = 400
    createdMenuName = ""
    scrollListName = ""
    selectedListItem = None
    selectedNodeName = None

    @staticmethod
    def createMenu():
        mainWindowName = mm.eval("string $temp = $gMainWindow;")
        TerroderUI.createdMenuName = cmds.menu(l="Terroder", p=mainWindowName)
        cmds.menuItem(
            l="Create Terroder Mesh",
            p=TerroderUI.createdMenuName,
            c=TerroderUI._createNode,
        )
        cmds.menuItem(
            l="Manage Timestamps",
            p=TerroderUI.createdMenuName,
            c=TerroderUI._createSavePointWindow,
        )
        cmds.menuItem(
            l="Help", p=TerroderUI.createdMenuName, c=TerroderUI._showHelpMenu
        )

    @staticmethod
    def destroyMenu():
        cmds.deleteUI(TerroderUI.createdMenuName)
        createdMenuName = ""

    @staticmethod
    def _createNode(*args) -> None:
        terrainTransformNodeName = cmds.createNode("transform", n="terrainTransform#")
        terrainVisibleMeshNodeName = cmds.createNode(
            "mesh", parent=terrainTransformNodeName, n="terrainMesh#"
        )

        # Create a lambert shader to shade the output
        lambertShaderName = cmds.shadingNode(
            "lambert", asShader=True, name="lambert_shader#"
        )
        lambertShaderSet = cmds.sets(renderable=True, noSurfaceShader=True, empty=True)
        cmds.connectAttr(
            lambertShaderName + ".outColor",
            lambertShaderSet + ".surfaceShader",
            force=True,
        )
        lambertShadingGroup = cmds.listConnections(
            lambertShaderName, type="shadingEngine"
        )[-1]
        cmds.sets(terrainVisibleMeshNodeName, fe=lambertShadingGroup)

        # terroderNodeName = cmds.createNode(TerroderNode.TYPE_NAME)
        terroderNodeName = cmds.createNode(TYPE_NAME)

        cmds.setAttr(f"{terroderNodeName}.toggleSaveTimestamp", lock=True)
        cmds.setAttr(f"{terroderNodeName}.toggleLoadTimestamp", lock=True)
        cmds.setAttr(f"{terroderNodeName}.toggleStartNewSimulation", lock=True)
        cmds.setAttr(f"{terroderNodeName}.toggleDeleteTimestamp", lock=True)

        cmds.connectAttr(
            # "time1.outTime", f"{terroderNodeName}.{TerroderNode.TIME_ATTR_LONG_NAME}"
            "time1.outTime", f"{terroderNodeName}.{TIME_ATTR_LONG_NAME}"
        )
        cmds.connectAttr(
            # f"{terroderNodeName}.{TerroderNode.OUTPUT_MESH_ATTR_LONG_NAME}",
            f"{terroderNodeName}.{OUTPUT_MESH_ATTR_LONG_NAME}",
            f"{terrainVisibleMeshNodeName}.inMesh",
        )

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
            TerroderUI.selectedListItem = cmds.iconTextScrollList(
                TerroderUI.scrollListName, q=1, si=1
            )
            TerroderUI._toggleValue(selected[0], "toggleLoadTimestamp")
        elif attrName == "startNewSimulation":
            TerroderUI._toggleValue(selected[0], "toggleStartNewSimulation")
        elif attrName == "deleteTimestamp":
            TerroderUI.selectedListItem = cmds.iconTextScrollList(
                TerroderUI.scrollListName, q=1, si=1
            )
            TerroderUI._toggleValue(selected[0], "toggleDeleteTimestamp")
        else:
            print(f"Node {selected[0]} doesn't have attribute {attrName}.")
            return

        cmds.iconTextScrollList(
            TerroderUI.getScrollListName(), edit=True, deselectAll=True
        )

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

        cmds.window(
            "savePointWindow", title="Saved Terroder Meshes", width=TerroderUI.minWidth, retain=True
        )

        cmds.menuBarLayout()  # Need to call this first before adding a menu
        # parent = cmds.menu(label="Mesh List")
        # cmds.menuItem(label="New")

        cmds.paneLayout()
        TerroderUI.scrollListName = cmds.iconTextScrollList(
            numberOfRows=8,
            allowMultiSelection=False,
            doubleClickCommand=TerroderUI._toggleLoadTimestamp,
        )
        cmds.setParent("..")

        cmds.columnLayout()
        cmds.button(
            label="Save Timestamp",
            command=TerroderUI._toggleSaveTimestamp,
            width=TerroderUI.minWidth,
        )
        cmds.button(
            label="Load Timestamp",
            command=TerroderUI._toggleLoadTimestamp,
            width=TerroderUI.minWidth,
        )
        cmds.button(
            label="Start New Simulation",
            command=TerroderUI._toggleStartNewSimulation,
            width=TerroderUI.minWidth,
        )
        cmds.button(
            label="Delete Timestamp",
            command=TerroderUI._toggleDeleteTimestamp,
            width=TerroderUI.minWidth,
        )

        # Get current time on timeline as float
        cmds.setParent("..")

        cmds.showWindow("savePointWindow")

    @staticmethod
    def _showHelpMenu(*args):
        defaultMsg = [
            "This is Terroder, a plugin for creating realistic terrain.",
            "The TerroderNode is a dependency graph node and can be accessed from the dependency graph or the attribute editor.",
            "Alter the properties in the attribute editor to change the appearance of the terrain.",
        ]

        manageTimestampsMsg = [
            "Save or load the mesh and parameters of the currently selected timestamp. A saved timestamp can also be deleted.",
            "Once a timestamp is loaded, it will be highlighted in green.",
            "All changes to the attributes will be applied on the loaded timestamp unless the Start New Simulation button is clicked.",
        ]

        if cmds.window("helpWindow", exists=True):
            cmds.deleteUI("helpWindow")

        cmds.window(
            "helpWindow",
            title="Terroder Help",
            width=TerroderUI.minWidth,
            sizeable=False,
        )

        mainTitleCol = [0.5, 0.5, 0.5]
        subTitleCol = [0.35, 0.35, 0.35]
        cmds.columnLayout(adjustableColumn=True)
        # Default message
        cmds.text(label="\n".join(defaultMsg), align="left")
        cmds.text(label="\n", align="left")
        # Menu bar message
        cmds.text(
            label="Menu Bar Items",
            align="left",
            font="boldLabelFont",
            backgroundColor=mainTitleCol,
        )
        cmds.text(
            label="Create Terroder Mesh",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(
            label="Create a mesh terrain connected to a TerroderNode.", align="left"
        )

        cmds.text(
            label="Manage Timestamps",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(label="\n".join(manageTimestampsMsg), align="left")

        cmds.text(
            label="Help",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(label="Invoke the help menu.", align="left")
        cmds.text(label="\n", align="left")
        # Attribute bar message
        cmds.text(
            label="Node Attributes",
            align="left",
            font="boldLabelFont",
            backgroundColor=mainTitleCol,
        )
        cmds.text(
            label="Time",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(label="The currently selected timestamp.", align="left")

        cmds.text(
            label="Uplift Map File",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(label="The uplift map texture to guide the simulation.", align="left")

        cmds.text(
            label="Min Uplift Ratio",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(
            label="Rescales the uplift found in the texture. The texture is grayscale is normally in the range [0,1]. This attribute rescales the range to [min uplift ratio, 1].",
            align="left",
        )

        cmds.text(
            label="Cell Size",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(label="The size of a grid for the terrain mesh.", align="left")

        cmds.text(
            label="Grid Scale X",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(label="The width of the terrain mesh.", align="left")

        cmds.text(
            label="Grid Scale Z",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(label="The height of the terrain mesh.", align="left")

        cmds.text(
            label="Uplift Relative Scale",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(
            label="Multiply the uplift term in the simulation by the provided value.",
            align="left",
        )

        cmds.text(
            label="Erosion Relative Scale",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(
            label="Multiply the erosion term in the simulation by the provided value.",
            align="left",
        )

        cmds.text(
            label="Water Retention Constant",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(
            label="Affects the decay of water in the drainage area computation. The higher the value, the slower the decay.",
            align="left",
        )

        cmds.text(
            label="Snow Height",
            align="left",
            font="smallBoldLabelFont",
            backgroundColor=subTitleCol,
        )
        cmds.text(
            label="For texturing purposes only. Parts of the terrain higher the provided value will appear snowy.",
            align="left",
        )

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