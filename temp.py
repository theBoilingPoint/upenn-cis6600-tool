class TerroderUpliftReader(object):
    def __init__(self):
        pass

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
    
    def readFromSelectedObject(self):
        selectedObjNames = cmds.ls(selection = True)
        om.MGlobal.displayInfo(f"[DEBUG] selected object names: [{', '.join(selectedObjNames)}]")
        if len(selectedObjNames) != 1:
            om.MGlobal.displayError("There must be exactly one object selected.")
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