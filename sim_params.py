import maya.api.OpenMaya as om
import math
import numpy as np
import cv2
from typing import Tuple
from PIL import Image

class TerroderSimulationParameters(object):
    # Intended to be read-only. If you need to change values, you need a new TerroderSimulationParameters object.
    # All parameters that can be changed by the user
    # Does NOT include time; if time alone changes, we might not need to redo the entire simulation

    def __init__(
        self,
        cellSize: float,
        targetGridScale: Tuple[float, float],
        upliftMapFile: str,
        minUpliftRatio: float,
        relUpliftScale: float,
        relErosionScale: float,
        waterHalfRetentionDist: float,
    ):
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

        return (
            math.isclose(self.cellSize, other.cellSize)
            and self.gridShape == other.gridShape
            and self.upliftMapFile == other.upliftMapFile
            and math.isclose(self.minUpliftRatio, other.minUpliftRatio)
            and math.isclose(self.relUpliftScale, other.relUpliftScale)
            and math.isclose(self.relErosionScale, other.relErosionScale)
            and math.isclose(self.waterHalfRetentionDist, other.waterHalfRetentionDist)
        )

    def computeGridShape(self) -> Tuple[int, int]:
        return (
            max(int(math.ceil(self._targetGridScale[0] / self.cellSize)), 4),
            max(int(math.ceil(self._targetGridScale[1] / self.cellSize)), 4),
        )

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
                                imagePixels[i][k] = (
                                    gsImage.getpixel((i, k)) / 255.0
                                )  # grayscale color seems to be 0-255

                        cv2shape = (
                            self.gridShape[1],
                            self.gridShape[0],
                        )  # because cv2 flips x/y compared to numpy
                        self._upliftMap = cv2.resize(
                            imagePixels, cv2shape
                        )  # resize/interpolate to fit gridShape
                        self._upliftMap = (
                            self.minUpliftRatio
                            + (1.0 - self.minUpliftRatio) * self._upliftMap
                        )
                        self._upliftMap = np.clip(self._upliftMap, 0.0, 1.0)
                except FileNotFoundError:
                    om.MGlobal.displayWarning(f'File "{self.upliftMapFile}" not found.')
                    self._upliftMap = np.zeros(self.gridShape)
            else:
                self._upliftMap = np.zeros(self.gridShape)

        return self._upliftMap