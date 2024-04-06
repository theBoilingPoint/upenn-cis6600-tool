import numpy as np

def f(x, y):
    return pow(x * x + y * y, 0.8)

NEIGHBOR_ORDER = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def getNeighborCells(shape, cell):
    neighborCells = []
    i, k = cell
    for di, dk in NEIGHBOR_ORDER:
        ni, nk = i + di, k + dk
        if 0 <= ni < shape[0] and 0 <= nk < shape[1]:
            neighborCells.append((ni, nk))
    return neighborCells

def computeDrainageAreaMap(heightMap: np.ndarray, cellSize: float) -> np.ndarray:
    # sort by descending height
    cellHeights = []
    for i in range(heightMap.shape[0]):
        for k in range(heightMap.shape[1]):
            cellHeights.append((i, k, heightMap[i][k]))
    cellHeights.sort(key = lambda ch: -ch[2]) 

    drainageAreaMap = np.ones(heightMap.shape) * cellSize
    for index in range(len(cellHeights)):
        i, k, h = cellHeights[index]
        neighborCells = getNeighborCells(heightMap.shape, (i, k))
        relFlows = {}
        totalRelFlow = 0.0
        for ni, nk in neighborCells:
            nh = heightMap[ni][nk]
            if nh >= h:
                continue
            
            di, dk = ni - i, nk - k
            dist = np.sqrt(di * di + dk * dk)
            relFlow = pow((h - nh) / dist, 4)
            relFlows[(ni, nk)] = relFlow
            totalRelFlow += relFlow

        if len(relFlows) == 0:
            continue

        passage = np.exp(-4 * cellSize)
        for ni, nk in relFlows:
            relFlowProportion = relFlows[(ni, nk)] / totalRelFlow
            drainageAreaMap[ni][nk] += drainageAreaMap[i][k] * relFlowProportion * passage

    return drainageAreaMap


def summarizeDrainageArea(n):
    xvals = [-1.0 + 2.0 * float(i) / float(n-1) for i in range(n)]
    yvals = [-1.0 + 2.0 * float(i) / float(n-1) for i in range(n)]
    heightMap = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            heightMap[i][j] = f(xvals[i], yvals[j])
    
    cellSize = 2.0 / float(n - 1)

    drainageArea = computeDrainageAreaMap(heightMap, cellSize)
    minA = np.min(drainageArea)
    maxA = np.max(drainageArea)
    avgA = np.mean(drainageArea)
    stdA = np.std(drainageArea)
    print(f"n = {n}, maxA = {maxA}, minA = {minA}, avgA = {avgA}, stdevA = {stdA}, max/avg = {maxA/avgA}")

if __name__ == "__main__":
    for n in [int(10 * (1.5 ** i)) for i in range(9)]:
        summarizeDrainageArea(n)
    