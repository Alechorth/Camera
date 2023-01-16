import numpy as np
import cv2
import cv2.aruco as aruco
import math
import depthai as dai
from psutil import virtual_memory
import cython

#region constant
#-----------Aruco--------------
#--- Define Tag   - [cm]
MARKER_SIZE_ROBOT = 7
MARKER_SIZE_MAT = 10
MARKER_SIZE_CAKE  = 3
    
#--- Define text 
BASE_POSITION = 50

#--- Get the camera calibration path
calib_path  = r"C:\\Users\\horth0a\\OneDrive - BOBST\\Apprentissage_auto\\Centre de formation\\AUB_CIE 3\\Camera\\Aruco\\opencv\\"
camera_matrix   = np.loadtxt(calib_path +'cameraMatrix.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path +'cameraDistortion.txt', delimiter=',')

# ---------OAK-D--------------
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
EXTENDE_DISPARITY = True
# Better accuracy for longer distance, fractional disparity 32-levels:
SUBPIXEL = False
# Better handling for occlusions:
LR_CHECK = True
#endregion


#OAK-D camera configuration
class StereoConfigHandler:

    
    class Trackbar:
        def __init__(self, trackbarName, windowName, minValue, maxValue, defaultValue, handler):
            self.min = minValue
            self.max = maxValue
            self.windowName = windowName
            self.trackbarName = trackbarName
            cv2.createTrackbar(trackbarName, windowName, minValue, maxValue, handler)
            cv2.setTrackbarPos(trackbarName, windowName, defaultValue)

        def set(self, value):
            if value < self.min:
                value = self.min
                print(f'{self.trackbarName} min value is {self.min}')
            if value > self.max:
                value = self.max
                print(f'{self.trackbarName} max value is {self.max}')
            cv2.setTrackbarPos(self.trackbarName, self.windowName, value)


    class CensusMaskHandler:

        stateColor = [(0, 0, 0), (255, 255, 255)]
        gridHeight = 50
        gridWidth = 50

        def fillRectangle(self, row, col):
            src = self.gridList[row][col]["topLeft"]
            dst = self.gridList[row][col]["bottomRight"]

            stateColor = self.stateColor[1] if self.gridList[row][col]["state"] else self.stateColor[0]
            self.changed = True

            cv2.rectangle(self.gridImage, src, dst, stateColor, -1)
            cv2.imshow(self.windowName, self.gridImage)


        def clickCallback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                col = x * (self.gridSize[1]) // self.width
                row = y * (self.gridSize[0]) // self.height
                self.gridList[row][col]["state"] = not self.gridList[row][col]["state"]
                self.fillRectangle(row, col)


        def __init__(self, windowName, gridSize):
            self.gridSize = gridSize
            self.windowName = windowName
            self.changed = False

            cv2.namedWindow(self.windowName)

            self.width = StereoConfigHandler.CensusMaskHandler.gridWidth * self.gridSize[1]
            self.height = StereoConfigHandler.CensusMaskHandler.gridHeight * self.gridSize[0]

            self.gridImage = np.zeros((self.height + 50, self.width, 3), np.uint8)

            cv2.putText(self.gridImage, "Click on grid to change mask!", (0, self.height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
            cv2.putText(self.gridImage, "White: ON   |   Black: OFF", (0, self.height+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

            self.gridList = [[dict() for _ in range(self.gridSize[1])] for _ in range(self.gridSize[0])]

            for row in range(self.gridSize[0]):
                rowFactor = self.height // self.gridSize[0]
                srcY = row*rowFactor + 1
                dstY = (row+1)*rowFactor - 1
                for col in range(self.gridSize[1]):
                    colFactor = self.width // self.gridSize[1]
                    srcX = col*colFactor + 1
                    dstX = (col+1)*colFactor - 1
                    src = (srcX, srcY)
                    dst = (dstX, dstY)
                    self.gridList[row][col]["topLeft"] = src
                    self.gridList[row][col]["bottomRight"] = dst
                    self.gridList[row][col]["state"] = False
                    self.fillRectangle(row, col)

            cv2.setMouseCallback(self.windowName, self.clickCallback)
            cv2.imshow(self.windowName, self.gridImage)


        def getMask(self) -> np.uint64:
            mask = np.uint64(0)
            for row in range(self.gridSize[0]):
                for col in range(self.gridSize[1]):
                    if self.gridList[row][col]["state"]:
                        pos = row*self.gridSize[1] + col
                        mask = np.bitwise_or(mask, np.uint64(1) << np.uint64(pos))

            return mask


        def setMask(self, _mask: np.uint64):
            mask = np.uint64(_mask)
            for row in range(self.gridSize[0]):
                for col in range(self.gridSize[1]):
                    pos = row*self.gridSize[1] + col
                    if np.bitwise_and(mask, np.uint64(1) << np.uint64(pos)):
                        self.gridList[row][col]["state"] = True
                    else:
                        self.gridList[row][col]["state"] = False

                    self.fillRectangle(row, col)


        def isChanged(self):
            changed = self.changed
            self.changed = False
            return changed


        def destroyWindow(self):
            cv2.destroyWindow(self.windowName)


    censusMaskHandler = None
    newConfig = False
    config = None
    trSigma = list()
    trConfidence = list()
    trLrCheck = list()
    trFractionalBits = list()
    trLineqAlpha = list()
    trLineqBeta = list()
    trLineqThreshold = list()
    trCostAggregationP1 = list()
    trCostAggregationP2 = list()
    trTemporalAlpha = list()
    trTemporalDelta = list()
    trThresholdMinRange = list()
    trThresholdMaxRange = list()
    trSpeckleRange = list()
    trSpatialAlpha = list()
    trSpatialDelta = list()
    trSpatialHoleFilling = list()
    trSpatialNumIterations = list()
    trDecimationFactor = list()
    trDisparityShift = list()


    def trackbarSigma(value):
        StereoConfigHandler.config.postProcessing.bilateralSigmaValue = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trSigma:
            tr.set(value)


    def trackbarConfidence(value):
        StereoConfigHandler.config.costMatching.confidenceThreshold = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trConfidence:
            tr.set(value)


    def trackbarLrCheckThreshold(value):
        StereoConfigHandler.config.algorithmControl.leftRightCheckThreshold = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trLrCheck:
            tr.set(value)


    def trackbarFractionalBits(value):
        StereoConfigHandler.config.algorithmControl.subpixelFractionalBits = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trFractionalBits:
            tr.set(value)


    def trackbarLineqAlpha(value):
        StereoConfigHandler.config.costMatching.linearEquationParameters.alpha = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trLineqAlpha:
            tr.set(value)


    def trackbarLineqBeta(value):
        StereoConfigHandler.config.costMatching.linearEquationParameters.beta = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trLineqBeta:
            tr.set(value)


    def trackbarLineqThreshold(value):
        StereoConfigHandler.config.costMatching.linearEquationParameters.threshold = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trLineqThreshold:
            tr.set(value)


    def trackbarCostAggregationP1(value):
        StereoConfigHandler.config.costAggregation.horizontalPenaltyCostP1 = value
        StereoConfigHandler.config.costAggregation.verticalPenaltyCostP1 = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trCostAggregationP1:
            tr.set(value)


    def trackbarCostAggregationP2(value):
        StereoConfigHandler.config.costAggregation.horizontalPenaltyCostP2 = value
        StereoConfigHandler.config.costAggregation.verticalPenaltyCostP2 = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trCostAggregationP2:
            tr.set(value)


    def trackbarTemporalFilterAlpha(value):
        StereoConfigHandler.config.postProcessing.temporalFilter.alpha = value / 100.
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trTemporalAlpha:
            tr.set(value)


    def trackbarTemporalFilterDelta(value):
        StereoConfigHandler.config.postProcessing.temporalFilter.delta = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trTemporalDelta:
            tr.set(value)


    def trackbarSpatialFilterAlpha(value):
        StereoConfigHandler.config.postProcessing.spatialFilter.alpha = value / 100.
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trSpatialAlpha:
            tr.set(value)


    def trackbarSpatialFilterDelta(value):
        StereoConfigHandler.config.postProcessing.spatialFilter.delta = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trSpatialDelta:
            tr.set(value)


    def trackbarSpatialFilterHoleFillingRadius(value):
        StereoConfigHandler.config.postProcessing.spatialFilter.holeFillingRadius = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trSpatialHoleFilling:
            tr.set(value)


    def trackbarSpatialFilterNumIterations(value):
        StereoConfigHandler.config.postProcessing.spatialFilter.numIterations = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trSpatialNumIterations:
            tr.set(value)


    def trackbarThresholdMinRange(value):
        StereoConfigHandler.config.postProcessing.thresholdFilter.minRange = value * 1000
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trThresholdMinRange:
            tr.set(value)


    def trackbarThresholdMaxRange(value):
        StereoConfigHandler.config.postProcessing.thresholdFilter.maxRange = value * 1000
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trThresholdMaxRange:
            tr.set(value)


    def trackbarSpeckleRange(value):
        StereoConfigHandler.config.postProcessing.speckleFilter.speckleRange = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trSpeckleRange:
            tr.set(value)


    def trackbarDecimationFactor(value):
        StereoConfigHandler.config.postProcessing.decimationFilter.decimationFactor = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trDecimationFactor:
            tr.set(value)


    def trackbarDisparityShift(value):
        StereoConfigHandler.config.algorithmControl.disparityShift = value
        StereoConfigHandler.newConfig = True
        for tr in StereoConfigHandler.trDisparityShift:
            tr.set(value)


    def handleKeypress(self, key, stereoDepthConfigInQueue):
        if key == ord('m'):
            StereoConfigHandler.newConfig = True
            medianSettings = [dai.MedianFilter.MEDIAN_OFF, dai.MedianFilter.KERNEL_3x3, dai.MedianFilter.KERNEL_5x5, dai.MedianFilter.KERNEL_7x7]
            currentMedian = StereoConfigHandler.config.postProcessing.median
            nextMedian = medianSettings[(medianSettings.index(currentMedian)+1) % len(medianSettings)]
            print(f"Changing median to {nextMedian.name} from {currentMedian.name}")
            StereoConfigHandler.config.postProcessing.median = nextMedian
        if key == ord('w'):
            StereoConfigHandler.newConfig = True
            StereoConfigHandler.config.postProcessing.spatialFilter.enable = not StereoConfigHandler.config.postProcessing.spatialFilter.enable
            state = "on" if StereoConfigHandler.config.postProcessing.spatialFilter.enable else "off"
            print(f"Spatial filter {state}")
        if key == ord('t'):
            StereoConfigHandler.newConfig = True
            StereoConfigHandler.config.postProcessing.temporalFilter.enable = not StereoConfigHandler.config.postProcessing.temporalFilter.enable
            state = "on" if StereoConfigHandler.config.postProcessing.temporalFilter.enable else "off"
            print(f"Temporal filter {state}")
        if key == ord('s'):
            StereoConfigHandler.newConfig = True
            StereoConfigHandler.config.postProcessing.speckleFilter.enable = not StereoConfigHandler.config.postProcessing.speckleFilter.enable
            state = "on" if StereoConfigHandler.config.postProcessing.speckleFilter.enable else "off"
            print(f"Speckle filter {state}")
        if key == ord('r'):
            StereoConfigHandler.newConfig = True
            temporalSettings = [dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.PERSISTENCY_OFF,
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_8_OUT_OF_8,
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3,
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4,
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_OUT_OF_8,
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_2,
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_5,
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_1_IN_LAST_8,
            dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.PERSISTENCY_INDEFINITELY,
            ]
            currentTemporal = StereoConfigHandler.config.postProcessing.temporalFilter.persistencyMode
            nextTemporal = temporalSettings[(temporalSettings.index(currentTemporal)+1) % len(temporalSettings)]
            print(f"Changing temporal persistency to {nextTemporal.name} from {currentTemporal.name}")
            StereoConfigHandler.config.postProcessing.temporalFilter.persistencyMode = nextTemporal
        if key == ord('n'):
            StereoConfigHandler.newConfig = True
            decimationSettings = [dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.PIXEL_SKIPPING,
            dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN,
            dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEAN,
            ]
            currentDecimation = StereoConfigHandler.config.postProcessing.decimationFilter.decimationMode
            nextDecimation = decimationSettings[(decimationSettings.index(currentDecimation)+1) % len(decimationSettings)]
            print(f"Changing decimation mode to {nextDecimation.name} from {currentDecimation.name}")
            StereoConfigHandler.config.postProcessing.decimationFilter.decimationMode = nextDecimation
        if key == ord('a'):
            StereoConfigHandler.newConfig = True
            alignmentSettings = [dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_RIGHT,
            dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT,
            dai.StereoDepthConfig.AlgorithmControl.DepthAlign.CENTER,
            ]
            currentAlignment = StereoConfigHandler.config.algorithmControl.depthAlign
            nextAlignment = alignmentSettings[(alignmentSettings.index(currentAlignment)+1) % len(alignmentSettings)]
            print(f"Changing alignment mode to {nextAlignment.name} from {currentAlignment.name}")
            StereoConfigHandler.config.algorithmControl.depthAlign = nextAlignment
        elif key == ord('c'):
            StereoConfigHandler.newConfig = True
            censusSettings = [dai.StereoDepthConfig.CensusTransform.KernelSize.AUTO, dai.StereoDepthConfig.CensusTransform.KernelSize.KERNEL_5x5, dai.StereoDepthConfig.CensusTransform.KernelSize.KERNEL_7x7, dai.StereoDepthConfig.CensusTransform.KernelSize.KERNEL_7x9]
            currentCensus = StereoConfigHandler.config.censusTransform.kernelSize
            nextCensus = censusSettings[(censusSettings.index(currentCensus)+1) % len(censusSettings)]
            if nextCensus != dai.StereoDepthConfig.CensusTransform.KernelSize.AUTO:
                censusGridSize = [(5,5), (7,7), (7,9)]
                censusDefaultMask = [np.uint64(0XA82415), np.uint64(0XAA02A8154055), np.uint64(0X2AA00AA805540155)]
                censusGrid = censusGridSize[nextCensus]
                censusMask = censusDefaultMask[nextCensus]
                StereoConfigHandler.censusMaskHandler = StereoConfigHandler.CensusMaskHandler("Census mask", censusGrid)
                StereoConfigHandler.censusMaskHandler.setMask(censusMask)
            else:
                print("Census mask config is not available in AUTO census kernel mode. Change using the 'c' key")
                StereoConfigHandler.config.censusTransform.kernelMask = 0
                StereoConfigHandler.censusMaskHandler.destroyWindow()
            print(f"Changing census transform to {nextCensus.name} from {currentCensus.name}")
            StereoConfigHandler.config.censusTransform.kernelSize = nextCensus
        elif key == ord('d'):
            StereoConfigHandler.newConfig = True
            dispRangeSettings = [dai.StereoDepthConfig.CostMatching.DisparityWidth.DISPARITY_64, dai.StereoDepthConfig.CostMatching.DisparityWidth.DISPARITY_96]
            currentDispRange = StereoConfigHandler.config.costMatching.disparityWidth
            nextDispRange = dispRangeSettings[(dispRangeSettings.index(currentDispRange)+1) % len(dispRangeSettings)]
            print(f"Changing disparity range to {nextDispRange.name} from {currentDispRange.name}")
            StereoConfigHandler.config.costMatching.disparityWidth = nextDispRange
        elif key == ord('f'):
            StereoConfigHandler.newConfig = True
            StereoConfigHandler.config.costMatching.enableCompanding = not StereoConfigHandler.config.costMatching.enableCompanding
            state = "on" if StereoConfigHandler.config.costMatching.enableCompanding else "off"
            print(f"Companding {state}")
        elif key == ord('v'):
            StereoConfigHandler.newConfig = True
            StereoConfigHandler.config.censusTransform.enableMeanMode = not StereoConfigHandler.config.censusTransform.enableMeanMode
            state = "on" if StereoConfigHandler.config.censusTransform.enableMeanMode else "off"
            print(f"Census transform mean mode {state}")
        elif key == ord('1'):
            StereoConfigHandler.newConfig = True
            StereoConfigHandler.config.algorithmControl.enableLeftRightCheck = not StereoConfigHandler.config.algorithmControl.enableLeftRightCheck
            state = "on" if StereoConfigHandler.config.algorithmControl.enableLeftRightCheck else "off"
            print(f"LR-check {state}")
        elif key == ord('2'):
            StereoConfigHandler.newConfig = True
            StereoConfigHandler.config.algorithmControl.enableSubpixel = not StereoConfigHandler.config.algorithmControl.enableSubpixel
            state = "on" if StereoConfigHandler.config.algorithmControl.enableSubpixel else "off"
            print(f"Subpixel {state}")
        elif key == ord('3'):
            StereoConfigHandler.newConfig = True
            StereoConfigHandler.config.algorithmControl.enableExtended = not StereoConfigHandler.config.algorithmControl.enableExtended
            state = "on" if StereoConfigHandler.config.algorithmControl.enableExtended else "off"
            print(f"Extended {state}")

        censusMaskChanged = False
        if StereoConfigHandler.censusMaskHandler is not None:
            censusMaskChanged = StereoConfigHandler.censusMaskHandler.isChanged()
        if censusMaskChanged:
            StereoConfigHandler.config.censusTransform.kernelMask = StereoConfigHandler.censusMaskHandler.getMask()
            StereoConfigHandler.newConfig = True

        StereoConfigHandler.sendConfig(stereoDepthConfigInQueue)


    def sendConfig(stereoDepthConfigInQueue):
        if StereoConfigHandler.newConfig:
            StereoConfigHandler.newConfig = False
            configMessage = dai.StereoDepthConfig()
            configMessage.set(StereoConfigHandler.config)
            stereoDepthConfigInQueue.send(configMessage)


    def updateDefaultConfig(config):
        StereoConfigHandler.config = config


    def registerWindow(stream):
        cv2.namedWindow(stream, cv2.WINDOW_NORMAL)

        StereoConfigHandler.trConfidence.append(StereoConfigHandler.Trackbar('Disparity confidence', stream, 0, 255, StereoConfigHandler.config.costMatching.confidenceThreshold, StereoConfigHandler.trackbarConfidence))
        StereoConfigHandler.trSigma.append(StereoConfigHandler.Trackbar('Bilateral sigma', stream, 0, 100, StereoConfigHandler.config.postProcessing.bilateralSigmaValue, StereoConfigHandler.trackbarSigma))
        StereoConfigHandler.trLrCheck.append(StereoConfigHandler.Trackbar('LR-check threshold', stream, 0, 16, StereoConfigHandler.config.algorithmControl.leftRightCheckThreshold, StereoConfigHandler.trackbarLrCheckThreshold))
        StereoConfigHandler.trFractionalBits.append(StereoConfigHandler.Trackbar('Subpixel fractional bits', stream, 3, 5, StereoConfigHandler.config.algorithmControl.subpixelFractionalBits, StereoConfigHandler.trackbarFractionalBits))
        StereoConfigHandler.trDisparityShift.append(StereoConfigHandler.Trackbar('Disparity shift', stream, 0, 100, StereoConfigHandler.config.algorithmControl.disparityShift, StereoConfigHandler.trackbarDisparityShift))
        StereoConfigHandler.trLineqAlpha.append(StereoConfigHandler.Trackbar('Linear equation alpha', stream, 0, 15, StereoConfigHandler.config.costMatching.linearEquationParameters.alpha, StereoConfigHandler.trackbarLineqAlpha))
        StereoConfigHandler.trLineqBeta.append(StereoConfigHandler.Trackbar('Linear equation beta', stream, 0, 15, StereoConfigHandler.config.costMatching.linearEquationParameters.beta, StereoConfigHandler.trackbarLineqBeta))
        StereoConfigHandler.trLineqThreshold.append(StereoConfigHandler.Trackbar('Linear equation threshold', stream, 0, 255, StereoConfigHandler.config.costMatching.linearEquationParameters.threshold, StereoConfigHandler.trackbarLineqThreshold))
        StereoConfigHandler.trCostAggregationP1.append(StereoConfigHandler.Trackbar('Cost aggregation P1', stream, 0, 500, StereoConfigHandler.config.costAggregation.horizontalPenaltyCostP1, StereoConfigHandler.trackbarCostAggregationP1))
        StereoConfigHandler.trCostAggregationP2.append(StereoConfigHandler.Trackbar('Cost aggregation P2', stream, 0, 500, StereoConfigHandler.config.costAggregation.horizontalPenaltyCostP2, StereoConfigHandler.trackbarCostAggregationP2))
        StereoConfigHandler.trTemporalAlpha.append(StereoConfigHandler.Trackbar('Temporal filter alpha', stream, 0, 100, int(StereoConfigHandler.config.postProcessing.temporalFilter.alpha*100), StereoConfigHandler.trackbarTemporalFilterAlpha))
        StereoConfigHandler.trTemporalDelta.append(StereoConfigHandler.Trackbar('Temporal filter delta', stream, 0, 100, StereoConfigHandler.config.postProcessing.temporalFilter.delta, StereoConfigHandler.trackbarTemporalFilterDelta))
        StereoConfigHandler.trSpatialAlpha.append(StereoConfigHandler.Trackbar('Spatial filter alpha', stream, 0, 100, int(StereoConfigHandler.config.postProcessing.spatialFilter.alpha*100), StereoConfigHandler.trackbarSpatialFilterAlpha))
        StereoConfigHandler.trSpatialDelta.append(StereoConfigHandler.Trackbar('Spatial filter delta', stream, 0, 100, StereoConfigHandler.config.postProcessing.spatialFilter.delta, StereoConfigHandler.trackbarSpatialFilterDelta))
        StereoConfigHandler.trSpatialHoleFilling.append(StereoConfigHandler.Trackbar('Spatial filter hole filling radius', stream, 0, 16, StereoConfigHandler.config.postProcessing.spatialFilter.holeFillingRadius, StereoConfigHandler.trackbarSpatialFilterHoleFillingRadius))
        StereoConfigHandler.trSpatialNumIterations.append(StereoConfigHandler.Trackbar('Spatial filter number of iterations', stream, 0, 4, StereoConfigHandler.config.postProcessing.spatialFilter.numIterations, StereoConfigHandler.trackbarSpatialFilterNumIterations))
        StereoConfigHandler.trThresholdMinRange.append(StereoConfigHandler.Trackbar('Threshold filter min range', stream, 0, 65, StereoConfigHandler.config.postProcessing.thresholdFilter.minRange, StereoConfigHandler.trackbarThresholdMinRange))
        StereoConfigHandler.trThresholdMaxRange.append(StereoConfigHandler.Trackbar('Threshold filter max range', stream, 0, 65, StereoConfigHandler.config.postProcessing.thresholdFilter.maxRange, StereoConfigHandler.trackbarThresholdMaxRange))
        StereoConfigHandler.trSpeckleRange.append(StereoConfigHandler.Trackbar('Speckle filter range', stream, 0, 240, StereoConfigHandler.config.postProcessing.speckleFilter.speckleRange, StereoConfigHandler.trackbarSpeckleRange))
        StereoConfigHandler.trDecimationFactor.append(StereoConfigHandler.Trackbar('Decimation factor', stream, 1, 4, StereoConfigHandler.config.postProcessing.decimationFilter.decimationFactor, StereoConfigHandler.trackbarDecimationFactor))


    def __init__(self, config):
        print("Control median filter using the 'm' key.")
        print("Control census transform kernel size using the 'c' key.")
        print("Control disparity search range using the 'd' key.")
        print("Control disparity companding using the 'f' key.")
        print("Control census transform mean mode using the 'v' key.")
        print("Control depth alignment using the 'a' key.")
        print("Control decimation algorithm using the 'a' key.")
        print("Control temporal persistency mode using the 'r' key.")
        print("Control spatial filter using the 'w' key.")
        print("Control temporal filter using the 't' key.")
        print("Control speckle filter using the 's' key.")
        print("Control left-right check mode using the '1' key.")
        print("Control subpixel mode using the '2' key.")
        print("Control extended mode using the '3' key.")

        StereoConfigHandler.config = config

        if StereoConfigHandler.config.censusTransform.kernelSize != dai.StereoDepthConfig.CensusTransform.KernelSize.AUTO:
            censusMask = StereoConfigHandler.config.censusTransform.kernelMask
            censusGridSize = [(5,5), (7,7), (7,9)]
            censusGrid = censusGridSize[StereoConfigHandler.config.censusTransform.kernelSize]
            if StereoConfigHandler.config.censusTransform.kernelMask == 0:
                censusDefaultMask = [np.uint64(0xA82415), np.uint64(0xAA02A8154055), np.uint64(0x2AA00AA805540155)]
                censusMask = censusDefaultMask[StereoConfigHandler.config.censusTransform.kernelSize]
            StereoConfigHandler.censusMaskHandler = StereoConfigHandler.CensusMaskHandler("Census mask", censusGrid)
            StereoConfigHandler.censusMaskHandler.setMask(censusMask)
        else:
            print("Census mask config is not available in AUTO Census kernel mode. Change using the 'c' key")


#OAK-D camera stream and lidar mode
class OAK_STREAM(StereoConfigHandler):
    def __init__(self, config = False):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        # source and outputs
        # stereo camera
        self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
        self.monoRight = self.pipeline.create(dai.node.MonoCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        self.xout = self.pipeline.create(dai.node.XLinkOut)
        self.xinStereoDepthConfig =self.pipeline.create(dai.node.XLinkIn)
        #RGB camera
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutVideo = self.pipeline.create(dai.node.XLinkOut)
        #self.xoutPreview = self.pipeline.create(dai.node.XLinkOut)

        #properties
        #stereo camera
        self.xout.setStreamName("disparity")
        self.xinStereoDepthConfig.setStreamName("stereoDepthConfig")
        self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        #RGB camera
        self.xoutVideo.setStreamName("video")
        #self.xoutPreview.setStreamName("preview")
        #self.camRgb.setPreviewSize(640, 400)
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        self.camRgb.setVideoSize(640,400)
        self.camRgb.setInterleaved(True)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # stereo config
        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setLeftRightCheck(LR_CHECK)
        self.stereo.setExtendedDisparity(EXTENDE_DISPARITY)
        self.stereo.setSubpixel(SUBPIXEL)

        # RGB config
        self.xoutVideo.input.setBlocking(False)
        self.xoutVideo.input.setQueueSize(1)

        # Linking
        #stereo camera
        self.monoLeft.out.link(self.stereo.left)
        self.monoRight.out.link(self.stereo.right)
        self.stereo.disparity.link(self.xout.input)
        self.xinStereoDepthConfig.out.link(self.stereo.inputConfig)
        #RGB camera
        self.camRgb.video.link(self.xoutVideo.input)
        #self.camRgb.preview.link(self.xoutPreview.input)

        self.config = config
        if config:
            super().__init__(self.stereo.initialConfig.get())
            StereoConfigHandler.registerWindow('Stereo dash board')


    def DepthMap2lidar(self, frame, density = 1000, treshold = 50, danger_value = 200):
        def get_indice_dense(data,depth = 200):
            return np.asmatrix(np.where(data > depth)).size
        if get_indice_dense(frame, danger_value) > density:
            print("danger")
        elif get_indice_dense(frame, danger_value - 1*treshold) > density:
            print("warning")
        elif get_indice_dense(frame, danger_value - 2*treshold) > density:
            print("object detected")
        else:
            print("ok")


    def DaiRGB(self, main = True):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
            preview = device.getOutputQueue('preview')

            if main:
                while True:
                    videoFrame = video.get()
                    previewFrame = preview.get()

                    # Get BGR frame from NV12 encoded video frame to show with opencv
                    cv2.imshow("video", videoFrame.getCvFrame())
                    # Show 'preview' frame as is (already in correct format, no copy is made)
                    cv2.imshow("preview", previewFrame.getFrame())
                    print('RAM Used (GB):', virtual_memory()[3]/1000000000)
                    if cv2.waitKey(1) == ord('q'):
                        break
            else:
                return video.get().getCvFrame()


    def DaiStereo(self, main = True):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:
            qin = device.getInputQueue("stereoDepthConfig")
            # Output queue will be used to get the disparity frames from the outputs defined above
            qout = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

            if main:
                while True:
                    inDisparity = qout.get()  # blocking call, will wait until a new data has arrived
                    frame = inDisparity.getFrame()
                    # Normalization for better visualization
                    frame = (frame * (255 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
                    self.DepthMap2lidar(frame)
                    # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                    cv2.imshow("disparity_color", frame)
                    #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    self.handleKeypress(key, qin)
            else:
                inDisparity = qout.get()  # blocking call, will wait until a new data has arrived
                frame = inDisparity.getFrame()
                # Normalization for better visualization
                self.stream = (frame * (255 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)


#Aruco tag and red cherries detection with an RGB camera
class RGB_Computation(OAK_STREAM):
    def __init__(self, calib_path, camera_matrix, camera_distortion, webcam = True):

        self.calib_path = calib_path
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion

        #--- 180 deg rotation matrix around the x axis
        self.R_flip  = np.zeros((3,3), dtype=np.float32)
        self.R_flip[0,0] = 1.0
        self.R_flip[1,1] =-1.0
        self.R_flip[2,2] =-1.0

        #--- Define the aruco dictionary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        self.parameters =  aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.cornerRefinementWinSize = 5  


        self.webcam = webcam
        if webcam:
            #--- Capture the videocamera (this may also be a video or a picture)
            self.cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            #-- Set the camera size as the one it was calibrated with (w1280,h960)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        else:
            super().__init__()

        #-- Font for the text in the image
        self.font = cv2.FONT_HERSHEY_PLAIN

        self.comparisonList = {}


    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6


    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self,R):
        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x_rotation = math.atan2(R[2, 1], R[2, 2])
            y_rotation = math.atan2(-R[2, 0], sy)
            z_rotation = math.atan2(R[1, 0], R[0, 0])
        else:
            x_rotation = math.atan2(-R[1, 2], R[1, 1])
            y_rotation = math.atan2(-R[2, 0], sy)
            z_rotation = 0

        return np.array([x_rotation, y_rotation, z_rotation])


    def find_marker(self):
            #-- Read the camera frame
            ret, self.frame = self.cap.read()

            #-- Convert in gray scale
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) #OpenCV stores color images in Blue, Green, Red

            #-- Find all the aruco markers in the image
            self.corners, self.ids, rejected = self.detector.detectMarkers(image=gray, dictionary=self.aruco_dict, parameters=self.parameters)


    def find_marker_position(self, id, size):
        ret = aruco.estimatePoseSingleMarkers(self.corners[id], size, camera_matrix, camera_distortion)
        #-- Unpack the output, get only the first
        self.rvec, self.tvec = ret[0][0,0,:], ret[1][0,0,:]
    

    def find_marker_rotation(self):
        #-- Obtain the rotation matrix tag->camera
        R_ct    = np.matrix(cv2.Rodrigues(self.rvec)[0])
        R_tc    = R_ct.T

        #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
        self.roll_marker, self.pitch_marker, self.yaw_marker = self.rotationMatrixToEulerAngles(self.R_flip*R_tc)

        #-- Now get Position and attitude f the camera respect to the marker
        self.pos_camera = -R_tc*np.matrix(self.tvec).T

        #-- Get the attitude of the camera respect to the frame
        self.roll_camera, self.pitch_camera, self.yaw_camera = self.rotationMatrixToEulerAngles(self.R_flip*R_tc)

    
    def pushComparison(self, id, ID_tvec, value = False):
        if id in self.comparisonList.keys():
            if value:
                if not np.array_equal(self.comparisonList[id], ID_tvec):
                    self.comparisonList[id] = ID_tvec
            else:    
                return
        else:
            print(id)
            self.comparisonList[id] = ID_tvec


    def distanceBetweenMarker(self, ID1, ID2):
        if ID1 in self.comparisonList.keys() and ID2 in self.comparisonList.keys():
            x1 = self.comparisonList[ID1][0]
            x2 = self.comparisonList[ID2][0]
            y1 = self.comparisonList[ID1][1]
            y2 = self.comparisonList[ID2][1]
            return np.hypot(x1-x2, y1-y2)
        else:
            return


    def display_marker_coord(self, id, rotation = False, camera_pos = False):
        #-- Draw the detected marker and put a reference frame over it
        #-- aruco.drawDetectedMarkers(self.frame, self.corners)
        cv2.drawFrameAxes(self.frame, camera_matrix, camera_distortion, self.rvec, self.tvec, 10)#-- Print the tag position in camera frame
        str_position = f"MARKER {self.ids[id]} Position x=%4.2f  y=%4.2f  z=%4.2f"%(self.tvec[0], self.tvec[1], self.tvec[2])
        cv2.putText(self.frame, str_position, (0, (BASE_POSITION)+(id*50)), self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        if rotation:
            #-- Print the marker's attitude respect to camera frame
            str_attitude = f"MARKER {self.ids[id]} Attitude r=%4.2f  p=%4.2f  y=%4.2f"%(math.degrees(self.roll_marker),math.degrees(self.pitch_marker),
                                math.degrees(self.yaw_marker))
            cv2.putText(self.frame, str_attitude, (0, (BASE_POSITION + 20)+(id*50)), self.font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if camera_pos:        
            str_position = "CAMERA Position x=%4.2f  y=%4.2f  z=%4.2f"%(self.pos_camera[0], self.pos_camera[1], self.pos_camera[2])
            cv2.putText(self.frame, str_position, (0, (BASE_POSITION + 100)+(id*250)), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            #-- Get the attitude of the camera respect to the frame
            str_attitude = "CAMERA Attitude r=%4.2f  p=%4.2f  y=%4.2f"%(math.degrees(self.roll_camera),math.degrees(self.pitch_camera),
                                math.degrees(self.yaw_camera))
            cv2.putText(self.frame, str_attitude, (0, (BASE_POSITION + 150)+(id*250)), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        

    def detect_cherries(self, img):
        # Captures the live stream frame-by-frame


        frame = cv2.bitwise_not(img)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red = np.array([90-10, 200, 150]) 
        upper_red= np.array([90+10, 255, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(hsv, hsv, mask=mask)
        mask = cv2.blur(mask, (3,3))
        
        contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        
        no_iteration = True
        cherries_counter = 0
        comparison_list = [0]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                for zone in comparison_list:
                    if area < zone + 300 and area > zone - 300:
                        no_iteration = False
                        cherries_counter += 1
                        comparison_list[comparison_list.index(zone)] = (zone + comparison_list[comparison_list.index(zone)])/2
                if no_iteration:
                    comparison_list.append(area)
        
        if cherries_counter > 6:
            print("cherries")
        else:
            print("0")


        cv2.drawContours(image=res, contours=contours, contourIdx=-1, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        
        # points are in x,y coordinates
        #cv2.imshow('frame', frame)
        #cv2.imshow('mask', mask)    
        cv2.imshow('res', res)


    def main_Webcam(self):
        while 1:
            self.find_marker()
            #interesting marker detection and memorisation
            if len(self.corners):
                for i in range(0, len(self.ids)):
                    if (self.ids[i] >= 20 and self.ids[i] <= 23):
                        self.find_marker_position(i, MARKER_SIZE_MAT)
                        self.display_marker_coord(i)
                        self.pushComparison(self.ids[i][0], self.tvec)
                    if self.ids[i] == 2 or self.ids[i] == 7:
                        self.find_marker_position(i, MARKER_SIZE_ROBOT)
                        self.display_marker_coord(i)
                        self.pushComparison(self.ids[i][0], self.tvec, value=True)
            
            print("distance between Tag 2 et Tag 7 : ",self.distanceBetweenMarker(2, 7))
            
            #--- Display the frame
            cv2.imshow('frame', self.frame)
            #--- use 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break


    def main_Dai(self):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline, usb2Mode=True) as device:

            video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
            #preview = device.getOutputQueue('preview')
            qin = device.getInputQueue("stereoDepthConfig")
            # Output queue will be used to get the disparity frames from the outputs defined above
            qout = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
            while True:
                videoFrame = video.get()
                #previewFrame = preview.get()
                inDisparity = qout.get()  # blocking call, will wait until a new data has arrived
                frame = inDisparity.getFrame()
                # Normalization for better visualization
                frame = (frame * (255 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
                #self.DepthMap2lidar(frame)
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                cv2.imshow("disparity_color", frame)
                

                # Get BGR frame from NV12 encoded video frame to show with opencv
                #-- Read the camera frame
                self.frame = videoFrame.getCvFrame()
                # Show 'preview' frame as is (already in correct format, no copy is made)
                #previewFrame.getFrame()
                
                self.detect_cherries(self.frame)

                #-- Convert in gray scale
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) #OpenCV stores color images in Blue, Green, Red

                #-- Find all the aruco markers in the image
                self.corners, self.ids, rejected = aruco.detectMarkers(image=gray, dictionary=self.aruco_dict, parameters=self.parameters)
                
                #interesting marker detection and memorisation
                if len(self.corners):
                    for i in range(0, len(self.ids)):
                        if self.ids[i] == 47 or self.ids[i] == 13:
                            self.find_marker_position(i, MARKER_SIZE_ROBOT)
                            self.display_marker_coord(i)
                            self.pushComparison(self.ids[i][0], self.tvec, value=True)
                    #print("distance between Tag 2 et Tag 7 : ",self.distanceBetweenMarker(47, 13))
                
                #--- Display the frame
                cv2.imshow('frame', self.frame)
                
                print('RAM Used (GB):', virtual_memory()[3]/1000000000)

                #--- use 'q' to quit
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
                if self.config:
                    self.handleKeypress(key, qin)


#main loop
if __name__ == "__main__":
    try:
        cam = RGB_Computation(calib_path, camera_matrix, camera_distortion, webcam=False)
        cam.main_Dai()

    except KeyboardInterrupt:
        print("script manually interrupted")