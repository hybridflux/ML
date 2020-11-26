# -*- coding: utf-8 -*-
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("libraries")
sys.path.append("../libraries")
from helpers import *
from helpers_cntk import *
from PARAMETERS import *


####################################
# Create Pickle File
####################################
random.seed(0)
makeDirectory(rootDir)
makeDirectory(procDir)
amlLogger = getAmlLogger()
if amlLogger != []:
    amlLogger.log("amlrealworld.ImageClassificationUsingCntk.7_evalModel", "true")


imgDictEval  = dict()
subdirs = getDirectoriesInDirectory(imgEvalDir)

for subdir in subdirs:
    imgDictEval[subdir]=[]
    filenames = getFilesInDirectory(pathJoin(imgEvalDir, subdir), ".jpg")
    for filename in filenames:
         key = "/".join([subdir, filename]).lower()
         imgDictEval[subdir].append(filename)
            
           
writePickle(imgDictEvalPath,  imgDictEval)



####################################
# Evaluation COde
####################################
makeDirectory("outputs")
model = load_model(cntkRefinedModelPath)
mapPath = pathJoin(workingDir, "rundnn_map.txt")
node  = getModelNode(classifier)

# Load data
print("Loading data...")
dnnOutput   = readPickle(dnnOutputPath)

lutLabel2Id = readPickle(lutLabel2IdPath)
lutId2Label = readPickle(lutId2LabelPath)

imgDict = readPickle(imgDictEvalPath)

dataTest  = getImgLabelList(imgDict,  imgOrigDir, lutLabel2Id)
writeTable(cntkTestMapPath,  dataTest)
print("Running DNN for eval set..")
dnnOutputEval  = runCntkModelAllImages(model, readPickle(imgDictEvalPath),  imgEvalDir, mapPath, node, run_mbSize)

for label in list(dnnOutputEval.keys()):
    outEval  = dnnOutputEval[label]
    dnnOutput[label] = mergeDictionaries(dnnOutput, outEval)


#imgDict = readPickle(imgDictTestPath)
#print(readPickle(imgDictTestPath))

# Predicted labels and scores
scoresMatrix, imgFilenames, gtLabels = runClassifier(classifier, dnnOutput, imgDict,  lutLabel2Id, svmPath, svm_boL2Normalize)
predScores = [np.max(scores)    for scores in scoresMatrix]
predLabels = [np.argmax(scores) for scores in scoresMatrix]
writePickle(pathJoin(procDir, "EscoresMatrix.pickle"), scoresMatrix)
writePickle(pathJoin(procDir, "EpredLabels.pickle"),   predLabels)
writePickle(pathJoin(procDir, "EgtLabels.pickle"),   gtLabels)

print(scoresMatrix)

# Plot ROC curve
classes = [lutId2Label[i] for i in range(len(lutId2Label))]
fig = plt.figure(figsize=(14,6))
plt.subplot(121)
rocComputePlotCurves(gtLabels, scoresMatrix, classes)

# Plot confusion matrix
# Note: Let C be the confusion matrix. Then C_{i, j} is the number of observations known to be in group i but predicted to be in group j.
plt.subplot(122)
confMatrix = metrics.confusion_matrix(gtLabels, predLabels)

cmPlot(confMatrix, classes, normalize=False)
plt.show()
fig.savefig('outputs/ErocCurve_confMat.jpg', bbox_inches='tight', dpi = 200)

# Print accuracy to console
globalAcc, classAccs = cmPrintAccuracies(confMatrix, classes, gtLabels)
if amlLogger != []:
    amlLogger.log("clasifier", classifier)
    amlLogger.log("Global accuracy", 100 * globalAcc)
    amlLogger.log("Class-average accuracy", 100 * np.mean(classAccs))
    for className, acc in zip(classes,classAccs):
        amlLogger.log("Accuracy of class %s" % className, 100 * np.mean(classAccs))

# Visualize results
for counter, (gtLabel, imgFilename, predScore, predLabel) in enumerate(zip(gtLabels, imgFilenames, predScores, predLabels)):
    if counter > 5:
        break
    if predLabel == gtLabel:
        drawColor = (0, 255, 0)
    else:
        drawColor = (0, 0, 255)
    img = imread(pathJoin(imgEvalDir, lutId2Label[gtLabel], imgFilename))
    img = imresizeToSize(img, targetWidth = 800)
    cv2.putText(img, "{} with score {:2.2f}".format(lutId2Label[predLabel], predScore), (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, drawColor, 2)
    drawCircle(img, (50, 50), 40, drawColor, -1)
    imshow(img, maxDim = 800, waitDuration=100)
    #imwrite(imresizeMaxDim(img,800)[0], "outputs/result_img{}.jpg".format(counter))
print("DONE.")











print("DONE.")
