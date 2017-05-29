from SVGCropper import SVGCropper
from PIL import Image
import numpy as np
from FeatureExtractor import FeatureExtractor
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def runKWS(query, imagePath, svgPath):
    svc = SVGCropper()
    fe = FeatureExtractor()
    result =[]
    #set threshold here
    threshold = 40 
    output = ""
    print("Cropping the segments")

    keywordsList = svc.cropWords(imagePath, svgPath)
    fq = fe.getFeatureVector(query)

    dists = []
    for keyword in keywordsList:
        f = fe.getFeatureVector(keyword[0])
        dist, path = fastdtw(f, fq, dist=euclidean)
        print "distance from ",keyword[1]," : ", dist
        dists.append(dist)
        output+=keyword[1]+","+str(dist)+" "
        
    return output

def runTest(keywordsFile, testFileList, imgLoc, svgLoc):
    file = open(keywordsFile)
    svc = SVGCropper()
    output = open("output.txt","w")
    for line in file:
        spltline = line.split(',')
        name = spltline[0]
        svgid = spltline[1]
        filename = spltline[1].split('-')[0]
        imagePath = imgLoc + filename + ".jpg"
        svgPath = svgLoc + filename + ".svg"
        query = svc.getQueryImage(imgLoc, svgLoc, line)
        
        testFiles = open(testFileList)
        for testFile in testFiles:
            print("\nSearching for query "+name+" in "+testFile[0:-1]+".jpg")
            print("-------------------------")
            output.write(name + " ")

            testFilePath = imgLoc+testFile[0:-1]+".jpg"
            testFileSVG = svgLoc+testFile[0:-1]+".svg"
            out = runKWS(query,testFilePath,testFileSVG)
            output.write(out)
            output.write("\n\n")
    output.close()                            



        
def main():
    imgLoc = "images/"
    svgLoc = "ground-truth/locations/"
    
    runTest("task/keywords.txt", "task/test.txt", imgLoc, svgLoc)


main()    