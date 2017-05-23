from SVGCropper import SVGCropper
from PIL import Image
import numpy as np
from FeatureExtractor import FeatureExtractor
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def runKWS(queryPath, imagePath, svgPath):
    query = Image.open(queryPath)
    svc = SVGCropper()
    fe = FeatureExtractor()
    result =[]
    #set threshold here
    threshold = 32  

    print("Cropping the segments")
    keywordsList = svc.cropWords(imagePath, svgPath)
    fq = fe.getFeatureVector(query)

    dists = []
    i = 0
    for keyword in keywordsList:
        f = fe.getFeatureVector(keyword)
        dist, path = fastdtw(f, fq, dist=euclidean)
        print "distance ",i,": ", dist
        dists.append(dist)
        i += 1

    dists = np.array(dists)
    matches = dists < threshold   
    
    return matches
        
def main():
    queryPath = "query.jpg"
    imagePath = "images/270.jpg"
    svgPath = "locations/270.svg"

    matches = runKWS(queryPath, imagePath,svgPath)
    print matches


main()    