from SVGCropper import SVGCropper
from PIL import Image
import numpy as np
from FeatureExtractor import FeatureExtractor
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def runKWS(queryPath, imagePath, svgPath):
    query = Image.open(queryPath)
    print "query size: ", query.size
    svc = SVGCropper()
    fe = FeatureExtractor()

    keywordsList = svc.cropWords(imagePath, svgPath)
    fq = fe.getFeatureVector(query)

    for keyword in keywordsList:
        f = fe.getFeatureVector(keyword)
        dist, path = fastdtw(f, fq, dist=euclidean)
        print "distance: ", dist
        
def main():
    queryPath = "query.jpg"
    imagePath = "images/270.jpg"
    svgPath = "locations/270.svg"

    runKWS(queryPath, imagePath,svgPath)


main()    