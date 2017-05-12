from svgpathtools import svg2paths
from svgpathtools import parse_path
from PIL import Image, ImageDraw
import numpy as np
import os

class SVGCropper:

    def cropWords(self,imgPath, svgPath, location):
    
        threshold = 170

        #extract paths and store them in section_svg_strings
        paths, attributes = svg2paths(svgPath)
        section_svg_strings = []
        for k in range(len(attributes)):
            section_svg_strings.append(attributes[k]['d'])
        
        #load the image for processing
        img = Image.open(imgPath)
        img = np.asarray(img)
        
        #for each section do the cropping
        for i in range(len(section_svg_strings)):
            path = parse_path(section_svg_strings[i])
            #path = parse_path(p)
            polygon = self.getPolygon(path) 
            
            maskImg = Image.new('L', (img.shape[1], img.shape[0]), 0)
            maskImgBound = Image.new('L', (img.shape[1], img.shape[0]), 255)
            ImageDraw.Draw(maskImg).polygon(polygon, outline=1, fill=1)
            ImageDraw.Draw(maskImgBound).polygon(polygon, outline=0, fill=0)
            
            mask = np.array(maskImg)
            maskBound = np.array(maskImgBound)

            final = np.multiply(img, mask)
            final = np.add(final,maskBound)

            #binarize the final image
            low_value_indices = final < threshold
            high_value_indices = final >= threshold
            final[low_value_indices] = 0
            final[high_value_indices] = 255

            newImg = Image.fromarray(final)
            area = self.getBoundingBox(polygon)
            newImg = newImg.crop(area)
            if not os.path.exists(location):
                os.makedirs(location)

            filename = location + '/' + str(i) + '.jpg'

            newImg.save(filename)
            print "Word ", i, " Cropped Successfuly"


    #Converts SVG Path to Polygon
    def getPolygon(self, path):
        ptsCount = len(path)
        polygon = [(int(path[0][0].real), int(path[0][0].imag)), (int(path[0][1].real), int(path[0][1].imag))]
        for i in range(1,ptsCount):
            polygon.append((path[i][1].real, path[i][1].imag))
        
        return polygon
    

    #Gets Bouding Box of a Polygon
    def getBoundingBox(self, polygon):
        #Area format : (x_left, y_left, x_right, y_right)
        min_h = polygon[0][0]
        min_w = polygon[0][1]
        max_h = polygon[0][0]
        max_w = polygon[0][1]
        for i in range(1,len(polygon)):
            if polygon[i][0] < min_h:
                min_h = polygon[i][0]
            elif polygon[i][0] > max_h:
                max_h = polygon[i][0]

            if polygon[i][1] < min_w:
                min_w = polygon[i][1]
            elif polygon[i][1] > max_w:
                max_w = polygon[i][1]  
        
        return (int(min_h), int(min_w), int(max_h), int(max_w))    


  