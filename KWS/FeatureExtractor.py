from PIL import Image
import numpy as np

#def. col : window of width 1

class FeatureExtractor():
    def getFeatureVector(self, image):
        feat = []
        ncols, nrows = image.size
        image = self.binarizeImage(image)
        #each row correnponds to a col in it's transpose
        image = np.transpose(image)
        
        mid = nrows / 2
        for i in range(ncols-1):
            f = []
            col = image[i]
            UCn = col[0:mid]
            LCn = col[mid+1:nrows-1]

            col_next = image[i+1]
            UCn_next = col_next[0:mid]
            LCn_next = col_next[mid+1:nrows-1]

            #MODIFICATIONS NEEDED!!!!!
            
            #add the features to the feature vector 
            f.append(self.getBlackPixelFraction(col))
            f.append(self.getUpperBlackPixel(col))
            f.append(self.getLowerBlackPixel(col))
            f.append(self.getTrasitions(col))
            #f.append(self.getBlackPixelFraction(UCn))
            #f.append(self.getBlackPixelFraction(LCn))
            f.append(self.getBlackPixelFractionUandL(UCn,LCn))
            #f.append(self.getGradient(UCn,UCn_next))
            #f.append(self.getGradient(LCn,LCn_next))    
    
            #normalize f
            
            norm = np.linalg.norm(f, ord=2)
            if norm==0:
                continue
            else:
                f = [el/float(norm) for el in f]
            
            feat.append(f)

        return feat

    def binarizeImage(self, image):
        image = np.array(image)
        low_ind = image < 50
        high_ind = image >= 50
        image[low_ind] = 1 #black pixels
        image[high_ind] = 0 #white pixels

        return image
    
    def getTrasitions(self, col):
        tranCount = 0
        for i in range(len(col)-1):
            tranCount += (col[i] ^ col[i+1])

        return tranCount/10 

    def getBlackPixelFraction(self, col):
        return sum(col)/float(col.size)
    
    def getBlackPixelCount(self, col):
        return 

    def getBlackPixelFractionUandL(self, U, L):
        if sum(L) == 0:
            return 0
        else:
            
            fu = sum(U)/float(U.size) 
            fl = sum(L)/float(L.size)
            return fu/fl 



    def getGradient(self, col1, col2):
        return np.sum(np.abs(np.subtract(col1,col2)))

    def getUpperBlackPixel(self, col):
        pos = 1
        for i in range(col.size):
            if col[i]==1:
                pos = i/float(col.size)
                break
        return  pos

    def getLowerBlackPixel(self, col):
        pos = 1
        for i in range(col.size-1,0,-1):
            if col[i]==1:
                pos = i/float(col.size)
                break
        return  pos  
