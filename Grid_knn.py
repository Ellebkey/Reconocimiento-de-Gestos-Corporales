#!/usr/bin/env python

##########################################################################
#Grid_knn.py:   Programa de reconocimiento de posturas corporales        #
#               usando un cuadriula de ocupacion y knn para su analisis  #
##########################################################################

__author__  = 'Joel Barranco'
__email__   = 'contacto@joelbarranco.com'
__version__ = '1.0.1'
__license__ = 'MIT License'


'''Comienza Programa'''

import freenect
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

#Clase para reconocimiento, todo lo relacionado con freenect y opencv
class Recognition:
        
    THRESHOLD = 100
    CURRENT_DEPTH = 800
    COLS = 0    
    ROWS = 0
    SIZE = 0
    POINTS = []
    dev = 0

    def __init__(self, size):
        self.SIZE = size
        self.COLS = 640/size
        self.ROWS = 480/size

    #Obtener Imagen de profundidad y aplicar la mascara a la profundidad establecida
    def getDepth(self):

        depth,_ = freenect.sync_get_depth()

        depth = cv2.GaussianBlur(depth,(3,3),0)
        
        depth = 255 * np.logical_and(depth >= self.CURRENT_DEPTH - self.THRESHOLD,
                                     depth <= self.CURRENT_DEPTH + self.THRESHOLD)
        depth = depth.astype(np.uint8)

        return depth

    #Obtener imagen RGB
    def getVideo(self):
        image,_ = freenect.sync_get_video()
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        return image

    #Mostrar las imagenes
    def showFrame(self,window_name , frame):
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, frame)

    #Dibujar la cuadricula
    def draw_grid(self,frame):
        y,x = frame.shape
        m = x/self.COLS
        n = y/self.ROWS

        for i in range(self.COLS+1):
            for j in range(self.ROWS+1):
                coords = m*i,n*j
                cv2.circle(frame,coords,2,[255,0,255],2)
                if not (coords in self.POINTS):
                    self.POINTS.append((m*i,n*j))

    #Calcula Area
    def getArea(self,frame):
        area = 0
        
        contours, jerarquia = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0: 
            cnt = contours[0]
            moments = cv2.moments(cnt)
            area = moments['m00']
        return area

    #Calcula porcentaje ocupado en la cuadricula
    def percent(self,part,whole):
        per = (part * 100)/whole
        return per

    #Crear cuadricula
    def createGrid(self,gridOccupation):
        img = self.getDepth()
        x = self.SIZE-2

        for i in range(self.COLS):
            for j in range(self.ROWS):
                #Creamos un ROI del tamanio de la cuadricula que recorra toda la imagen
                ROI = img[j*self.SIZE:j*self.SIZE+self.SIZE,i*self.SIZE:i*self.SIZE+self.SIZE]
                #Calculamos el area en el ROI
                area = self.getArea(ROI)
                #Contamos el numero de pixeles que se ocupa en el grid
                n = cv2.countNonZero(ROI)
                if (area > 0) :
                    #calculamos el porcentaje de area ocupada y lo guardamos
                    p = self.percent(n,x*x)
                    gridOccupation[j][i] = p
        return gridOccupation
                    

    def getContour(self):

        frame = self.getDepth()
        frame2 = self.getVideo()

        contours, jerarquia = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for n, cnt in enumerate(contours):

            hull = cv2.convexHull(cnt)

            foo = cv2.convexHull(cnt, returnPoints = False)
            cv2.drawContours(frame2, contours, n, (0, 35, 245))
            if len(cnt) > 3 and len(foo) > 2:
                defectos = cv2.convexityDefects(cnt, foo)
                if defectos is not None:
                    defectos = defectos.reshape(-1, 4)
                    puntos = cnt.reshape(-1, 2)
                    for d in defectos:
                        if d[3] > 20:
                            cv2.circle(frame2, tuple(puntos[d[0]]), 5, (255, 255, 0), 2)
                            cv2.circle(frame2, tuple(puntos[d[1]]), 5, (255, 255, 0), 2)
                            cv2.circle(frame2, tuple(puntos[d[2]]), 5, (0, 0, 255), 2)

            lista = np.reshape(hull, (1, -1, 2))
            cv2.polylines(frame2, lista, True, (0, 255, 0), 3)
            center, radius = cv2.minEnclosingCircle(cnt)
            center = tuple(map(int, center))
            radius = int(radius)
            cv2.circle(frame2, center, radius, (255, 0, 0), 3)

        cv2.imshow('Contornos',frame2)

#Clase para lectura e interpretacion de datos
class Knn(Recognition):

    def __init__(self, rec):
        self.COLS = rec.COLS
        self.ROWS = rec.ROWS

    #Cargar datos del archivo
    def loadDataset(self,filename):
        responses = []
        trainData = []

        with open(filename, 'rb') as csvfile:
            lines = csv.reader(csvfile)
            trainData = list(lines)

            num = len(trainData[0])-1

            for i in range (len(trainData)):
                for j in range(num):
                    trainData[i][j] = int(trainData[i][j])
                x = len(trainData[i])
                responses.append(int(trainData[i][x-1]))
                trainData[i].pop()
        
        #convert the list of response on a list of lists
        finalResponse = [responses[1*i : 1*(i+1)] for i in range(len(trainData))]
        #convert the list on numpy arrays
        finalTrainData = np.array(trainData).astype(np.float32)
        finalResponse = np.array(finalResponse).astype(np.float32)

        return finalTrainData, finalResponse

    def upData(self, filename, occupation, pose):
        row = []
        finalRow = []
        with open(filename, 'a') as csvfile:
            actualMatrix = self.createGrid(occupation)
            lines = csv.writer(csvfile)
            for x in actualMatrix:
                row += x #Convertimos la matriz a array
            
            row.append(pose)  #Agregamos a que pose pertenece
            finalRow.append(row)
            lines.writerows('')
            lines.writerows(finalRow)

    def getNewGrid(self, data):
        size = self.ROWS * self.COLS
        row = []

        for x in data:
            row += x
        finalRow = [row[size*i : size*(i+1)] for i in range(1)]
        finalRow = np.array(finalRow).astype(np.float32)
        return finalRow

    def getNeighbors(self, trainData, responses,newcomer):
        knn = cv2.KNearest()
        knn.train(trainData, responses)
        ret, results, neighbours, dist = knn.find_nearest(newcomer, 15)

        #print "result: ", results, "\n"
        #print "neighbours: ", neighbours, "\n"
        #print "distance: ", dist
        return results

    def graficar(self, trainData, responses):
        plt.plot(trainData)
        plt.show() 
            
if __name__ == '__main__':
    posiciones = ["Normal", "B.Izquierdo", "B.Derecho", "Hola 01", "Hola 02", "Ven 01", "Alto", "Brazos"]
    p = 0
    #Inializacion del archivo y regilla
    print 'Press ESC in window to stop'  
    file = 'Capturas'#raw_input('File name: \n')
    size = 80#input('\nGrid Size:\n')

    #Creacion de los objetos
    rec = Recognition(size)
    k = Knn(rec)

    #Carga de los datos de un archivo
    #foo = raw_input('Load a File ( Y/N )')
    foo = 'y'
    if foo == 'y':
        #bar = raw_input('File to load:\n')
        bar = 'Datos80.data'
        datafile = 'Datos/'+bar
        a,b = k.loadDataset(datafile)


    data = [[0] * rec.COLS for i in range(rec.ROWS)]
           
    #k.graficar(a,b)
    while 1:
        mask = rec.getDepth()
        frame = rec.getVideo()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"Postura: "+ str(posiciones[p]),(20,440), font, 1.8,(255,0,0),4)
        rec.draw_grid(mask)
        rec.createGrid(data)
        rec.showFrame('RGB', frame)
        rec.showFrame('Depth', mask)
        
        newPos = k.getNewGrid(data)
        
        key = cv2.waitKey(5) & 0xFF


        if foo == 'y': #p 112
            prediction = k.getNeighbors(a,b,newPos)
            p = int(prediction[0][0])

        if key == 107: #k
        	#Ver Pose Actual
            print newPos

        if key == 48: #0
            k.upData( file+'.data', data, '0')
            k.upData( file+'b.data', data, '[0,0,0]')
            print 'Neutro'

        if key == 49: #1
            k.upData( file+'.data', data, '1')
            k.upData( file+'b.data', data, '[0,0,1]')
            print 'Izquierda'

        if key == 50: #2
            k.upData( file+'.data', data, '2')
            k.upData( file+'b.data', data, '[0,1,0]')
            print 'Derecha'
        
        if key == 51: #3
            k.upData( file+'.data', data, '3')
            k.upData( file+'b.data', data, '[0,1,1]')
            print 'Saludo01'
        
        if key == 52: #4
            k.upData( file+'.data', data, '4')
            k.upData( file+'b.data', data, '[1,0,0]')
            print 'Saludo02'

        if key == 53: #5
            k.upData( file+'.data', data, '5')
            k.upData( file+'b.data', data, '[1,0,1]')
            print 'Saludo03'

        if key == 54: #6
            k.upData( file+'.data', data, '6')
            k.upData( file+'b.data', data, '[1,1,0]')
            print 'Saludo04'

        if key == 55: #7
            k.upData( file+'.data', data, '7')
            k.upData( file+'b.data', data, '[1,1,1]')
            print 'Brazos Abiertos'


        if key == 27:  #escape
            break
            cv2.destroyAllWindows()

