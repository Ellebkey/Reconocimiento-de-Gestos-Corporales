[1mdiff --git a/Grid_knn.py b/Grid_knn.py[m
[1mdeleted file mode 100644[m
[1mindex ea777ea..0000000[m
[1m--- a/Grid_knn.py[m
[1m+++ /dev/null[m
[36m@@ -1,309 +0,0 @@[m
[31m-#!/usr/bin/env python[m
[31m-[m
[31m-##########################################################################[m
[31m-#Grid_knn.py:   Programa de reconocimiento de posturas corporales        #[m
[31m-#               usando un cuadriula de ocupacion y knn para su analisis  #[m
[31m-##########################################################################[m
[31m-[m
[31m-__author__  = 'Joel Barranco'[m
[31m-__email__   = 'contacto@joelbarranco.com'[m
[31m-__version__ = '1.0.1'[m
[31m-__license__ = 'MIT License'[m
[31m-[m
[31m-[m
[31m-'''Comienza Programa'''[m
[31m-[m
[31m-import freenect[m
[31m-import cv2[m
[31m-import numpy as np[m
[31m-import csv[m
[31m-import matplotlib.pyplot as plt[m
[31m-[m
[31m-#Clase para reconocimiento, todo lo relacionado con freenect y opencv[m
[31m-class Recognition:[m
[31m-        [m
[31m-    THRESHOLD = 100[m
[31m-    CURRENT_DEPTH = 800[m
[31m-    COLS = 0    [m
[31m-    ROWS = 0[m
[31m-    SIZE = 0[m
[31m-    POINTS = [][m
[31m-    dev = 0[m
[31m-[m
[31m-    def __init__(self, size):[m
[31m-        self.SIZE = size[m
[31m-        self.COLS = 640/size[m
[31m-        self.ROWS = 480/size[m
[31m-[m
[31m-    #Obtener Imagen de profundidad y aplicar la mascara a la profundidad establecida[m
[31m-    def getDepth(self):[m
[31m-[m
[31m-        depth,_ = freenect.sync_get_depth()[m
[31m-[m
[31m-        depth = cv2.GaussianBlur(depth,(3,3),0)[m
[31m-        [m
[31m-        depth = 255 * np.logical_and(depth >= self.CURRENT_DEPTH - self.THRESHOLD,[m
[31m-                                     depth <= self.CURRENT_DEPTH + self.THRESHOLD)[m
[31m-        depth = depth.astype(np.uint8)[m
[31m-[m
[31m-        return depth[m
[31m-[m
[31m-    #Obtener imagen RGB[m
[31m-    def getVideo(self):[m
[31m-        image,_ = freenect.sync_get_video()[m
[31m-        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)[m
[31m-        [m
[31m-        return image[m
[31m-[m
[31m-    #Mostrar las imagenes[m
[31m-    def showFrame(self,window_name , frame):[m
[31m-        cv2.namedWindow(window_name)[m
[31m-        cv2.imshow(window_name, frame)[m
[31m-[m
[31m-    #Dibujar la cuadricula[m
[31m-    def draw_grid(self,frame):[m
[31m-        y,x = frame.shape[m
[31m-        m = x/self.COLS[m
[31m-        n = y/self.ROWS[m
[31m-[m
[31m-        for i in range(self.COLS+1):[m
[31m-            for j in range(self.ROWS+1):[m
[31m-                coords = m*i,n*j[m
[31m-                cv2.circle(frame,coords,2,[255,0,255],2)[m
[31m-                if not (coords in self.POINTS):[m
[31m-                    self.POINTS.append((m*i,n*j))[m
[31m-[m
[31m-    #Calcula Area[m
[31m-    def getArea(self,frame):[m
[31m-        area = 0[m
[31m-        [m
[31m-        contours, jerarquia = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[m
[31m-[m
[31m-        if len(contours) > 0: [m
[31m-            cnt = contours[0][m
[31m-            moments = cv2.moments(cnt)[m
[31m-            area = moments['m00'][m
[31m-        return area[m
[31m-[m
[31m-    #Calcula porcentaje ocupado en la cuadricula[m
[31m-    def percent(self,part,whole):[m
[31m-        per = (part * 100)/whole[m
[31m-        return per[m
[31m-[m
[31m-    #Crear cuadricula[m
[31m-    def createGrid(self,gridOccupation):[m
[31m-        img = self.getDepth()[m
[31m-        x = self.SIZE-2[m
[31m-[m
[31m-        for i in range(self.COLS):[m
[31m-            for j in range(self.ROWS):[m
[31m-                #Creamos un ROI del tamanio de la cuadricula que recorra toda la imagen[m
[31m-                ROI = img[j*self.SIZE:j*self.SIZE+self.SIZE,i*self.SIZE:i*self.SIZE+self.SIZE][m
[31m-                #Calculamos el area en el ROI[m
[31m-                area = self.getArea(ROI)[m
[31m-                #Contamos el numero de pixeles que se ocupa en el grid[m
[31m-                n = cv2.countNonZero(ROI)[m
[31m-                if (area > 0) :[m
[31m-                    #calculamos el porcentaje de area ocupada y lo guardamos[m
[31m-                    p = self.percent(n,x*x)[m
[31m-                    gridOccupation[j][i] = p[m
[31m-        return gridOccupation[m
[31m-                    [m
[31m-[m
[31m-    def getContour(self):[m
[31m-[m
[31m-        frame = self.getDepth()[m
[31m-        frame2 = self.getVideo()[m
[31m-[m
[31m-        contours, jerarquia = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[m
[31m-[m
[31m-        for n, cnt in enumerate(contours):[m
[31m-[m
[31m-            hull = cv2.convexHull(cnt)[m
[31m-[m
[31m-            foo = cv2.convexHull(cnt, returnPoints = False)[m
[31m-            cv2.drawContours(frame2, contours, n, (0, 35, 245))[m
[31m-            if len(cnt) > 3 and len(foo) > 2:[m
[31m-                defectos = cv2.convexityDefects(cnt, foo)[m
[31m-                if defectos is not None:[m
[31m-                    defectos = defectos.reshape(-1, 4)[m
[31m-                    puntos = cnt.reshape(-1, 2)[m
[31m-                    for d in defectos:[m
[31m-                        if d[3] > 20:[m
[31m-                            cv2.circle(frame2, tuple(puntos[d[0]]), 5, (255, 255, 0), 2)[m
[31m-                            cv2.circle(frame2, tuple(puntos[d[1]]), 5, (255, 255, 0), 2)[m
[31m-                            cv2.circle(frame2, tuple(puntos[d[2]]), 5, (0, 0, 255), 2)[m
[31m-[m
[31m-            lista = np.reshape(hull, (1, -1, 2))[m
[31m-            cv2.polylines(frame2, lista, True, (0, 255, 0), 3)[m
[31m-            center, radius = cv2.minEnclosingCircle(cnt)[m
[31m-            center = tuple(map(int, center))[m
[31m-            radius = int(radius)[m
[31m-            cv2.circle(frame2, center, radius, (255, 0, 0), 3)[m
[31m-[m
[31m-        cv2.imshow('Contornos',frame2)[m
[31m-[m
[31m-#Clase para lectura e interpretacion de datos[m
[31m-class Knn(Recognition):[m
[31m-[m
[31m-    def __init__(self, rec):[m
[31m-        self.COLS = rec.COLS[m
[31m-        self.ROWS = rec.ROWS[m
[31m-[m
[31m-    #Cargar datos del archivo[m
[31m-    def loadDataset(self,filename):[m
[31m-        responses = [][m
[31m-        trainData = [][m
[31m-[m
[31m-        with open(filename, 'rb') as csvfile:[m
[31m-            lines = csv.reader(csvfile)[m
[31m-            trainData = list(lines)[m
[31m-[m
[31m-            num = len(trainData[0])-1[m
[31m-[m
[31m-            for i in range (len(trainData)):[m
[31m-                for j in range(num):[m
[31m-                    trainData[i][j] = int(trainData[i][j])[m
[31m-                x = len(trainData[i])[m
[31m-                responses.append(int(trainData[i][x-1]))[m
[31m-                trainData[i].pop()[m
[31m-        [m
[31m-        #convert the list of response on a list of lists[m
[31m-        finalResponse = [responses[1*i : 1*(i+1)] for i in range(len(trainData))][m
[31m-        #convert the list on numpy arrays[m
[31m-        finalTrainData = np.array(trainData).astype(np.float32)[m
[31m-        finalResponse = np.array(finalResponse).astype(np.float32)[m
[31m-[m
[31m-        return finalTrainData, finalResponse[m
[31m-[m
[31m-    def upData(self, filename, occupation, pose):[m
[31m-        row = [][m
[31m-        finalRow = [][m
[31m-        with open(filename, 'a') as csvfile:[m
[31m-            actualMatrix = self.createGrid(occupation)[m
[31m-            lines = csv.writer(csvfile)[m
[31m-            for x in actualMatrix:[m
[31m-                row += x #Convertimos la matriz a array[m
[31m-            [m
[31m-            row.append(pose)  #Agregamos a que pose pertenece[m
[31m-            finalRow.append(row)[m
[31m-            lines.writerows('')[m
[31m-            lines.writerows(finalRow)[m
[31m-[m
[31m-    def getNewGrid(self, data):[m
[31m-        size = self.ROWS * self.COLS[m
[31m-        row = [][m
[31m-[m
[31m-        for x in data:[m
[31m-            row += x[m
[31m-        finalRow = [row[size*i : size*(i+1)] for i in range(1)][m
[31m-        finalRow = np.array(finalRow).astype(np.float32)[m
[31m-        return finalRow[m
[31m-[m
[31m-    def getNeighbors(self, trainData, responses,newcomer):[m
[31m-        knn = cv2.KNearest()[m
[31m-        knn.train(trainData, responses)[m
[31m-        ret, results, neighbours, dist = knn.find_nearest(newcomer, 15)[m
[31m-[m
[31m-        #print "result: ", results, "\n"[m
[31m-        #print "neighbours: ", neighbours, "\n"[m
[31m-        #print "distance: ", dist[m
[31m-        return results[m
[31m-[m
[31m-    def graficar(self, trainData, responses):[m
[31m-        plt.plot(trainData)[m
[31m-        plt.show() [m
[31m-            [m
[31m-if __name__ == '__main__':[m
[31m-    posiciones = ["Normal", "B.Izquierdo", "B.Derecho", "Hola 01", "Hola 02", "Ven 01", "Alto", "Brazos"][m
[31m-    p = 0[m
[31m-    #Inializacion del archivo y regilla[m
[31m-    print 'Press ESC in window to stop'  [m
[31m-    file = 'Capturas'#raw_input('File name: \n')[m
[31m-    size = 80#input('\nGrid Size:\n')[m
[31m-[m
[31m-    #Creacion de los objetos[m
[31m-    rec = Recognition(size)[m
[31m-    k = Knn(rec)[m
[31m-[m
[31m-    #Carga de los datos de un archivo[m
[31m-    #foo = raw_input('Load a File ( Y/N )')[m
[31m-    foo = 'y'[m
[31m-    if foo == 'y':[m
[31m-        #bar = raw_input('File to load:\n')[m
[31m-        bar = 'Datos80.data'[m
[31m-        datafile = 'Datos/'+bar[m
[31m-        a,b = k.loadDataset(datafile)[m
[31m-[m
[31m-[m
[31m-    data = [[0] * rec.COLS for i in range(rec.ROWS)][m
[31m-           [m
[31m-    #k.graficar(a,b)[m
[31m-    while 1:[m
[31m-        mask = rec.getDepth()[m
[31m-        frame = rec.getVideo()[m
[31m-        font = cv2.FONT_HERSHEY_SIMPLEX[m
[31m-        cv2.putText(frame,"Postura: "+ str(posiciones[p]),(20,440), font, 1.8,(255,0,0),4)[m
[31m-        rec.draw_grid(mask)[m
[31m-        rec.createGrid(data)[m
[31m-        rec.showFrame('RGB', frame)[m
[31m-        rec.showFrame('Depth', mask)[m
[31m-        [m
[31m-        newPos = k.getNewGrid(data)[m
[31m-        [m
[31m-        key = cv2.waitKey(5) & 0xFF[m
[31m-[m
[31m-[m
[31m-        if foo == 'y': #p 112[m
[31m-            prediction = k.getNeighbors(a,b,newPos)[m
[31m-            p = int(prediction[0][0])[m
[31m-[m
[31m-        if key == 107: #k[m
[31m-        	#Ver Pose Actual[m
[31m-            print newPos[m
[31m-[m
[31m-        if key == 48: #0[m
[31m-            k.upData( file+'.data', data, '0')[m
[31m-            k.upData( file+'b.data', data, '[0,0,0]')[m
[31m-            print 'Neutro'[m
[31m-[m
[31m-        if key == 49: #1[m
[31m-            k.upData( file+'.data', data, '1')[m
[31m-            k.upData( file+'b.data', data, '[0,0,1]')[m
[31m-            print 'Izquierda'[m
[31m-[m
[31m-        if key == 50: #2[m
[31m-            k.upData( file+'.data', data, '2')[m
[31m-            k.upData( file+'b.data', data, '[0,1,0]')[m
[31m-            print 'Derecha'[m
[31m-        [m
[31m-        if key == 51: #3[m
[31m-            k.upData( file+'.data', data, '3')[m
[31m-            k.upData( file+'b.data', data, '[0,1,1]')[m
[31m-            print 'Saludo01'[m
[31m-        [m
[31m-        if key == 52: #4[m
[31m-            k.upData( file+'.data', data, '4')[m
[31m-            k.upData( file+'b.data', data, '[1,0,0]')[m
[31m-            print 'Saludo02'[m
[31m-[m
[31m-        if key == 53: #5[m
[31m-            k.upData( file+'.data', data, '5')[m
[31m-            k.upData( file+'b.data', data, '[1,0,1]')[m
[31m-            print 'Saludo03'[m
[31m-[m
[31m-        if key == 54: #6[m
[31m-            k.upData( file+'.data', data, '6')[m
[31m-            k.upData( file+'b.data', data, '[1,1,0]')[m
[31m-            print 'Saludo04'[m
[31m-[m
[31m-        if key == 55: #7[m
[31m-            k.upData( file+'.data', data, '7')[m
[31m-            k.upData( file+'b.data', data, '[1,1,1]')[m
[31m-            print 'Brazos Abiertos'[m
[31m-[m
[31m-[m
[31m-        if key == 27:  #escape[m
[31m-            break[m
[31m-            cv2.destroyAllWindows()[m
[31m-[m
[1mdiff --git a/Skeleton_NN.py b/Skeleton_NN.py[m
[1mdeleted file mode 100644[m
[1mindex 30d86ea..0000000[m
[1m--- a/Skeleton_NN.py[m
[1m+++ /dev/null[m
[36m@@ -1,267 +0,0 @@[m
[31m-#!/usr/bin/env python[m
[31m-[m
[31m-###########################################################################[m
[31m-#Skeleton_NN.py:   Programa de reconocimiento de posturas corporales      #[m
[31m-#                   detectando el esqueleto mediante la libreria openni   #[m
[31m-#                   y neuronal networks para su analisis                  #[m
[31m-###########################################################################[m
[31m-[m
[31m-__author__  = 'Joel Barranco'[m
[31m-__email__   = 'contacto@joelbarranco.com'[m
[31m-__version__ = '0.9.2'[m
[31m-__license__ = 'MIT License'[m
[31m-[m
[31m-[m
[31m-'''Comienza Programa'''[m
[31m-[m
[31m-from nimblenet.activation_functions import sigmoid_function, tanh_function, linear_function, LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function, softmax_function, softplus_function, softsign_function[m
[31m-from nimblenet.cost_functions import sum_squared_error, cross_entropy_cost, hellinger_distance, softmax_neg_loss[m
[31m-from nimblenet.learning_algorithms import backpropagation, scaled_conjugate_gradient, scipyoptimize, resilient_backpropagation[m
[31m-from nimblenet.evaluation_functions import binary_accuracy[m
[31m-from nimblenet.neuralnet import NeuralNet[m
[31m-from nimblenet.preprocessing import construct_preprocessor, standarize, replace_nan, whiten[m
[31m-from nimblenet.data_structures import Instance[m
[31m-from nimblenet.tools import print_test[m
[31m-from openni import *[m
[31m-import cv2[m
[31m-import numpy as np[m
[31m-import math[m
[31m-#Esta libreria son mis datos debes ingresar datos para entrenar tu red o ingresar una red ya entrenada en la linea 66[m
[31m-#import trainSk as tk[m
[31m-import binario as b[m
[31m-[m
[31m-limbs = {'Cuerpo' : (SKEL_HEAD, SKEL_NECK, SKEL_TORSO), 'Brazos' : (SKEL_LEFT_HAND, SKEL_LEFT_ELBOW, SKEL_LEFT_SHOULDER, SKEL_RIGHT_SHOULDER, SKEL_RIGHT_ELBOW, SKEL_RIGHT_HAND), 'Piernas' : (SKEL_RIGHT_FOOT, SKEL_RIGHT_KNEE, SKEL_RIGHT_HIP, SKEL_TORSO, SKEL_LEFT_HIP, SKEL_LEFT_KNEE, SKEL_LEFT_FOOT)}[m
[31m-posiciones = ["Normal", "B.Izquierdo", "B.Derecho", "Hola 01", "Hola 02", "Ven 01", "Ven 02", "Ven 03", "Brazos"][m
[31m-# Pose to use to calibrate the user[m
[31m-pose_to_use = 'Psi'[m
[31m-[m
[31m-dataset             = tk.DatosTrain()[m
[31m-preprocess          = construct_preprocessor( dataset, [standarize] ) [m
[31m-training_data       = preprocess( dataset )[m
[31m-test_data           = preprocess( dataset )[m
[31m-[m
[31m-[m
[31m-cost_function       = cross_entropy_cost[m
[31m-[m
[31m-settings            = {[m
[31m-    # Required settings[m
[31m-    "n_inputs"              : 14,       # Number of network input signals[m
[31m-    "layers"                : [  (5, sigmoid_function), (4, sigmoid_function) ],[m
[31m-                                        # [ (number_of_neurons, activation_function) ][m
[31m-                                        # The last pair in the list dictate the number of output signals[m
[31m-    [m
[31m-    # Optional settings[m
[31m-    "weights_low"           : -0.1,    # Lower bound on the initial weight value[m
[31m-    "weights_high"          : 0.4,      # Upper bound on the initial weight value[m
[31m-}[m
[31m-[m
[31m-[m
[31m-# initialize the neural network[m
[31m-network             = NeuralNet( settings )[m
[31m-network.check_gradient( training_data, cost_function )[m
[31m-[m
[31m-[m
[31m-[m
[31m-## load a stored network configuration[m
[31m-network           = NeuralNet.load_network_from_file( "redsk_00009.pkl" )[m
[31m-[m
[31m-ctx = Context()[m
[31m-ctx.init()[m
[31m-[m
[31m-# Create the user generator[m
[31m-user = UserGenerator()[m
[31m-user.create(ctx)[m
[31m-[m
[31m-#Obtener imagen[m
[31m-depth = DepthGenerator()[m
[31m-depth.create(ctx)[m
[31m-depth.set_resolution_preset(DefResolution.RES_VGA)[m
[31m-depth.fps = 30[m
[31m-ctx.start_generating_all()[m
[31m-[m
[31m-image = ImageGenerator()[m
[31m-image.create(ctx)[m
[31m-image.set_resolution_preset(DefResolution.RES_VGA)[m
[31m-image.fps = 30[m
[31m-ctx.start_generating_all()[m
[31m-[m
[31m-# Obtain the skeleton & pose detection capabilities[m
[31m-skel_cap = user.skeleton_cap[m
[31m-pose_cap = user.pose_detection_cap[m
[31m-[m
[31m-# Declare the callbacks[m
[31m-def new_user(src, id):[m
[31m-    print "1/4 User {} detected. Looking for pose..." .format(id)[m
[31m-    pose_cap.start_detection(pose_to_use, id)[m
[31m-[m
[31m-def pose_detected(src, pose, id):[m
[31m-    print "2/4 Detected pose {} on user {}. Requesting calibration..." .format(pose,id)[m
[31m-    pose_cap.stop_detection(id)[m
[31m-    skel_cap.request_calibration(id, True)[m
[31m-[m
[31m-def calibration_start(src, id):[m
[31m-    print "3/4 Calibration started for user {}." .format(id)[m
[31m-[m
[31m-def calibration_complete(src, id, status):[m
[31m-    if status == CALIBRATION_STATUS_OK:[m
[31m-        print "4/4 User {} calibrated successfully! Starting to track." .format(id)[m
[31m-        skel_cap.start_tracking(id)[m
[31m-    else:[m
[31m-        print "ERR User {} failed to calibrate. Restarting process." .format(id)[m
[31m-        new_user(user, id)[m
[31m-[m
[31m-def lost_user(src, id):[m
[31m-    print "--- User {} lost." .format(id)[m
[31m-[m
[31m-def formDeg(u1,u2,v1,v2):[m
[31m-    i = (u1*v1)+(u2*v2)[m
[31m-    math.fabs(i)[m
[31m-    j = (math.sqrt((u1**2)+(u2**2)))*(math.sqrt((v1**2)+(v2**2)))[m
[31m-    final = i/j[m
[31m-    return final[m
[31m-[m
[31m-def calcDeg(point1,point2,point3):[m
[31m-    u = [][m
[31m-    v = [][m
[31m-[m
[31m-    x1 = point3[0]-point2[0][m
[31m-    u.append(x1)[m
[31m-    y1 = point3[1]-point2[1][m
[31m-    u.append(y1)[m
[31m-[m
[31m-    x2 = point1[0]-point2[0][m
[31m-    v.append(x2)[m
[31m-    y2 = point1[1]-point2[1][m
[31m-    v.append(y2)[m
[31m-[m
[31m-    ang = formDeg(u[0],u[1],v[0],v[1])[m
[31m-[m
[31m-    ang = math.acos(ang)[m
[31m-    ang = math.degrees(ang)[m
[31m-    return ang[m
[31m-[m
[31m-def distancia(p1,p2):[m
[31m-    d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))[m
[31m-    return d[m
[31m-[m
[31m-def upData(filename, data, pose):[m
[31m-[m
[31m-    with open(filename, 'a') as csvfile:[m
[31m-        lines = csv.writer(csvfile)[m
[31m-        #print data[m
[31m-        data.append(pose)  #Agregamos a que pose pertenece[m
[31m-        lines.writerow(data)[m
[31m-[m
[31m-# Register them[m
[31m-user.register_user_cb(new_user, lost_user)[m
[31m-pose_cap.register_pose_detected_cb(pose_detected)[m
[31m-skel_cap.register_c_start_cb(calibration_start)[m
[31m-skel_cap.register_c_complete_cb(calibration_complete)[m
[31m-[m
[31m-# Set the profile[m
[31m-skel_cap.set_profile(SKEL_PROFILE_ALL)[m
[31m-[m
[31m-# Start generating[m
[31m-ctx.start_generating_all()[m
[31m-print "0/4 Starting to detect users. Press Ctrl-C to exit."[m
[31m-[m
[31m-datos = [0] * 14[m
[31m-grado = [0] * 4[m
[31m-distancias = [0] * 4[m
[31m-while True:[m
[31m-    # Update to next frame[m
[31m-    ctx.wait_and_update_all()[m
[31m-[m
[31m-    blank_image = np.zeros((480,640,3), np.uint8)[m
[31m-    blank_image[:,0:] = (255,255,255)[m
[31m-    frame = np.fromstring(depth.get_raw_depth_map_8(), "uint8").reshape(480,640)[m
[31m-    bgrframe = np.fromstring(image.get_raw_image_map_bgr(), dtype=np.uint8).reshape(image.metadata.res[1],image.metadata.res[0],3)[m
[31m-[m
[31m-    font = cv2.FONT_HERSHEY_SIMPLEX[m
[31m-    [m
[31m-    # Extract head position of each tracked user[m
[31m-    points = {}[m
[31m-    for id in user.users:[m
[31m-[m
[31m-        if skel_cap.is_tracking(id):[m
[31m-            for limb in limbs:[m
[31m-                points[limb] = [][m
[31m-                [m
[31m-                for junta in limbs[limb]:[m
[31m-                    points[limb].append(skel_cap.get_joint_position(id, junta).point)[m
[31m-[m
[31m-            for limb in points:[m
[31m-                points[limb] = depth.to_projective(points[limb])[m
[31m-                for i in xrange(len(points[limb])):[m
[31m-                    cv2.circle(blank_image, (int(points[limb][i][0]),int(points[limb][i][1])), 4,[0,255,0],10)  [m
[31m-                    if i+1 < len(points[limb]):[m
[31m-                        cv2.line(blank_image,(int(points[limb][i][0]),int(points[limb][i][1])),(int(points[limb][i+1][0]),int(points[limb][i+1][1])),(255,0,0),5)[m
[31m-            #Los puntos de la cabeza[m
[31m-            #'Brazos' : (SKEL_LEFT_HAND, SKEL_LEFT_ELBOW, SKEL_LEFT_SHOULDER, SKEL_RIGHT_SHOULDER, SKEL_RIGHT_ELBOW, SKEL_RIGHT_HAND)[m
[31m-            p1 = points['Brazos'][0][:2][m
[31m-            p2 = points['Brazos'][1][:2][m
[31m-            p3 = points['Brazos'][2][:2][m
[31m-[m
[31m-            a1 = points['Brazos'][3][:2][m
[31m-            a2 = points['Brazos'][4][:2][m
[31m-            a3 = points['Brazos'][5][:2][m
[31m-            [m
[31m-            b1 = points['Cuerpo'][1][:2][m
[31m-            b2 = points['Brazos'][2][:2][m
[31m-            b3 = points['Brazos'][1][:2][m
[31m-            [m
[31m-            c1 = points['Cuerpo'][1][:2][m
[31m-            c2 = points['Brazos'][3][:2][m
[31m-            c3 = points['Brazos'][4][:2][m
[31m-[m
[31m-            dist1 = points['Cuerpo'][1][:2][m
[31m-            dist2 = points['Cuerpo'][2][:2][m
[31m-[m
[31m-            d = distancia(dist1,dist2)[m
[31m-            d1 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][0][:2])[m
[31m-            d2 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][1][:2])[m
[31m-            d3 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][4][:2])[m
[31m-            d4 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][5][:2])[m
[31m-[m
[31m-            distancias[0] = d1 / d[m
[31m-            distancias[1] = d2 / d[m
[31m-            distancias[2] = d3 / d[m
[31m-            distancias[3] = d4 / d[m
[31m-[m
[31m-            grado[0] = calcDeg(p1,p2,p3)[m
[31m-            grado[1] = calcDeg(a1,a2,a3)[m
[31m-            grado[2] = calcDeg(b1,b2,b3)[m
[31m-            grado[3] = calcDeg(c1,c2,c3)[m
[31m-[m
[31m-            for k in range(4):[m
[31m-                datos[k+10] = round(distancias[k],2)[m
[31m-[m
[31m-            for i in range(4):[m
[31m-                datos[i+6] = round(grado[i],2)[m
[31m-[m
[31m-            z = points['Cuerpo'][2][2][m
[31m-[m
[31m-            for j in range(6):[m
[31m-                datos[j] = round(points['Brazos'][j][2]-z,2)[m
[31m-[m
[31m-            prediction_set = [ Instance(datos)][m
[31m-            prediction_set = preprocess( prediction_set )[m
[31m-            #print "\nPrediccion:"[m
[31m-            prediction = network.predict( prediction_set ) # produce the output signal[m
[31m-            #print prediction[0][0][m
[31m-            predicFormat = b.convertBin(prediction[0])[m
[31m-            [m
[31m-            cv2.putText(blank_image,str(predicFormat),(30,50), font, 1.0,(0,0,255),2)    [m
[31m-[m
[31m-[m
[31m-    cv2.imshow("Depth", blank_image)[m
[31m-    cv2.imshow("RGB", bgrframe)[m
[31m-[m
[31m-    k = cv2.waitKey(5)&0xFF[m
[31m-[m
[31m-    if k == 27:[m
[31m-        break[m
[31m-[m
[31m-print datos[m
[31m-cv2.destroyAllWindows()[m
[31m-[m
[1mdiff --git a/Skeleton_knn.py b/Skeleton_knn.py[m
[1mdeleted file mode 100644[m
[1mindex ef47645..0000000[m
[1m--- a/Skeleton_knn.py[m
[1m+++ /dev/null[m
[36m@@ -1,276 +0,0 @@[m
[31m-#!/usr/bin/env python[m
[31m-[m
[31m-###########################################################################[m
[31m-#Skeleton_knn.py:   Programa de reconocimiento de posturas corporales     #[m
[31m-#                   detectando el esqueleto mediante la libreria openni   #[m
[31m-#                   y knn para su analisis                                #[m
[31m-###########################################################################[m
[31m-[m
[31m-__author__  = 'Joel Barranco'[m
[31m-__email__   = 'contacto@joelbarranco.com'[m
[31m-__version__ = '0.9.8'[m
[31m-__license__ = 'MIT License'[m
[31m-[m
[31m-[m
[31m-'''Comienza Programa'''[m
[31m-[m
[31m-from openni import *[m
[31m-import cv2[m
[31m-from cv2 import *[m
[31m-import numpy as np[m
[31m-import math[m
[31m-import knn as k[m
[31m-import csv[m
[31m-[m
[31m-limbs = {'Cuerpo' : (SKEL_HEAD, SKEL_NECK, SKEL_TORSO), 'Brazos' : (SKEL_LEFT_HAND, SKEL_LEFT_ELBOW, SKEL_LEFT_SHOULDER, SKEL_RIGHT_SHOULDER, SKEL_RIGHT_ELBOW, SKEL_RIGHT_HAND), 'Piernas' : (SKEL_RIGHT_FOOT, SKEL_RIGHT_KNEE, SKEL_RIGHT_HIP, SKEL_TORSO, SKEL_LEFT_HIP, SKEL_LEFT_KNEE, SKEL_LEFT_FOOT)}[m
[31m-posiciones = ["Normal", "B.Izquierdo", "B.Derecho", "Hola 01", "Hola 02", "Ven 01", "Ven 02", "Ven 03", "Brazos"][m
[31m-# Pose to use to calibrate the user[m
[31m-pose_to_use = 'Psi'[m
[31m-[m
[31m-ctx = Context()[m
[31m-ctx.init()[m
[31m-[m
[31m-# Create the user generator[m
[31m-user = UserGenerator()[m
[31m-user.create(ctx)[m
[31m-[m
[31m-#Obtener imagen[m
[31m-depth = DepthGenerator()[m
[31m-depth.create(ctx)[m
[31m-depth.set_resolution_preset(DefResolution.RES_VGA)[m
[31m-depth.fps = 30[m
[31m-ctx.start_generating_all()[m
[31m-[m
[31m-image = ImageGenerator()[m
[31m-image.create(ctx)[m
[31m-image.set_resolution_preset(DefResolution.RES_VGA)[m
[31m-image.fps = 30[m
[31m-ctx.start_generating_all()[m
[31m-[m
[31m-# Obtain the skeleton & pose detection capabilities[m
[31m-skel_cap = user.skeleton_cap[m
[31m-pose_cap = user.pose_detection_cap[m
[31m-[m
[31m-# Declare the callbacks[m
[31m-def new_user(src, id):[m
[31m-    print "1/4 User {} detected. Looking for pose..." .format(id)[m
[31m-    pose_cap.start_detection(pose_to_use, id)[m
[31m-[m
[31m-def pose_detected(src, pose, id):[m
[31m-    print "2/4 Detected pose {} on user {}. Requesting calibration..." .format(pose,id)[m
[31m-    pose_cap.stop_detection(id)[m
[31m-    skel_cap.request_calibration(id, True)[m
[31m-[m
[31m-def calibration_start(src, id):[m
[31m-    print "3/4 Calibration started for user {}." .format(id)[m
[31m-[m
[31m-def calibration_complete(src, id, status):[m
[31m-    if status == CALIBRATION_STATUS_OK:[m
[31m-        print "4/4 User {} calibrated successfully! Starting to track." .format(id)[m
[31m-        skel_cap.start_tracking(id)[m
[31m-    else:[m
[31m-        print "ERR User {} failed to calibrate. Restarting process." .format(id)[m
[31m-        new_user(user, id)[m
[31m-[m
[31m-def lost_user(src, id):[m
[31m-    print "--- User {} lost." .format(id)[m
[31m-[m
[31m-def formDeg(u1,u2,v1,v2):[m
[31m-    i = (u1*v1)+(u2*v2)[m
[31m-    math.fabs(i)[m
[31m-    j = (math.sqrt((u1**2)+(u2**2)))*(math.sqrt((v1**2)+(v2**2)))[m
[31m-    final = i/j[m
[31m-    return final[m
[31m-[m
[31m-def calcDeg(point1,point2,point3):[m
[31m-    u = [][m
[31m-    v = [][m
[31m-[m
[31m-    x1 = point3[0]-point2[0][m
[31m-    u.append(x1)[m
[31m-    y1 = point3[1]-point2[1][m
[31m-    u.append(y1)[m
[31m-[m
[31m-    x2 = point1[0]-point2[0][m
[31m-    v.append(x2)[m
[31m-    y2 = point1[1]-point2[1][m
[31m-    v.append(y2)[m
[31m-[m
[31m-    ang = formDeg(u[0],u[1],v[0],v[1])[m
[31m-[m
[31m-    ang = math.acos(ang)[m
[31m-    ang = math.degrees(ang)[m
[31m-    return ang[m
[31m-[m
[31m-def distancia(p1,p2):[m
[31m-    d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))[m
[31m-    return d[m
[31m-[m
[31m-def getNeighbors(trainData, responses,newcomer,vecinos):[m
[31m-    knn = cv2.KNearest()[m
[31m-    knn.train(trainData, responses)[m
[31m-    ret, results, neighbours, dist = knn.find_nearest(newcomer, vecinos)[m
[31m-[m
[31m-    #print "result: ", results, "\n"[m
[31m-    #print "neighbours: ", neighbours, "\n"[m
[31m-    #print "distance: ", dist[m
[31m-    return results[m
[31m-[m
[31m-def upData(filename, data, pose):[m
[31m-[m
[31m-    with open(filename, 'a') as csvfile:[m
[31m-        lines = csv.writer(csvfile)[m
[31m-        #print data[m
[31m-        data.append(pose)  #Agregamos a que pose pertenece[m
[31m-        lines.writerow(data)[m
[31m-[m
[31m-def loadDataset(filename):[m
[31m-    responses = [][m
[31m-    trainData = [][m
[31m-[m
[31m-    with open(filename, 'rb') as csvfile:[m
[31m-        lines = csv.reader(csvfile)[m
[31m-        trainData = list(lines)[m
[31m-[m
[31m-        num = len(trainData[0])-1[m
[31m-[m
[31m-        for i in range (len(trainData)):[m
[31m-            for j in range(num):[m
[31m-                trainData[i][j] = float(trainData[i][j])[m
[31m-            x = len(trainData[i])[m
[31m-            responses.append(float(trainData[i][x-1]))[m
[31m-            trainData[i].pop()[m
[31m-    [m
[31m-    #convert the list of response on a list of lists[m
[31m-    finalResponse = [responses[1*i : 1*(i+1)] for i in range(len(trainData))][m
[31m-    #convert the list on numpy arrays[m
[31m-    finalTrainData = np.array(trainData).astype(np.float32)[m
[31m-    finalResponse = np.array(finalResponse).astype(np.float32)[m
[31m-[m
[31m-    return finalTrainData, finalResponse[m
[31m-[m
[31m-def getNewPos( data):[m
[31m-    finalRow = [data[14*i : 14*(i+1)] for i in range(1)][m
[31m-    finalRow = np.array(finalRow).astype(np.float32)[m
[31m-    return finalRow[m
[31m-[m
[31m-# Register them[m
[31m-user.register_user_cb(new_user, lost_user)[m
[31m-pose_cap.register_pose_detected_cb(pose_detected)[m
[31m-skel_cap.register_c_start_cb(calibration_start)[m
[31m-skel_cap.register_c_complete_cb(calibration_complete)[m
[31m-[m
[31m-# Set the profile[m
[31m-skel_cap.set_profile(SKEL_PROFILE_ALL)[m
[31m-[m
[31m-# Start generating[m
[31m-ctx.start_generating_all()[m
[31m-print "0/4 Starting to detect users. Press Ctrl-C to exit."[m
[31m-[m
[31m-file = raw_input('File name: \n')[m
[31m-[m
[31m-datos = [0] * 14[m
[31m-grado = [0] * 4[m
[31m-distancias = [0] * 4[m
[31m-[m
[31m-[m
[31m-datafile = 'Datos/knnSke.data'[m
[31m-a,b = loadDataset(datafile)[m
[31m-[m
[31m-while True:[m
[31m-    # Update to next frame[m
[31m-    ctx.wait_and_update_all()[m
[31m-[m
[31m-    blank_image = np.zeros((480,640,3), np.uint8)[m
[31m-    blank_image[:,0:] = (255,255,255)[m
[31m-    frame = np.fromstring(depth.get_raw_depth_map_8(), "uint8").reshape(480,640)[m
[31m-    bgrframe = np.fromstring(image.get_raw_image_map_bgr(), dtype=np.uint8).reshape(image.metadata.res[1],image.metadata.res[0],3)[m
[31m-[m
[31m-    font = cv2.FONT_HERSHEY_SIMPLEX[m
[31m-    [m
[31m-    # Extract head position of each tracked user[m
[31m-    points = {}[m
[31m-    for id in user.users:[m
[31m-[m
[31m-        if skel_cap.is_tracking(id):[m
[31m-            for limb in limbs:[m
[31m-                points[limb] = [][m
[31m-                [m
[31m-                for junta in limbs[limb]:[m
[31m-                    points[limb].append(skel_cap.get_joint_position(id, junta).point)[m
[31m-[m
[31m-            for limb in points:[m
[31m-                points[limb] = depth.to_projective(points[limb])[m
[31m-                for i in xrange(len(points[limb])):[m
[31m-                    cv2.circle(blank_image, (int(points[limb][i][0]),int(points[limb][i][1])), 4,[0,255,0],10)  [m
[31m-                    if i+1 < len(points[limb]):[m
[31m-                        cv2.line(blank_image,(int(points[limb][i][0]),int(points[limb][i][1])),(int(points[limb][i+1][0]),int(points[limb][i+1][1])),(255,0,0),5)[m
[31m-            #Los puntos de la cabeza[m
[31m-            #'Brazos' : (SKEL_LEFT_HAND, SKEL_LEFT_ELBOW, SKEL_LEFT_SHOULDER, SKEL_RIGHT_SHOULDER, SKEL_RIGHT_ELBOW, SKEL_RIGHT_HAND)[m
[31m-            p1 = points['Brazos'][0][:2][m
[31m-            p2 = points['Brazos'][1][:2][m
[31m-            p3 = points['Brazos'][2][:2][m
[31m-[m
[31m-            a1 = points['Brazos'][3][:2][m
[31m-            a2 = points['Brazos'][4][:2][m
[31m-            a3 = points['Brazos'][5][:2][m
[31m-            [m
[31m-            b1 = points['Cuerpo'][1][:2][m
[31m-            b2 = points['Brazos'][2][:2][m
[31m-            b3 = points['Brazos'][1][:2][m
[31m-            [m
[31m-            c1 = points['Cuerpo'][1][:2][m
[31m-            c2 = points['Brazos'][3][:2][m
[31m-            c3 = points['Brazos'][4][:2][m
[31m-[m
[31m-            dist1 = points['Cuerpo'][1][:2][m
[31m-            dist2 = points['Cuerpo'][2][:2][m
[31m-[m
[31m-            d = distancia(dist1,dist2)[m
[31m-            d1 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][0][:2])[m
[31m-            d2 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][1][:2])[m
[31m-            d3 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][4][:2])[m
[31m-            d4 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][5][:2])[m
[31m-[m
[31m-            distancias[0] = d1 / d[m
[31m-            distancias[1] = d2 / d[m
[31m-            distancias[2] = d3 / d[m
[31m-            distancias[3] = d4 / d[m
[31m-[m
[31m-            grado[0] = calcDeg(p1,p2,p3)[m
[31m-            grado[1] = calcDeg(a1,a2,a3)[m
[31m-            grado[2] = calcDeg(b1,b2,b3)[m
[31m-            grado[3] = calcDeg(c1,c2,c3)[m
[31m-[m
[31m-            for k in range(4):[m
[31m-                datos[k+10] = round(distancias[k],2)[m
[31m-[m
[31m-            for i in range(4):[m
[31m-                datos[i+6] = round(grado[i],2)[m
[31m-[m
[31m-            z = points['Cuerpo'][2][2][m
[31m-[m
[31m-            for j in range(6):[m
[31m-                datos[j] = round(points['Brazos'][j][2]-z,2)[m
[31m-[m
[31m-            #Se cargan las posiciones actuales para su predicción[m
[31m-            newPos = getNewPos(datos)[m
[31m-            #Se ingresa el nuevo set de datos en el knn[m
[31m-            prediction = getNeighbors(a,b,newPos,3)[m
[31m-            [m
[31m-            #Se muestra en pantalla la postura [m
[31m-            cv2.putText(blank_image,str(posiciones[int(prediction[0][0])]),(30,50), font, 1.0,(0,0,255),2)    [m
[31m-[m
[31m-[m
[31m-    cv2.imshow("Depth", blank_image)[m
[31m-    cv2.imshow("RGB", bgrframe)[m
[31m-[m
[31m-    k = cv2.waitKey(5)&0xFF[m
[31m-[m
[31m-   [m
[31m-    if k == 27:[m
[31m-        break[m
[31m-[m
[31m-print datos[m
[31m-cv2.destroyAllWindows()[m
[31m-[m
