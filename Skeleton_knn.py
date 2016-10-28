#!/usr/bin/env python

###########################################################################
#Skeleton_knn.py:   Programa de reconocimiento de posturas corporales     #
#                   detectando el esqueleto mediante la libreria openni   #
#                   y knn para su analisis                                #
###########################################################################

__author__  = 'Joel Barranco'
__email__   = 'contacto@joelbarranco.com'
__version__ = '0.9.8'
__license__ = 'MIT License'


'''Comienza Programa'''

from openni import *
import cv2
from cv2 import *
import numpy as np
import math
import knn as k
import csv

limbs = {'Cuerpo' : (SKEL_HEAD, SKEL_NECK, SKEL_TORSO), 'Brazos' : (SKEL_LEFT_HAND, SKEL_LEFT_ELBOW, SKEL_LEFT_SHOULDER, SKEL_RIGHT_SHOULDER, SKEL_RIGHT_ELBOW, SKEL_RIGHT_HAND), 'Piernas' : (SKEL_RIGHT_FOOT, SKEL_RIGHT_KNEE, SKEL_RIGHT_HIP, SKEL_TORSO, SKEL_LEFT_HIP, SKEL_LEFT_KNEE, SKEL_LEFT_FOOT)}
posiciones = ["Normal", "B.Izquierdo", "B.Derecho", "Hola 01", "Hola 02", "Ven 01", "Ven 02", "Ven 03", "Brazos"]
# Pose to use to calibrate the user
pose_to_use = 'Psi'

ctx = Context()
ctx.init()

# Create the user generator
user = UserGenerator()
user.create(ctx)

#Obtener imagen
depth = DepthGenerator()
depth.create(ctx)
depth.set_resolution_preset(DefResolution.RES_VGA)
depth.fps = 30
ctx.start_generating_all()

image = ImageGenerator()
image.create(ctx)
image.set_resolution_preset(DefResolution.RES_VGA)
image.fps = 30
ctx.start_generating_all()

# Obtain the skeleton & pose detection capabilities
skel_cap = user.skeleton_cap
pose_cap = user.pose_detection_cap

# Declare the callbacks
def new_user(src, id):
    print "1/4 User {} detected. Looking for pose..." .format(id)
    pose_cap.start_detection(pose_to_use, id)

def pose_detected(src, pose, id):
    print "2/4 Detected pose {} on user {}. Requesting calibration..." .format(pose,id)
    pose_cap.stop_detection(id)
    skel_cap.request_calibration(id, True)

def calibration_start(src, id):
    print "3/4 Calibration started for user {}." .format(id)

def calibration_complete(src, id, status):
    if status == CALIBRATION_STATUS_OK:
        print "4/4 User {} calibrated successfully! Starting to track." .format(id)
        skel_cap.start_tracking(id)
    else:
        print "ERR User {} failed to calibrate. Restarting process." .format(id)
        new_user(user, id)

def lost_user(src, id):
    print "--- User {} lost." .format(id)

def formDeg(u1,u2,v1,v2):
    i = (u1*v1)+(u2*v2)
    math.fabs(i)
    j = (math.sqrt((u1**2)+(u2**2)))*(math.sqrt((v1**2)+(v2**2)))
    final = i/j
    return final

def calcDeg(point1,point2,point3):
    u = []
    v = []

    x1 = point3[0]-point2[0]
    u.append(x1)
    y1 = point3[1]-point2[1]
    u.append(y1)

    x2 = point1[0]-point2[0]
    v.append(x2)
    y2 = point1[1]-point2[1]
    v.append(y2)

    ang = formDeg(u[0],u[1],v[0],v[1])

    ang = math.acos(ang)
    ang = math.degrees(ang)
    return ang

def distancia(p1,p2):
    d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))
    return d

def getNeighbors(trainData, responses,newcomer,vecinos):
    knn = cv2.KNearest()
    knn.train(trainData, responses)
    ret, results, neighbours, dist = knn.find_nearest(newcomer, vecinos)

    #print "result: ", results, "\n"
    #print "neighbours: ", neighbours, "\n"
    #print "distance: ", dist
    return results

def upData(filename, data, pose):

    with open(filename, 'a') as csvfile:
        lines = csv.writer(csvfile)
        #print data
        data.append(pose)  #Agregamos a que pose pertenece
        lines.writerow(data)

def loadDataset(filename):
    responses = []
    trainData = []

    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        trainData = list(lines)

        num = len(trainData[0])-1

        for i in range (len(trainData)):
            for j in range(num):
                trainData[i][j] = float(trainData[i][j])
            x = len(trainData[i])
            responses.append(float(trainData[i][x-1]))
            trainData[i].pop()
    
    #convert the list of response on a list of lists
    finalResponse = [responses[1*i : 1*(i+1)] for i in range(len(trainData))]
    #convert the list on numpy arrays
    finalTrainData = np.array(trainData).astype(np.float32)
    finalResponse = np.array(finalResponse).astype(np.float32)

    return finalTrainData, finalResponse

def getNewPos( data):
    finalRow = [data[14*i : 14*(i+1)] for i in range(1)]
    finalRow = np.array(finalRow).astype(np.float32)
    return finalRow

# Register them
user.register_user_cb(new_user, lost_user)
pose_cap.register_pose_detected_cb(pose_detected)
skel_cap.register_c_start_cb(calibration_start)
skel_cap.register_c_complete_cb(calibration_complete)

# Set the profile
skel_cap.set_profile(SKEL_PROFILE_ALL)

# Start generating
ctx.start_generating_all()
print "0/4 Starting to detect users. Press Ctrl-C to exit."

file = raw_input('File name: \n')

datos = [0] * 14
grado = [0] * 4
distancias = [0] * 4


datafile = 'Datos/knnSke.data'
a,b = loadDataset(datafile)

while True:
    # Update to next frame
    ctx.wait_and_update_all()

    blank_image = np.zeros((480,640,3), np.uint8)
    blank_image[:,0:] = (255,255,255)
    frame = np.fromstring(depth.get_raw_depth_map_8(), "uint8").reshape(480,640)
    bgrframe = np.fromstring(image.get_raw_image_map_bgr(), dtype=np.uint8).reshape(image.metadata.res[1],image.metadata.res[0],3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Extract head position of each tracked user
    points = {}
    for id in user.users:

        if skel_cap.is_tracking(id):
            for limb in limbs:
                points[limb] = []
                
                for junta in limbs[limb]:
                    points[limb].append(skel_cap.get_joint_position(id, junta).point)

            for limb in points:
                points[limb] = depth.to_projective(points[limb])
                for i in xrange(len(points[limb])):
                    cv2.circle(blank_image, (int(points[limb][i][0]),int(points[limb][i][1])), 4,[0,255,0],10)  
                    if i+1 < len(points[limb]):
                        cv2.line(blank_image,(int(points[limb][i][0]),int(points[limb][i][1])),(int(points[limb][i+1][0]),int(points[limb][i+1][1])),(255,0,0),5)
            #Los puntos de la cabeza
            #'Brazos' : (SKEL_LEFT_HAND, SKEL_LEFT_ELBOW, SKEL_LEFT_SHOULDER, SKEL_RIGHT_SHOULDER, SKEL_RIGHT_ELBOW, SKEL_RIGHT_HAND)
            p1 = points['Brazos'][0][:2]
            p2 = points['Brazos'][1][:2]
            p3 = points['Brazos'][2][:2]

            a1 = points['Brazos'][3][:2]
            a2 = points['Brazos'][4][:2]
            a3 = points['Brazos'][5][:2]
            
            b1 = points['Cuerpo'][1][:2]
            b2 = points['Brazos'][2][:2]
            b3 = points['Brazos'][1][:2]
            
            c1 = points['Cuerpo'][1][:2]
            c2 = points['Brazos'][3][:2]
            c3 = points['Brazos'][4][:2]

            dist1 = points['Cuerpo'][1][:2]
            dist2 = points['Cuerpo'][2][:2]

            d = distancia(dist1,dist2)
            d1 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][0][:2])
            d2 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][1][:2])
            d3 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][4][:2])
            d4 =  distancia(points['Cuerpo'][2][:2], points['Brazos'][5][:2])

            distancias[0] = d1 / d
            distancias[1] = d2 / d
            distancias[2] = d3 / d
            distancias[3] = d4 / d

            grado[0] = calcDeg(p1,p2,p3)
            grado[1] = calcDeg(a1,a2,a3)
            grado[2] = calcDeg(b1,b2,b3)
            grado[3] = calcDeg(c1,c2,c3)

            for k in range(4):
                datos[k+10] = round(distancias[k],2)

            for i in range(4):
                datos[i+6] = round(grado[i],2)

            z = points['Cuerpo'][2][2]

            for j in range(6):
                datos[j] = round(points['Brazos'][j][2]-z,2)

            #Se cargan las posiciones actuales para su predicción
            newPos = getNewPos(datos)
            #Se ingresa el nuevo set de datos en el knn
            prediction = getNeighbors(a,b,newPos,3)
            
            #Se muestra en pantalla la postura 
            cv2.putText(blank_image,str(posiciones[int(prediction[0][0])]),(30,50), font, 1.0,(0,0,255),2)    


    cv2.imshow("Depth", blank_image)
    cv2.imshow("RGB", bgrframe)

    k = cv2.waitKey(5)&0xFF

   
    if k == 27:
        break

print datos
cv2.destroyAllWindows()

