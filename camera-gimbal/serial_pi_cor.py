#!/usr/bin/python
# -*- coding:utf-8 -*-
import serial

ser = serial.Serial("/dev/ttyAMA0",9600,8,'N',1,timeout = 0.5) #树莓派
#ser = serial.Serial("/dev/ttyTHS1",9600) #jetson nano

print("serial start ...")
print("CTRL + c is end!")

#ser.write(bytes('$A010#','utf-8'))
try:
    while True:

        todao = input('please input way(A-X):')
        while todao >'X' or todao <'A' or len(todao)>1:
            todao = input('please Reinput way(A-X):')

        angle = int(input('please input angle(0-180)：'))
        while angle >180 or angle <0 :
            angle = int(input('please Reinput angle(0-180):'))

        if angle >=100:
            xixin = '$'+todao+str(angle)+'#'            #$A100#
        elif angle ==0:
            xixin = '$'+todao+'00'+str(angle)+'#'       #$A000#
        else:
            xixin = '$'+todao+'0'+str(angle)+'#'        #$A010#
        
        xixin = bytes(xixin,'utf-8')
        ser.write(xixin)

except KeyboardInterrupt:
    if ser != None:
        ser.close()
