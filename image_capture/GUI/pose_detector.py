import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image, PIL.ImageDraw
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import os
import numpy as np
#import paho.mqtt.client as mqtt

#This file is for trt_pose to 

'''
img is PIL format
'''
def draw_keypoints(key,edgeimg):
    thickness = 4
    #w, h = img.size
    w, h = edgeimg.size
    draw = PIL.ImageDraw.Draw(edgeimg)#img
    if len(key)>16:
        #draw Rankle -> RKnee (16-> 14)
        if all(key[16]) and all(key[14]):
            draw.line([ round(key[16][2] * w), round(key[16][1] * h), round(key[14][2] * w), round(key[14][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[16][2] * w)-1, round(key[16][1] * h), round(key[16][2] * w)+1, round(key[16][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[16][2] * w), round(key[16][1] * h)-1, round(key[16][2] * w), round(key[16][1] * h)+1],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[14][2] * w)-1, round(key[14][1] * h),round(key[14][2] * w)+1, round(key[14][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[14][2] * w), round(key[14][1] * h)-1,round(key[14][2] * w)+1, round(key[14][1] * h)+1],width = thickness*4, fill=(0,255,0))        
    if len(key)>14:
        #draw RKnee -> Rhip (14-> 12)
        if all(key[14]) and all(key[12]):
            draw.line([ round(key[14][2] * w), round(key[14][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[12][2] * w)-1, round(key[12][1] * h),round(key[12][2] * w)+1, round(key[12][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[12][2] * w), round(key[12][1] * h)-1,round(key[12][2] * w), round(key[12][1] * h)+1],width = thickness*4, fill=(0,255,0))
        #draw Rhip -> Lhip (12-> 11)
        # if all(key[12]) and all(key[11]):
        #     draw.line([ round(key[12][2] * w), round(key[12][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=(31,31,204))
        #draw Lhip -> Lknee (11-> 13)
    if len(key)>13:    
        if all(key[11]) and all(key[13]):
            draw.line([ round(key[11][2] * w), round(key[11][1] * h), round(key[13][2] * w), round(key[13][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[11][2] * w)-1, round(key[11][1] * h),round(key[11][2] * w)+1, round(key[11][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[11][2] * w), round(key[11][1] * h)-1,round(key[11][2] * w), round(key[11][1] * h)+1],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[13][2] * w)-1, round(key[13][1] * h),round(key[13][2] * w)+1, round(key[13][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[13][2] * w), round(key[13][1] * h)-1,round(key[13][2] * w), round(key[13][1] * h)+1],width = thickness*4, fill=(0,255,0))
    
    if len(key)>15:
        #draw Lknee -> Lankle (13-> 15)
        if all(key[13]) and all(key[15]):
            draw.line([ round(key[13][2] * w), round(key[13][1] * h), round(key[15][2] * w), round(key[15][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[15][2] * w)-1, round(key[15][1] * h),round(key[15][2] * w)+1, round(key[15][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[15][2] * w), round(key[15][1] * h)-1,round(key[15][2] * w), round(key[15][1] * h)+1],width = thickness*4, fill=(0,255,0))
    if len(key)>10:
        #draw Rwrist -> Relbow (10-> 8)
        if all(key[10]) and all(key[8]):
            draw.line([ round(key[10][2] * w), round(key[10][1] * h), round(key[8][2] * w), round(key[8][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[10][2] * w)-1, round(key[10][1] * h),round(key[10][2] * w)+1, round(key[10][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[10][2] * w), round(key[10][1] * h)-1,round(key[10][2] * w), round(key[10][1] * h)+1],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[8][2] * w)-1, round(key[8][1] * h),round(key[8][2] * w)+1, round(key[8][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[8][2] * w), round(key[8][1] * h)-1,round(key[8][2] * w), round(key[8][1] * h)+1],width = thickness*4, fill=(0,255,0))
    if len(key)>8:
        #draw Relbow -> Rshoulder (8-> 6)
        if all(key[8]) and all(key[6]):
            draw.line([ round(key[8][2] * w), round(key[8][1] * h), round(key[6][2] * w), round(key[6][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[6][2] * w)-1, round(key[6][1] * h),round(key[6][2] * w)+1, round(key[6][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[6][2] * w), round(key[6][1] * h)-1,round(key[6][2] * w), round(key[6][1] * h)+1],width = thickness*4, fill=(0,255,0))
    if len(key)>6:
        #draw Rshoulder -> Lshoulder (6-> 5)
        if all(key[6]) and all(key[5]):
            draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[5][2] * w), round(key[5][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[5][2] * w)-1, round(key[5][1] * h),round(key[5][2] * w)+1, round(key[5][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[5][2] * w), round(key[5][1] * h)-1,round(key[5][2] * w), round(key[5][1] * h)+1],width = thickness*4, fill=(0,255,0))
    if len(key)>7:
        #draw Lshoulder -> Lelbow (5-> 7)
        if all(key[5]) and all(key[7]):
            draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[7][2] * w), round(key[7][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[7][2] * w)-1, round(key[7][1] * h),round(key[7][2] * w)+1, round(key[7][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[7][2] * w), round(key[7][1] * h)-1,round(key[7][2] * w), round(key[7][1] * h)+1],width = thickness*4, fill=(0,255,0))
    if len(key)>9:
        #draw Lelbow -> Lwrist (7-> 9)
        if all(key[7]) and all(key[9]):
            draw.line([ round(key[7][2] * w), round(key[7][1] * h), round(key[9][2] * w), round(key[9][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[9][2] * w)-1, round(key[9][1] * h),round(key[9][2] * w)+1, round(key[9][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[9][2] * w), round(key[9][1] * h)-1,round(key[9][2] * w), round(key[9][1] * h)+1],width = thickness*4, fill=(0,255,0))
    if len(key)>12:
        #draw Rshoulder -> RHip (6-> 12)
        if all(key[6]) and all(key[12]):
            draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness//2, fill=(255,255,255))
    if len(key)>11:
        #draw Lshoulder -> LHip (5-> 11)
        if all(key[5]) and all(key[11]):
            draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness//2, fill=(255,255,255))


    #draw nose -> Reye (0-> 2)
    # if all(key[0][1:]) and all(key[2]):
    #     draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[2][2] * w), round(key[2][1] * h)],width = thickness, fill=(219,150,219))

    # #draw Reye -> Rear (2-> 4)
    # if all(key[2]) and all(key[4]):
    #     draw.line([ round(key[2][2] * w), round(key[2][1] * h), round(key[4][2] * w), round(key[4][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> Leye (0-> 1)
    # if all(key[0][1:]) and all(key[1]):
    #     draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[1][2] * w), round(key[1][1] * h)],width = thickness, fill=(219,0,219))
    
    # #draw Leye -> Lear (1-> 3)
    # if all(key[1]) and all(key[3]):
    #     draw.line([ round(key[1][2] * w), round(key[1][1] * h), round(key[3][2] * w), round(key[3][1] * h)],width = thickness, fill=(219,0,219))
    #print(key,all(key[0][1:]),all(key[17]))
    if len(key)>17:
        #draw nose -> neck (0-> 17)
        if all(key[0][1:]) and all(key[17]):
            draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[17][2] * w), round(key[17][1] * h)],width = thickness//2, fill=(255,255,255))
            draw.line([ round(key[0][2] * w)-1, round(key[0][1] * h),round(key[0][2] * w)+1, round(key[0][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[0][2] * w), round(key[0][1] * h)-1,round(key[0][2] * w), round(key[0][1] * h)+1],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[17][2] * w)-1, round(key[17][1] * h),round(key[17][2] * w)+1, round(key[17][1] * h)],width = thickness*4, fill=(0,255,0))
            draw.line([ round(key[17][2] * w), round(key[17][1] * h)-1,round(key[17][2] * w), round(key[17][1] * h)+1],width = thickness*4, fill=(0,255,0))
            
            #print(len(key),key)
            #draw hip extension 
    if len(key)>19:
            if all(key[18]) and all(key[19]):
                draw.line([ round(key[19][2] * w), round(key[19][1] * h), round(key[18][2] * w), round(key[18][1] * h)],width = thickness, fill=(250,50,10))
    if len(key)>21:
        #draw waist 
        if all(key[20]) and all(key[21]):
            draw.line([ round(key[21][2] * w), round(key[21][1] * h), round(key[20][2] * w), round(key[20][1] * h)],width = thickness, fill=(0,200,250))
    return edgeimg#img

'''
hnum: 0 based human index
kpoint : index + keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height)
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None'%(j) )
    return kpoint

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

'''
Draw to inference (small)image
'''
def execute(img):
    start = time.time()
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    end = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    for i in range(counts[0]):
        #print("Human index:%d "%( i ))
        get_keypoint(objects, i, peaks)
    #print("Human count:%d len:%d "%(counts[0], len(counts)))
    #print('===== Net FPS :%f ====='%( 1 / (end - start)))
    draw_objects(img, counts, objects, peaks)
    return img

'''
Draw to original image
'''
def execute_2(img, org):
    start = time.time()
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    end = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    keypoints =[]
    org_eq = org.filter(PIL.ImageFilter.GaussianBlur(radius = 3))
    arrimg=np.array(org_eq)#org
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(arrimg, cv2.COLOR_BGR2YCrCb)
    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    
    h,w,n=arrimg.shape
    #w,h=org_eq.size
    org_eq=PIL.Image.fromarray(equalized_img).copy()
    for i in range(counts[0]):
        #print("Human index:%d "%( i )) 
        kpoint = get_keypoint(objects, i, peaks)
        keypoints.append(kpoint)
    #add actual_hip key points   
    # print('allkp for hip:',(all(keypoints[0][11])) and (all(keypoints[0][12])))
    #if type(keypoints)=='list': 
    if len(keypoints)>0:
        if len(keypoints[0])>12:
            if (all(keypoints[0][11])) and (all(keypoints[0][12])):
                res=find_edge_points(org_eq,int(w*keypoints[0][12][2]),int(h*keypoints[0][12][1]),
                    int(w*keypoints[0][11][2]),int(h*keypoints[0][11][1]))
                if res:
                    lhpx,lhpy,rhpx,rhpy,flimgh = res
                    #add keypoints
                    keypoints[0].append((len(keypoints[0]),rhpy/h,rhpx/w))
                    keypoints[0].append((len(keypoints[0]),lhpy/h,lhpx/w))
            #org_eq=PIL.Image.fromarray(equalized_img).copy()
            if all(keypoints[0][5]) and all(keypoints[0][6]) and all(keypoints[0][12]) and all(keypoints[0][11]):
                #Adding waist keypoints using shoulder info:
                #The intiial keypoint has x from hip and y from 1/3 of the way to the shoulder
                #try with 3 positions and choose the smaller
                wx0=int(w*keypoints[0][12][2]+(0.02*w))
                wx1=int(w*keypoints[0][11][2]-(0.02*w))
                wy0=int(h*keypoints[0][12][1]+(h/3)*(keypoints[0][6][1]-keypoints[0][12][1]))
                wy1=int(h*keypoints[0][11][1]+(h/3)*(keypoints[0][5][1]-keypoints[0][11][1]))
                res= find_edge_points(org_eq,wx0,wy0,wx1,wy1,90)
                if res:
                    lwpx1,lwpy1,rwpx1,rwpy1,flimgw1=res
                    width1=rwpx1-lwpx1
                    #try2:
                    wy0=int(h*keypoints[0][12][1]+(h/2)*(keypoints[0][6][1]-keypoints[0][12][1]))
                    wy1=int(h*keypoints[0][11][1]+(h/2)*(keypoints[0][5][1]-keypoints[0][11][1]))
                    lwpx2,lwpy2,rwpx2,rwpy2,flimgw2 = find_edge_points(org_eq,wx0,wy0,wx1,wy1,110)
                    width2=rwpx2-lwpx2
                    #try3:
                    wy0=int(h*keypoints[0][12][1]+(h/4)*(keypoints[0][6][1]-keypoints[0][12][1]))
                    wy1=int(h*keypoints[0][11][1]+(h/4)*(keypoints[0][5][1]-keypoints[0][11][1]))
                    lwpx3,lwpy3,rwpx3,rwpy3,flimgw3 = find_edge_points(org_eq,wx0,wy0,wx1,wy1,110)
                    width3=rwpx3-lwpx3
                    pmin=np.argmin([width1,width2,width3])
                    wimg=[flimgw1,flimgw2,flimgw3]
                    wimg=wimg[pmin]
                    poslist=[[lwpx1,lwpy1,rwpx1,rwpy1],[lwpx2,lwpy2,rwpx2,rwpy2],[lwpx3,lwpy3,rwpx3,rwpy3]]
                    lwpx,lwpy,rwpx,rwpy=poslist[pmin]
                    #add keypoints
                    keypoints[0].append((len(keypoints[0]),rwpy/h,rwpx/w))
                    keypoints[0].append((len(keypoints[0]),lwpy/h,lwpx/w))
    #     print('torgg before:',type(org))
    for kpoint in keypoints:      
        org = draw_keypoints(kpoint,org)#wimg)#org
    return keypoints,org

def cut_gaps_r(trace,gapsize):
    #Crops the last part of a trace (right) if there ar emore than gapsize points above 0 after the midle point of the trace
    lt=len(trace)
    #select the 2nd half of the trace
    toclip=trace[lt//2:]
    #making it a direct multiple of gapsize
    clipped=toclip[0:gapsize*(len(toclip)//gapsize)]
    #reshape in gapsize groups
    #print('reshaping:',len(clipped),gapsize,len(toclip)//gapsize)
    tocrop=clipped.reshape(len(toclip)//gapsize,gapsize)
    #Check if any group is all>0
    arrm=np.min(tocrop,axis=1)
    pcut=np.where(arrm>0)[0]
    if len(pcut)>0:
        posc=pcut[0]
        return trace[0:lt//2+(posc*gapsize)]
    else:
        return trace
def cut_gaps_l(trace,gapsize):
    #Crops the first part of a trace (left) if there are more than gapsize points above 0 before the midle point of the trace
    lt=len(trace)
    #select the 1st half of the trace
    toclip=trace[0:lt//2]
    #making it a direct multiple of gapsize
    clipped=toclip[len(toclip)-gapsize*((lt//2)//gapsize):]
    #reshape in gapsize groups
    tocrop=clipped.reshape(((lt//2)//gapsize),gapsize)
    #Check if any group is all>0
    arrm=np.min(tocrop,axis=1)
    pcut=np.where(arrm>0)[0]
    if len(pcut)>0:
        posc=pcut[-1]
        return trace[posc*gapsize:]
    else:
        return trace
    
def find_edge_points(img,x0,y0,x1,y1,tol=100):
    #finds the coordinates at the edge of the person in the image from the left and right along the line that connects the 0 and 1 points
    #the image may have histogram equalization. It should be a  PIL image. The input coordinates must be inside the person's region
    #uses flood-filling to find the edges
    #Select points above and below the original points for flood-filling in black, displaced inwards or outwards according to dirn ('in' or 'out)
    #Output: lx,ly,rx,ry
    w,h=img.size
    wd=int(0.02*w)
    hd=int(0.015*h)
    # if dirn=='in':
    #     wd*=(-1)
    #print('pixels for x:',wd,' pixels for y:',hd)
    # seedLup = x0-wd,y0+hd
    # seedRup = x1+wd,y1+hd
    # seedLdown = x0-wd,y0-hd
    # seedRdown = x1+wd,y1-hd
    seedLup = x0,y0+hd
    seedRup = x1,y1+hd
    seedLdown = x0,y0-hd
    seedRdown = x1,y1-hd
    # Pixel Value which would be used for replacement 
    rep_value = (0, 0, 0)
    # Calling the floodfill() function and passing the image
    aux0=img.copy()
    out=PIL.ImageDraw.floodfill(aux0, seedLup, rep_value, thresh=tol)
    out=PIL.ImageDraw.floodfill(aux0, seedRup, rep_value, thresh=tol)
    out=PIL.ImageDraw.floodfill(aux0, seedLdown, rep_value, thresh=tol)
    out=PIL.ImageDraw.floodfill(aux0, seedRdown, rep_value, thresh=tol)
    #aux=img.copy()
    aux=np.array(aux0)
    m = (y0-y1)/(x0-x1)
    n = y0-(m*x0)
    #Right limit
    x=np.arange(x1,int(np.min([x1+3*(x1-x0),w])))
    y=(m*x+n).astype(int)
    figline=np.zeros(x.shape)
    for i in range(len(x)):
        figline[i]=np.mean(aux[y[i],x[i],:])
    #if there are wd continous pixels>0 after the halfpoint, crop figline to the first one
    figline=cut_gaps_r(figline,wd)
    #The edge of the body should be at the last 0 in figline (axis=-1)
    #print('right:',figline)
    if len(figline)>10:
        pos=np.argmin(figline,axis=-1)
        x=x[0:len(figline)]
        y=y[0:len(figline)]
        if figline[pos]>0 or pos==0:
            #If not found, assign a half-way point value
            pos=len(x)//2
            #print('guessing pos')
        #output coordinates:
        rx=x[pos]
        ry=y[pos]
    
    #Left limit
    # x=np.arange(int(np.max([0,(2*x0)-x1])),x0)
    x=np.arange(int(np.max([0,x0-1.5*(x1-x0)])),x0)
    y=(m*x+n).astype(int)
    figline=np.zeros(x.shape)
    for i in range(len(x)):
        figline[i]=np.mean(aux[y[i],x[i],:])
    figline=cut_gaps_l(figline,wd)
    if len(figline)>10:
        #Adjusting the x and y vectors to match figline if the first points were cropped
        x=x[len(x)-len(figline):]
        y=y[len(y)-len(figline):]
        pos=np.argmin(figline)
        #print('left:',figline)
        #This should not have zeros at the beggining if there are gaps before the mid point
        if figline[pos]>0 or pos==0:
            pos=len(x)//2
            print('guessing pos')
        lx=x[pos]
        ly=y[pos]
        return lx,ly,rx,ry,aux0
    else:
        return None

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
#t1 = time.time()

#print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def detect_pose(body_img):
    #try:
    # if type(body_img)=='numpy.ndarray':
    #     image=body_img.copy()
    image=np.array(body_img) #array image
    body_img=PIL.Image.fromarray(body_img)
    # else:
        
    #     print(type(body_img))
    #pilimg = PIL.Image.fromarray(body_img)
    orgimg = body_img.copy() #PIL image
    image = cv2.resize(image, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    #img = image.copy()
    #img = execute(img)
    keypoints, processed_img = execute_2(image, orgimg)
    return (orgimg, keypoints, processed_img)
    # except:
    #     None

# dir, filename = os.path.split(args.image)
# name, ext = os.path.splitext(filename)
# pilimg.save('/home/lindayang/Desktop/mids/W251---Final-project-Weight-watchers/imageProcessor/%s_%s.png'%(args.model, name))


