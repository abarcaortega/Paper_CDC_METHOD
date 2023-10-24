import matplotlib.pyplot as plt
import numpy as np
import glob, os
from skimage.measure import profile_line
from scipy.optimize import curve_fit
from skimage import io
from matplotlib.backend_bases import FigureCanvasBase
from simple_colors import *
from PIL import Image
import cv2


def nam (type):
    cwd = os.getcwd()

    if type == "dir":
        tp = "/exp_*/"
        l = -2
    if type == "txt":
        tp = "/*.txt"
        l = -1
    s = glob.glob(cwd+tp)
    names = list()

    for i in range(len(s)):
        names.append((s[i].split("\\"))[l])

    return names

def status():

    cwd = os.getcwd()
    nams = nam("dir")
    
    try:
        os.mkdir("Results")
    except:
        pass

    os.chdir("Results")
    r = nam("txt")
    rl = list()

    nams = ['res_' + s for s in nams]
    for i in range(len(r)):
        rl.append((r[i].split(".txt"))[0])

    tot = 0
    nlist = list()

    for i in range(len(nams)):

        if nams[i] != "Results":
            nlist.append(nams[i])

        if any(x == nams[i] for x in rl):
            nlist.remove(nams[i])
            tot += 1

    nxt = nlist[0]
    msg = "Analisys complete "+str(tot)+" of "+str(len(nams))+". Next analysis: "+nxt

    print(msg)
    os.chdir(cwd)

    return nxt,len(nlist)
    
def yes_or_no(question):
    hx = 1
    while hx == 1:
        answer = input(question + ' (y/n): ').lower().strip()
        if answer in ('y', 'yes', 'n', 'no'):
            if answer in ('y', 'yes'):
              hhhh = 1
              hx = 0
              return hhhh
            if answer in ('n', 'no'):
               hhhh = 0
               hx = 0
               return hhhh
            else:
               print('You must answer yes or no.')
               
def coord(name,scale):
    im = cv2.imread(name)
    large = cv2.resize(im, (0,0), fx=scale, fy=scale) 

    r = cv2.selectROI(large)
    rr = np.zeros(len(r))
    rr[0] = round(r[0]/scale)
    rr[1] = round(r[1]/scale)
    rr[2] = round(r[2]/scale)
    rr[3] = round(r[3]/scale)

    cv2.destroyAllWindows()
    return rr

def rotate_image(image, angle):
  
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  
  return result

def crop(name,rr,rotate,ang_ent):
    
    im = cv2.imread(name)
    imCrop = im[int(rr[1]):int(rr[1]+rr[3]), int(rr[0]):int(rr[0]+rr[2])]
    ang = 0
    dthet = 0
    ang_1 = np.zeros(2)
    hi = 0
    
    try:
                
        while rotate == 0:
            
            if hi == 0:
                imCrop = rotate_image(im[int(rr[1]):int(rr[1]+rr[3]), int(rr[0]):int(rr[0]+rr[2])],ang_1[1])
            if hi == 1:
                imCrop = rotate_image(im[int(rr[1]):int(rr[1]+rr[3]), int(rr[0]):int(rr[0]+rr[2])],ang_1[1])
            
            cv2.imshow("Rotate", imCrop)
            cv2.waitKey(0)

            x = yes_or_no("Is the image well rotated? (current angle of " + str(ang_1[1]) + "): ")
            if x == 0:
                dthet = input("Insert an angle of rotation (counterclockwise direction): ")
                if dthet != 0:
                    ang_1[1] = ang_1[1] + float(dthet)
                    hi = 1  
            if x == 1:
                rotate = 1
                ang_ent = ang_1[1]
                break 
        
        if rotate != 0:
            #print(ang_ent)
            #print(ang_1[1])
            ang_1[1] = ang_ent
            imCrop = rotate_image(im[int(rr[1]):int(rr[1]+rr[3]), int(rr[0]):int(rr[0]+rr[2])],ang_1[1])
        
    except:
        pass        
        
    return imCrop,ang_1[1]

def all_files(ext):
    # subfolders = [ f.path for f in os.scandir() if f.is_dir() ]
    names = []
    # for i in range(len(subfolders)):
    #     os.chdir(subfolders[i])
    #     for file in glob.glob(ext):
    #         names.append(file)
    #     return names
    for file in glob.glob(ext):
        names.append(file)
    return names
    
def profile(img_name):
    # Load some image
    im = io.imread(img_name)
    im.shape
    #im = np.rot90(im)
    #im = np.flip(im)

    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # Extract intensity values along some profile line
    p = profile_line(im, ((round(im.shape[1]/2)),0), (round(im.shape[1]/2),im.shape[0]))
    #print(p)
    p = np.flip(p)
    #print(p)
    # Extract values from image directly for comparison
    i =im[0:(im.shape[0]+1), (round(im.shape[1]/2))]
    ## #print(i)
    ## plt.plot(p)
    ## plt.ylabel('intensity')
    ## plt.xlabel('line path')
    ## plt.show(block=False)
    ## plt.pause(0.1)
    ## plt.close()
    return p

def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def res_x(or_dir,jj,name):
    # or_dir = os.getcwd()
    names = all_files("Basler*")
    #print(names)
    crop_dir = "crop_d_"+name
    ii = np.zeros(len(names))
    pix = 1/0.775 #um/px en objetivo x4
    
    #os.chdir(crop_dir)
    
    crop_dir = "crop_res"
    try:
        os.mkdir(crop_dir)
    except:
        pass
    
    x = 0
    
    input("Enclose the cell in a box (Press ENTER) ")
    while x == 0:
        rr = coord(names[0],10)
        dia = (rr[3]+rr[2])*0.5*pix
        print("The cell diameter is "+str(dia)+" micrometers")
        x = yes_or_no("Is the cell diameter well select?: ")


## 
    input("Select in a rectangle the constriction width (Press ENTER) ")
## 
    x = 0
    while x!=1:
        ##os.chdir(crop_dir)
        rr = coord(names[0],10)
        # print(rr[0]-rr[2])
        wch = abs(rr[3])*pix
        # print(pix)
        print("The constriction width is "+str(wch)+" micrometers")
        x = yes_or_no("Is the constriction width well selected?")
        #os.chdir("..")

    bch = input("Enter the number of blocked channels: ")

    x = 0
    
    while x == 0:
        #pres_e = press()
        input("Select the inside of the pipette (Press ENTER)")
        rr = coord(names[0],10)
        a = []
        aa = []
        
        for i in range(len(names)):    
            
        
            imcrop,dthet = crop(names[i],rr, 1, 0)
            
            os.chdir(crop_dir)
            cv2.imwrite("res_"+names[i], cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY))               
            
            a = profile("res_"+names[i])
            #print(a)
            m = min(a)
            iih = [h for h, j in enumerate(a) if j == m]
            ii[i] = iih[0]
            os.chdir("..")

        ## aa = [-1*(i-ii[0])*pix/(float(pd)/2) for i in ii] #AL/RC
        aa = [-1*(i-ii[0])*pix for i in ii] #AL/RC
        
        jh = 0
        hz = 0.02 #time between frames in seconds
        xx = list(range(0,len(aa)))
        xx = [x*hz for x in xx]
        # print(xx)
        

        plt.scatter(xx, aa, alpha=0.4)
        plt.plot(xx, aa, 'r')
        plt.xlabel("Time [s]")
        plt.ylabel("Aspirated Length ($A_L$) [$\mu m$]")
        plt.show()
        
        x = yes_or_no("Is the analysis well done?: ")
    
    
    os.chdir(or_dir)
    
    try:
        os.mkdir("Results")
    except:
        pass 

    os.chdir("Results")

    jj=0
    
    with open(name+'.txt', 'w') as f:
        for item in aa:
            #save.append(pres_e[jj], item)
            f.write("%s %s\n" % (xx[jj] ,item))
            jj=jj+1
            
    with open("mec_res"+'.txt', 'a') as f:
        #save.append(pres_e[jj], item)
        f.write(name+ " %s %s %s\n" % (dia, wch, bch))
    #print(save)
    #print(or_dir)
    os.chdir(or_dir)
    return aa

def all_dir():
    or_dir = os.getcwd()
    subfolders = [ f.path for f in os.scandir(or_dir) if f.is_dir() ]
    subfolders = [ x for x in subfolders if "exp_" in x ]
    return subfolders


if __name__ == "__main__":
    
    or_dir = os.getcwd()
    subfolders = all_dir()
    i = 1
    
    for subf in subfolders:
        
        cont =1
        while cont == 1:
            try:
                nxt = status()
                #print(subf)
            except:
                input(" All the tests have been done!! ")
                cont = 0
                break
            
            name = nxt[0]
            #print(name)
            #print(subf)
            os.chdir('exp_'+name.split('_')[-1])
            res = res_x(or_dir,i,name)
            i=i+1
            cont = 0
            cont = yes_or_no("Do you wanna continue with the next analysis?")   
        
        if cont == 0:
            input(" #### See you later!! #### ")
            break
        
    os.chdir(or_dir) 
