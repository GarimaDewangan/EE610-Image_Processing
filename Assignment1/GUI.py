import PySimpleGUI as sg
import numpy as np
import os.path
import cv2 
import shutil
sg.theme('DarkAmber')

#Histogram Equalizer
def Equalize(vi):
    pdf = {} #PDF of the intensities in image
    cdf = {} #CDF of intensities in image
    hist = {} #Transformed(Equalized) mappping of intensities 
    (l,w) = vi.shape
    for i in range(0,256):
        pdf[i] = (vi==i).sum()/(l*w) #Calculating pdf of each intensity
        if(i==0):#Calculating cdf of each intensity
            cdf[0] = pdf[0]        
        else:
            cdf[i] = cdf[i-1]+pdf[i]
        hist[i] = int(cdf[i]*255)#updating hist with mapping of orignal and transformed intensity 
    for i in range(l):
        for j in range(w):
            vi[i][j] = hist[vi[i][j]]  #Updating intensity of pixels in images
    return(vi)

#Gamma Transform
def Gamma(vi,g): #vi - input image, g = gamma value
    tf = [] #Mapping of orignaland transformed intensity
    for i in range(256):
        tf.append(i**g)
    (l,w) = vi.shape
    M = max(tf)
    for i in range(256):
        tf[i] = int(tf[i]*255/M) #Normalising values
    for i in range(l):
        for j in range(w):
            vi[i][j] = tf[vi[i][j]] #Updating pixel intemsity values
    return(vi)

#log Transform
def Log(vi):
    tf = []
    for i in range(256):
        tf.append(np.log(1+i))
    (l,w) = vi.shape
    M = max(tf)
    for i in range(256):
        tf[i] = int(tf[i]*255/M)
    #print(tf)
    for i in range(l):
        for j in range(w):
            vi[i][j] = tf[vi[i][j]]
    return(vi)

#Blurring the Image
def blur(img,typ,fs) : #img - input image, typ - Blurring type(Mean or Guassian), fs - filter size(3,5,7,9)
    flt = np.ones((fs,fs)) #Mean filter
    r = int(fs/2)
    if(typ == 'Mean'):
        flt = flt/flt.sum()#Normalizing the mean filter
    if(typ == 'Guassian'):
        for i in range(fs):
            for j in range(fs):
                flt[i][j] = np.exp(-((i-r)**2 + (j-r)**2)) #Updating filter values correspondong to guassian filter
        flt = flt/flt.sum() #Normalising the filter
    (x,y) = img.shape
    img_p = img
    img_n = img.copy()
    img_p = np.pad(img_p, ((r,r),(r,r)), 'constant' , constant_values = ((0,0),(0,0)))#Zero padding
    for i in range(x): #Convolving Filter with the image
        for j in range(y):
            win=img_p[i:i+fs,j:j+fs]
            img_n[i,j] = int((np.multiply(win,flt).sum()))
    return(img_n)

#Sharpening the Image
def sharp(img,c):
    flt = np.ones((5,5)) #Mean filter
    r = 2;
    flt = flt/flt.sum() #Normalizing the mean filter
    (x,y) = img.shape
    img_p = img.copy()
    img_n = img.copy()
    img_p = np.pad(img_p, ((r,r),(r,r)), 'constant' , constant_values = ((0,0),(0,0)))#Zero padding
    for i in range(x):#Convolving Filter with the image to blur the image
        for j in range(y):
            win=img_p[i:i+5,j:j+5]
            img_n[i][j] = int((np.multiply(win,flt).sum()))
    img_s = np.subtract(img,img_n)#Subtracting blurred image from orignal image
    img_r = img
    for i in range(x):
        for j in range(y):
            img_r[i,j]= int(c*img_s[i,j]+ (1-c)*img[i,j]) #Adding difference to the orignal image
    return(img_r)

#Binarizing the Image
def binarize(img,thr): #img- input image, thr - threshold
    (x,y) = img.shape
    for i in range(x):
        for j in range(y):
            if(img[i,j]>=thr): #If greater than certain threshold pixel value is 255 else 0
                img[i,j]=255 
            else:
                img[i,j]=0
    return(img)

#Edge Detection using Sobel Operator
def edge(img,trv):
    flt = [[-1 ,0,1],[-1,0,1],[-1,0,1]] #Vertical Edge Detection filter
    r = 2
    (x,y) = img.shape
    img_p = img.copy()
    img_n = img.copy()
    img_p = np.pad(img_p, ((r,r),(r,r)), 'constant' , constant_values = ((0,0),(0,0)))
    for i in range(x):#Convolving Filter with the image to blur the image
        for j in range(y):
            win=img_p[i:i+3,j:j+3]
            img_n[i][j] = int((np.multiply(win,flt).sum()))
    flt = [[-1,-1,-1],[0,0,0],[1,1,1]] #Horizontal Edge Detection filter
    img_r = img.copy()
    for i in range(x):#Convolving Filter with the image to blur the image
        for j in range(y):
            win=img_p[i:i+3,j:j+3]
            img_r[i][j] = int((np.multiply(win,flt).sum()))
    for i in range(x):#Updating pixel values of image
        for j in range(y):
            if (int(np.sqrt(img_n[i,j]**2 + img_r[i,j]**2))>trv):
                img[i,j] = 0
            else:
                img[i,j] = 255 

    return(img)

#Defining the GUI Layout
Functions = [
    [   sg.Button('Histogram_Equalizer'),],
    [   sg.Button('Gamma_Correction'),],
    [   sg.Button('Log_Transform'),],
    [   sg.Button('Blur'),sg.Radio('Mean', "Blur_type" ,default = True, key='-Mean-'), sg.Radio('Guassian', "Blur_type", key='-Guassian-') ],
    [   sg.Text('Filter Size'), sg.Radio('3', "F_size" ,default = True,key='-3-'), sg.Radio('5', "F_size",key='-5-'), sg.Radio('7', "F_size",key='-7-'), sg.Radio('9', "F_size",key='-9-')],
    [   sg.Button("Sharpen"),sg.Text('Parameter'),sg.Slider(range=(0,1), default_value=0.1, resolution = 0.01, orientation = 'h', size = (50,5), key = '-parm-')],
    [   sg.Button("Binarize"),],
    [   sg.Text('Threshold'),sg.Slider(range=(0,255), default_value=128, orientation = 'h', size = (50,5), key = '-bin_t-')],
    [   sg.Button('Detect_Edges')],
    [sg.Button("UNDO"), sg.Button("RESET"),],
    [sg.Button("SAVE"),],
]

Image = [
    [sg.Text('Processed Image')],
    [sg.Image(filename='' , key='-IMG-',size=(100,100))],
]

layout = [
[   sg.Text('Upload Image'),sg.Input(size=(75,1)), sg.FileBrowse(key = '-INP-'), sg.Button('Select') ],
[   sg.Column(Functions),sg.VSeperator(),sg.Column(Image),],
]

window = sg.Window("Image Processing",layout, size = (1350,750))

stack = []#To store all the enhanced images in order.Useful if we wannna revert back to any of them(UNDO)

#Infinite loop which calls the required python function based on the option schosen and button pressed
while(True):
    event, values = window.read() #Reads the events , values pressed or entered in the window
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event=='Select':
        file = values['-INP-']
        org = cv2.imread(file)        
        if(org.shape[2]==3):
            col = True
            cur = cv2.cvtColor(org, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(cur)
            cur = v
        else:
            col = False
            cur = org 
            v = org 
        orignal = cur.copy() 
        prev = cur.copy()     
    elif event == 'Histogram_Equalizer':
        prev = cur.copy()
        cur = Equalize(cur)
    elif event == 'Gamma_Correction':
        g = sg.popup_get_text('Enter Value of Gamma')
        prev = cur.copy()
        cur = Gamma(cur,float(g))
    elif event== 'Log_Transform':
        prev = cur.copy()
        cur = Log(cur)
    elif event == 'Blur':
        prev = cur.copy()
        if values['-Mean-']:
            typ = 'Mean'
        elif values['-Guassian-']:
            typ = 'Guassian'
        if values['-3-']:
            fs = 3
        elif values['-5-']:
            fs = 5
        elif values['-7-']:
            fs = 7
        elif values['-9-']:
            fs = 9
        cur = blur(cur,typ,fs)
    elif event == 'Sharpen':
        prev = cur.copy()
        p = values['-parm-']
        cur = sharp(cur,p)
    elif event == 'Binarize':
        prev = cur.copy()
        t = int(values['-bin_t-'])
        cur = binarize(cur,t)
    elif event == 'Detect_Edges':
        trv = sg.popup_get_text('Enter Value of Threshold')
        prev = cur.copy()
        cur = edge(cur,int(trv))
    elif event =='RESET':
        prev = cur.copy()
        cur = orignal.copy() #Return the orignal image
    elif event =='UNDO':
        cur = stack.pop() #Pop th elast stored image on stack
    elif event =='SAVE':
        s_name = sg.popup_get_file('Please enter a file name', save_as=True)
        cv2.imwrite(s_name,im)

    st = prev.copy()
    stack.append(st)

    if(col): #If it is a colour image
        im = cv2.merge((h,s,cur)) #Merge the hue, saturation and value
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR) #Convery the image back to RGB
    byt = cv2.imencode(".png", im)[1].tobytes() 
    window['-IMG-'].update(data=byt) #Displaying on the window

window.close()