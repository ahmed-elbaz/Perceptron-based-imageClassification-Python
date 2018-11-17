
###############################################################################
#In this code we design a Perceptron-based classification algorithm and we will
#use it to classify images for digits from 0 to 9
###############################################################################

##########################
#importing needed packages
##########################
import numpy as np
import os                   
from scipy import misc

#####################################################################################    
#perceptron function has 2 inputs the images after reshaping and lables of each image
#####################################################################################
def perceptron(points,labels):

    weta=[] # in this variable we save values of w with different values of eta
    
    # the below loop is to get w vector for each eta
    for eta in [1,10**-1,10**-2,10**-3,10**-4,10**-5,10**-6,10**-7,10**-8,10**-9]:
    
        winit=np.zeros(784)
        w=np.append(1,winit)
        iteration = 0
        mispoint=[0,0] #any initial value to start while loop
        
        while len(mispoint) != 0:
            iteration += 1
            y=[]
            t=[]
            # in this loop I classify points based on w
            for point in points:
                    y=np.dot(w,point)
                    if y >= 0:
                         t.append(1)
                    else:
                         t.append(-1)
                    
            # In this loop I compare trained labels to labels calculated by w0
            mispoint=[]
            for i in range(len(labels)):
                    if labels[i] != t[i]:
                        mispoint= points[i] #get misclassified point
                        tmispoint = labels[i] #get t(label) of the misclassified point 
                        break
            
            if len(mispoint) != 0:
                # Get the difference between current w and next w which equals to eta*phi(xn)*tn
                wdif = [j*eta*tmispoint for j in mispoint]
                
                # Get next w
                w=np.add(w,wdif)
        weta.append(w)    
    return weta

#####################
#load training images
#####################
Train_Path='E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 1/Assignment 1 Dataset/Train'
os.chdir(Train_Path)
Training_Labels = np.loadtxt('Training Labels.txt')
files=os.listdir(Train_Path)
files.pop()
files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))

all_data=[]
for i in files:    
    img=misc.imread(i)
    img.shape
    #change dimention to 1 dimentional array instead of (28x28)
    img=img.reshape(784,)
    img=np.append(img,1)
    all_data.append(img)

#######################################################
##Getting weight vector for 10 lines for different etas
#######################################################
    
# Changing labels to get class 0 decision boundary
label0 = [1 if label == 0 else -1 for label in Training_Labels]

# apply perceptron function to get w0 for different eta
w0=perceptron(all_data,label0) 

# Changing labels to get class 1 decision boundary
label1 = [1 if label == 1 else -1 for label in Training_Labels]

# apply perceptron function to get w1 for different eta
w1=perceptron(all_data,label1)

# Changing labels to get class 2 decision boundary
label2 = [1 if label == 2 else -1 for label in Training_Labels]

# apply perceptron function to get w2 for different eta
w2=perceptron(all_data,label2)

# Changing labels to get class 3 decision boundary
label3 = [1 if label == 3 else -1 for label in Training_Labels]

# apply perceptron function to get w3 for different eta
w3=perceptron(all_data,label3)

# Changing labels to get class 4 decision boundary
label4 = [1 if label == 4 else -1 for label in Training_Labels]

# apply perceptron function to get w4 for different eta
w4=perceptron(all_data,label4)

# Changing labels to get class 5 decision boundary
label5 = [1 if label == 5 else -1 for label in Training_Labels]

# apply perceptron function to get w5 for different eta
w5=perceptron(all_data,label5)

# Changing labels to get class 6 decision boundary
label6 = [1 if label == 6 else -1 for label in Training_Labels]

# apply perceptron function to get w6 for different eta
w6=perceptron(all_data,label6)
              
# Changing labels to get class 7 decision boundary
label7 = [1 if label == 7 else -1 for label in Training_Labels]

# apply perceptron function to get w7 for different eta
w7=perceptron(all_data,label7)

# Changing labels to get class 8 decision boundary
label8 = [1 if label == 8 else -1 for label in Training_Labels]

# apply perceptron function to get w8 for different eta
w8=perceptron(all_data,label8)

# Changing labels to get class 9 decision boundary
label9 = [1 if label == 9 else -1 for label in Training_Labels]

# apply perceptron function to get w9 for different eta
w9=perceptron(all_data,label9)

#Save all values of w in wtotal
wtotal=[]   
wtotal.append(w0)
wtotal.append(w1)
wtotal.append(w2)
wtotal.append(w3)
wtotal.append(w4)
wtotal.append(w5)
wtotal.append(w6)
wtotal.append(w7)
wtotal.append(w8)
wtotal.append(w9)

################################
#Save values of w in a text file
################################
wpath='E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 1/weight vectors'
os.chdir(wpath)
np.savetxt('w0.txt',w0)
np.savetxt('w1.txt',w1)
np.savetxt('w2.txt',w2)
np.savetxt('w3.txt',w3)
np.savetxt('w4.txt',w4)
np.savetxt('w5.txt',w5)
np.savetxt('w6.txt',w6)
np.savetxt('w7.txt',w7)
np.savetxt('w8.txt',w8)
np.savetxt('w9.txt',w9)

