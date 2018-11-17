
##################################################################################
#in this part of code we will our model on training data and get  confusion matrix 
# and accuracy of classification for each eta
##################################################################################

##########################
#importing needed packages
##########################
import numpy as np
import os                   
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#############################################################################
#importing weight vectors that we got when we applied our perceptron function
#on training data
#############################################################################

# load weight vectors for each classfier 
wpath='E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 1/weight vectors'
os.chdir(wpath)
wtest0=np.loadtxt('w0.txt')
wtest1=np.loadtxt('w1.txt')
wtest2=np.loadtxt('w2.txt')
wtest3=np.loadtxt('w3.txt')
wtest4=np.loadtxt('w4.txt')
wtest5=np.loadtxt('w5.txt')
wtest6=np.loadtxt('w6.txt')
wtest7=np.loadtxt('w7.txt')
wtest8=np.loadtxt('w8.txt')
wtest9=np.loadtxt('w9.txt')

#append all weight vecotrs in one vector
wtotal=[]
wtotal.append(wtest0)
wtotal.append(wtest1)
wtotal.append(wtest2)
wtotal.append(wtest3)
wtotal.append(wtest4)
wtotal.append(wtest5)
wtotal.append(wtest6)
wtotal.append(wtest7)
wtotal.append(wtest8)
wtotal.append(wtest9)

#################
#load test images
#################
Test_Path='E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 1/Assignment 1 Dataset/Test'
os.chdir(Test_Path)
Testing_Labels = np.loadtxt('Test Labels.txt')
filestest=os.listdir(Test_Path)
filestest.pop()
filestest = sorted(filestest,key=lambda x: int(os.path.splitext(x)[0]))

all_test=[]
for i in filestest:    
    imgtest=misc.imread(i)
    imgtest.shape
    #change dimention to 1 dimentional array instead of (28x28)
    imgtest=imgtest.reshape(784,)
    imgtest=np.append(imgtest,1)
    all_test.append(imgtest)

##########################
#getting y for different w
##########################
# calculate y with different w and different eta
yw=[]
for etaloop in range(10):
    for points in range(200):
        for wloop in range(10):
            yeta = np.dot(np.array([wtotal[wloop][etaloop]]),all_test[points])
            yw=np.append(yw,yeta)

# this array has 2000 row each row has 10 valyes of y for same point and same eta
#for first 200 lines all 200 points are represented with eta=1 then next 200 lines
# are presented with eta=10**-1 ...etc
ytotal=yw.reshape(2000,10)

# here we get the classification 2000 values first 200 are for all images when eta=1
# then next 200 images for eta=10**-1
classify=[]
for row in range(2000):
    maxindex=np.argmax(ytotal[row])
    classify=np.append(classify,maxindex)

###########################
#Getting confusion matrices
##########################

# Confusion matrix for eta=1
conf_eta0=confusion_matrix(classify[0:200],Testing_Labels)
   
# Confusion matrix for eta=10**-1
conf_eta1=confusion_matrix(classify[200:400],Testing_Labels)
  
# Confusion matrix for eta=10**-2
conf_eta2=confusion_matrix(classify[400:600],Testing_Labels)

# Confusion matrix for eta=10**-3
conf_eta3=confusion_matrix(classify[600:800],Testing_Labels)
  
# Confusion matrix for eta=10**-4
conf_eta4=confusion_matrix(classify[800:1000],Testing_Labels)

# Confusion matrix for eta=10**-5
conf_eta5=confusion_matrix(classify[1000:1200],Testing_Labels)

# Confusion matrix for eta=10**-6
conf_eta6=confusion_matrix(classify[1200:1400],Testing_Labels)

# Confusion matrix for eta=10**-7
conf_eta7=confusion_matrix(classify[1400:1600],Testing_Labels)

# Confusion matrix for eta=10**-8
conf_eta8=confusion_matrix(classify[1600:1800],Testing_Labels)
 
# Confusion matrix for eta=10**-9
conf_eta9=confusion_matrix(classify[1800:2000],Testing_Labels)


#####################################
#Getting accuracy of confusion matrix
#####################################

#This function calcualtes accuracy of confusion matrix

def confusion_matrix_accuracy(conf_matrix):
    diag_sum=0
    
    for diagonal in range(10) :
        diag_sum+=conf_matrix[diagonal][diagonal]
    accuracy=(diag_sum/200)*100
    return accuracy

#Ouputs accuracy of each confusion matrix by calling the above accuracy function
print(confusion_matrix_accuracy(conf_eta0))   
print(confusion_matrix_accuracy(conf_eta1)) 
print(confusion_matrix_accuracy(conf_eta2))
print(confusion_matrix_accuracy(conf_eta3))   
print(confusion_matrix_accuracy(conf_eta4)) 
print(confusion_matrix_accuracy(conf_eta5))
print(confusion_matrix_accuracy(conf_eta6))
print(confusion_matrix_accuracy(conf_eta7))   
print(confusion_matrix_accuracy(conf_eta8)) 
print(confusion_matrix_accuracy(conf_eta9))

##########################################
#Saving confusion matrices into jpg images
##########################################

Deliverables='E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 1/deliverables'
os.chdir(Deliverables)

#fuction used to save confusion matrix as jpg image
def plot_confusion_matrix(conf_matrix, matrix_name='confusionmatrix.jpg'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax=ax.matshow(conf_matrix,cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.savefig(matrix_name)

#save the ten confusion matrices
plot_confusion_matrix(conf_matrix=conf_eta0,matrix_name='Confusion_0.jpg')
plot_confusion_matrix(conf_matrix=conf_eta1,matrix_name='Confusion_1.jpg')
plot_confusion_matrix(conf_matrix=conf_eta2,matrix_name='Confusion_2.jpg')
plot_confusion_matrix(conf_matrix=conf_eta3,matrix_name='Confusion_3.jpg')
plot_confusion_matrix(conf_matrix=conf_eta4,matrix_name='Confusion_4.jpg')
plot_confusion_matrix(conf_matrix=conf_eta5,matrix_name='Confusion_5.jpg')
plot_confusion_matrix(conf_matrix=conf_eta6,matrix_name='Confusion_6.jpg')
plot_confusion_matrix(conf_matrix=conf_eta7,matrix_name='Confusion_7.jpg')
plot_confusion_matrix(conf_matrix=conf_eta8,matrix_name='Confusion_8.jpg')
plot_confusion_matrix(conf_matrix=conf_eta9,matrix_name='Confusion_9.jpg')

