
########################################################################################
# In this part we will apply the weight vectors first in validation data to get the best
# eta for each classifier then we will apply the best classifiers to the test data again
# and find confusion matrix and accuracy
########################################################################################

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
w0=np.loadtxt('w0.txt')
w1=np.loadtxt('w1.txt')
w2=np.loadtxt('w2.txt')
w3=np.loadtxt('w3.txt')
w4=np.loadtxt('w4.txt')
w5=np.loadtxt('w5.txt')
w6=np.loadtxt('w6.txt')
w7=np.loadtxt('w7.txt')
w8=np.loadtxt('w8.txt')
w9=np.loadtxt('w9.txt')

#append all weight vecotrs in one vector
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


#######################
#load Validation images
#######################
Validation_Path='E:/NU BDDS-PD/Intro to Machine learning/Assignments/Assignment 1/Assignment 1 Dataset/Validation'
os.chdir(Validation_Path)
Validation_Labels = np.loadtxt('Validation Labels.txt')
filevalidation=os.listdir(Validation_Path)
filevalidation.pop()
filevalidation = sorted(filevalidation,key=lambda x: int(os.path.splitext(x)[0]))

all_validation=[]
for i in filevalidation:    
    imgvalidation=misc.imread(i)
    imgvalidation.shape
    #change dimention to 1 dimentional array instead of (28x28)
    imgvalidation=imgvalidation.reshape(784,)
    imgvalidation=np.append(imgvalidation,1)
    all_validation.append(imgvalidation)

#############################################
#getting y for different w in validation data
#############################################
# calculate y with different w and different eta
ywvalid=[]
for etaloop in range(10):
    for points in range(200):
        for wloop in range(10):
            yeta = np.dot(np.array([wtotal[wloop][etaloop]]),all_validation[points])
            ywvalid=np.append(ywvalid,yeta)

# this array has 2000 row each row has 10 valyes of y for same point and same eta
#for first 200 lines all 200 points are represented with eta=1 then next 200 lines
# are presented with eta=10**-1 ...etc
ytotal_valid=ywvalid.reshape(2000,10)

# here we get the classification 2000 values first 200 are for all images when eta=1
# then next 200 images for eta=10**-1
classify_valid=[]
for row in range(2000):
    maxindex_valid=np.argmax(ytotal_valid[row])
    classify_valid=np.append(classify_valid,maxindex_valid)
    
#################################################################
##Getting confusion matrices of validation data on different etas
#################################################################
    
confvalid_eta0=confusion_matrix(classify_valid[0:200],Validation_Labels)
confvalid_eta1=confusion_matrix(classify_valid[200:400],Validation_Labels)
confvalid_eta2=confusion_matrix(classify_valid[400:600],Validation_Labels)
confvalid_eta3=confusion_matrix(classify_valid[600:800],Validation_Labels)
confvalid_eta4=confusion_matrix(classify_valid[800:1000],Validation_Labels)
confvalid_eta5=confusion_matrix(classify_valid[1000:1200],Validation_Labels)
confvalid_eta6=confusion_matrix(classify_valid[1200:1400],Validation_Labels)
confvalid_eta7=confusion_matrix(classify_valid[1400:1600],Validation_Labels)
confvalid_eta8=confusion_matrix(classify_valid[1600:1800],Validation_Labels)
confvalid_eta9=confusion_matrix(classify_valid[1800:2000],Validation_Labels)

# put all confusion matrices in a list
confvalid_total=[]
confvalid_total.append(confvalid_eta0)
confvalid_total.append(confvalid_eta1)
confvalid_total.append(confvalid_eta2)
confvalid_total.append(confvalid_eta3)
confvalid_total.append(confvalid_eta4)
confvalid_total.append(confvalid_eta5)
confvalid_total.append(confvalid_eta6)
confvalid_total.append(confvalid_eta7)
confvalid_total.append(confvalid_eta8)
confvalid_total.append(confvalid_eta9)

#We will get diagonla values which represent accuracy in one list. first 10 values
#of the list shows accuracy of 0 classifier with different etas the next 10 values
#of list shows accuracy of 1 classifier with different etas and so on
acc = []
for index in range(10):
    for etaloop in range(10):
        accinit=confvalid_total[etaloop][index][index]
        acc.append(accinit)

# We will reshape the list to 10 rows and each row represents number of correclty 
# classified images for a certain class with the diffeent etas
acc =np.asarray(acc)
acc=acc.reshape(10,10)

# here we will get the best eta for every classifier in one list
best_eta=[]
for row in range(10):
    maxindex_eta=np.argmax(acc[row])
    best_eta=np.append(best_eta,maxindex_eta)

#print best eta for each classifier noting that the out put represents the absolute
# value of power of 10 of eta noting that there are some etas with the same number
# of correclty classified points and maybe this issue will not be faced in case of 
# greater number of imgaes in validation from our side when we have more than one eta 
# that can be selected as the best for one classifier we select the lowest eta as we 
# expect lower learning rate should get better accuracy
print(best_eta)


#wbest list includes w for each classifier with best eta performance when we applied
#the model in validation data 

wbest=[]
wbest.append(w0[9])
wbest.append(w1[6])
wbest.append(w2[5])
wbest.append(w3[9])
wbest.append(w4[7])
wbest.append(w5[9])
wbest.append(w6[7])
wbest.append(w7[9])
wbest.append(w8[9])
wbest.append(w9[9])

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
    
# now let's calculate y with different w
ybest=[]
for points in range(200):
        for wloop in range(10):
            yinitial = np.dot(np.array(wbest[wloop]),all_test[points])
            ybest=np.append(ybest,yinitial)

# reshpaing to have 10 ws for each point in one row then output the best class
ybest=ybest.reshape(200,10)

classify_best=[]
for row in range(200):
    maxindex_best=np.argmax(ybest[row])
    classify_best=np.append(classify_best,maxindex_best)

confbest=confusion_matrix(classify_best,Testing_Labels)

#print confusion matrix accuracy
def confusion_matrix_accuracy(conf_matrix):
    diag_sum=0
    
    for diagonal in range(10) :
        diag_sum+=conf_matrix[diagonal][diagonal]
    accuracy=(diag_sum/200)*100
    return accuracy

print(confusion_matrix_accuracy(confbest))

# save confusion matrix as image
def plot_confusion_matrix(conf_matrix, matrix_name='confusionmatrix.jpg'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax=ax.matshow(conf_matrix,cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.title('Accuracy = ' + str(confusion_matrix_accuracy(confbest))+'%')
    plt.savefig(matrix_name)
    
    
plot_confusion_matrix(conf_matrix=confbest,matrix_name='Confusion_b.jpg')


