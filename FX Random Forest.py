import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
import copy
from IPython.display import display, Image
import pydotplus
import math
import random
import time

starttime=time.time()

nAlpha=20   #Anzahl Random Forest für CV bei der Optimierung auf Alpha
t=1432   #Länge der Daten im Decision Tree (z.B. Anzahl Candles für 1 Monat)


M=pd.read_csv("USDZAR-XAUUSD3M_prepared_Ind.csv",index_col='Date')

N=len(M.index)
nForest=math.floor(((N/t)-nAlpha+1)/1.2 )        #Anzahl Bäume im Random Forest
nTree=nForest+nAlpha-1                          #Anzahl aller Bäume über gesamte Datenset
KForest=math.floor(nForest*t/5)                  #Größe Testset für Random Forest. Für Trees KTree=t


Tree={}
mse={}
ErrIn={}
maxdepth={}
LowLeaves=1
HighLeaves=10000


#Bilden der Bäume über alle Daten
i=0
while i<nTree:
    print(i, '/',nTree)
    Train=M[(N-KForest)-i*t-t:(N-KForest)-i*t]          #Trainingsset der Bäume
    Test=M[(N-KForest)-i*t:(N-KForest)-i*t+t+1]         #Testset der Bäume hat Größe t
    XTrain=Train[Train.columns[:len(Train.columns)-1]]  #Parameter des Trainigssets
    YTrain=Train[Train.columns[len(Train.columns)-1]]   #Zielwert des Treiningssets
    XTest=Test[Test.columns[:len(Test.columns)-1]]      #Parameter des Testsets
    YTest=Test[Test.columns[len(Test.columns)-1]]       #Zielwert des Testsets

    #Bilden der Bäume über verschiedene Anzahl an Leaves
    depth=LowLeaves
    j=0
    while depth <= HighLeaves:
        print(i, '/',nTree,',',depth,'/',HighLeaves)
        DTree=DecisionTreeRegressor(criterion="mse", max_depth=depth)
        #DTree = DecisionTreeRegressor(criterion="mse", max_leaf_nodes=depth)
        Tree[i,depth]=DTree.fit(XTrain,YTrain)
        YPredict=Tree[i,depth].predict(XTest)
        mse[i,depth]=mean_squared_error(YPredict,YTest)
        YInPredict=Tree[i,depth].predict(XTrain)
        ErrIn[i,depth]=mean_squared_error(YInPredict,YTrain)
        if depth>LowLeaves:
            if Tree[i,depth].tree_.node_count==Tree[i,depth-1].tree_.node_count:
                j=j+1
        if j==3:
            maxdepth[i]=depth
            depth=HighLeaves
        depth=depth+1
    i=i+1

#Ermitteln des optimalen Lambda
lb=0                    #untere Grenze des Lambda
hb=100                  #obere Grenze des Lambda
StepSize=1              #erste Schrittgröße, in der das Lambda gesucht werden soll
ErrCVStar=math.inf      #unendlich
ErrCVCheck=math.inf     #unendlich
lamb=lb                 #Anfangswert des Lambda
hbneu=hb
epsilon=1
j=0
depthlamb={}
while j<=3:
    #Durchgehen der einzelnen Lambdas und ermitteln der jeweiligen ErrCV
    while lamb <= hbneu:
        print(lamb)
        if epsilon==1:
            Rlamb=lamb
        else:
            # Um nicht in lokalem Minimun zu landen Random Nummer aus alter Range
            Rlamb=np.random.choice([lamb,random.uniform(lb,hb)],p=[1-1/epsilon,1/epsilon])
        if Rlamb==lamb:
            lamb=lamb+StepSize

        ErrCV=0
        #Durchgehen der einzelnen Bäume mit dem aktuellen Lambda
        i=0
        while i<nForest:
            print('Lambda:',Rlamb,'/',hbneu,' by StepSize ',StepSize,',',i,'/',nForest)
            ErrAugStar= math.inf    #unendlich
            #Anzahl der Leaves mit dem geringsten ErrAug ermitteln
            depth=LowLeaves
            while depth <=maxdepth[i]:
                ErrAug=ErrIn[i,depth]+Rlamb*depth
                #Auswahl des Baumes mit dem geringsten ErrAug
                if ErrAug < ErrAugStar:
                    ErrAugStar=ErrAug
                    mseTree=mse[i,depth]
                    depthlamb[i]=depth
                depth = depth + 1

            #Ermitteln des ErrCV für das jeweilige Lambda
            ErrCV=ErrCV+mseTree
            i=i+1

        #Speichern des Lambda und der Bäume mit dem geringsten ErrCV
        if ErrCV/nForest<ErrCVStar:
            lambStar=Rlamb
            ErrCVStar=ErrCV/nForest
            depthstar=depthlamb

    #Zähler, ob eine weitere Nachkommastelle sich lohnt
    if ErrCVCheck - ErrCVStar < 0.1:
        j = j + 1
    else:
        j = 0
    ErrCVCheck=ErrCVStar

    #Anpassen der neuen Grenzen und der neuen StepSize
    lamb=max(0,lambStar-StepSize)
    hbneu=lambStar+StepSize
    StepSize = StepSize / 10
    epsilon=epsilon+1

#Depth für restliche Bäume aus nAlpha ermitteln
i=nForest
while i < nTree:
    print('Lambda:', lambStar, ',', i, '/', nTree)
    ErrAugStar = math.inf  # unendlich
    # Anzahl der Leaves mit dem geringsten ErrAug ermitteln
    depth = LowLeaves
    while depth <= maxdepth[i]:
        ErrAug = ErrIn[i, depth] + lambStar * depth
        # Auswahl des Baumes mit dem geringsten ErrAug
        if ErrAug < ErrAugStar:
            ErrAugStar = ErrAug
            depthstar[i] = depth

        depth = depth + 1
    i=i+1


#Ermitteln des optimalen alpha
lb=0                    #untere Grenze des Alpha
hb=1                 #obere Grenze des Alpha
StepSize=0.01           #erste Schrittgröße, in der das Alpha gesucht werden soll
ErrCVStar=math.inf      #unendlich
ErrCVCheck=math.inf     #unendlich
alpha=lb                #Anfangswert des Alpha
hbneu=hb
epsilon=1
k=0
while k<=3:
    #Durchgehen der einzelnen Lambdas und ermitteln der jeweiligen ErrCV
    while alpha <= hbneu:
        print(alpha)
        if epsilon==1:
            Ralpha=alpha
        else:
            # Um nicht in lokalem Minimun zu landen Random Nummer aus alter Range
            Ralpha=np.random.choice([alpha,random.uniform(lb,hb)],p=[1-1/epsilon,1/epsilon])
        if Ralpha==alpha:
            alpha=alpha+StepSize

        if alpha!=0:
            ErrCV=0
            #Durchgehen der einzelnen Random Forests mit dem jeweiligen Alpha
            i=0
            while i<nAlpha:
                print('Alpha:',Ralpha,'/',hbneu,' by StepSize ',StepSize,',',i,'/',nAlpha)
                Test = M[(N - KForest) - i*t:N - i * t + 1]         #Test=M[(N-KForest)-i*t:(N-KForest)-i*t+KForest+1]#Durchgehen der einzelnen Bäume des Random Forests zur Ermittlung des ErrCV
                XTest = Test[Test.columns[:len(Test.columns) - 1]]  # Parameter des Testsets
                YTest = Test[Test.columns[len(Test.columns) - 1]]   # Zielwert des Testsets
                YPredictSum=0
                alphaSum=0

                #Prediction des Testsets. Später die Maschine
                #In For-Schleife vorhersage aus den einzelnen Bäumen, um Alpha diskontiert
                #Endgültige Vorhersage mittels YPredictAlpha
                j=0
                while j<nForest:
                    print('Alpha:',Ralpha,'/',hbneu,' by StepSize ',StepSize,',',i,'/',nAlpha,',',j,'/',nForest)
                    YPredict = alpha**j*Tree[j+i, depthstar[j+i]].predict(XTest) #Vorhersage aus Baum um Alpha diskontiert
                    YPredictSum=YPredictSum+YPredict
                    alphaSum=alphaSum+alpha**j
                    j=j+1

                YPredictAlpha=YPredictSum/alphaSum #Endgültige Vorhersage

                #Ermitteln des ErrCV für das jeweilige Alpha
                ErrCV=ErrCV+mean_squared_error(YPredictAlpha,YTest)
                i=i+1

            #Speichern des Alpha mit dem geringsten ErrCV
            if ErrCV/nAlpha<ErrCVStar:
                alphaStar=alpha
                ErrCVStar=ErrCV/nAlpha

    #Zähler, ob eine weitere Nachkommastelle sich lohnt
    if ErrCVCheck - ErrCVStar < 0.1:
        k = k + 1
    else:
        k = 0
    ErrCVCheck = ErrCVStar

    #Anpassen der neuen Grenzen und der neuen StepSize
    alpha=max(0,alphaStar-StepSize)
    hbneu=alphaStar+StepSize
    StepSize = StepSize / 10
    epsilon=epsilon+1

i=0
while i<nTree:
    dot_data = tree.export_graphviz(Tree[i, depthstar[i]], out_file = None, filled   = True, rounded  = True, special_characters = True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('FX Random Forest Tree {}.pdf'.format(i))
    i=i+1
print('LambdaStar: ',lambStar)
print('Alpha Star: ',alphaStar)
print('ErrCV Star:', ErrCVStar)
print('Depth: ',depthstar)
print('ErrIn:',ErrIn)
print('MSE:', mse)
print(time.time()-starttime)