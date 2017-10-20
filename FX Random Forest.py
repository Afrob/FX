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

nAlpha=12   #Anzahl Random Forest für CV bei der Optimierung auf Alpha
t=1000000   #Länge der Daten im Decision Tree (z.B. Anzahl Candles für 1 Monat)


M=pd.read_csv("USDZAR_prepared_Ind.csv")

N=len(M.index)
nForest=((N/t)-nAlpha+1)/1.2        #Anzahl Bäume im Random Forest
nTree=nForest+nAlpha-1              #Anzahl aller Bäume über gesamte Datenset
KForest=nForest*t/5                 #Größe Testset für Random Forest. Für Trees KTree=t


Tree={}
mse={}
LowLeaves=5
HighLeaves=1000


#Bilden der Bäume über alle Daten
for i in np.arange(0,nTree,1):
    Train=M[(N-KForest)-i*t-t:(N-KForest)-i*t]          #Trainingsset der Bäume
    Test=M[(N-KForest)-i*t:(N-KForest)-i*t+t+1]         #Testset der Bäume hat Größe t
    XTrain=Train[Train.columns[:len(Train.columns)-1]]  #Parameter des Trainigssets
    YTrain=Train[Train.columns[len(Train.columns)-1]]   #Zielwert des Treiningssets
    XTest=Test[Test.columns[:len(Test.columns)-1]]      #Parameter des Testsets
    YTest=Test[Test.columns[len(Test.columns)-1]]       #Zielwert des Testsets

        #Bilden der Bäume über verschiedene Anzahl an Leaves
        for depth in np.arange(LowLeaves,HighLeaves,1):
            DTree=DecisionTreeRegressor(criterion="mse", max_depth=depth)
            Tree[i,depth]=DTree.fit(XTrain,YTrain)
            YPredict=Tree[i,depth].predict(XTrain)
            mse[i,depth]=mean_squared_error(YPredict,YTest)

#Ermitteln des optimalen Lambda
lb=0                    #untere Grenze des Lambda
hb=0.00001                  #obere Grenze des Lambda
StepSize=0.000001        #erste Schrittgröße, in der das Lambda gesucht werden soll
ErrCVStar=math.inf      #unendlich
ErrCVCheck=math.inf     #unendlich
lamb=lb                 #Anfangswert des Lambda
hbneu=hb
epsilon=1
j=0
depthlamb={}
while j<3:
    #Durchgehen der einzelnen Lambdas und ermitteln der jeweiligen ErrCV
    while lamb <= hbneu:
        if epsilon==0:
            Rlamb=lamb
        else:
            # Um nicht in lokalem Minimun zu landen Random Nummer aus alter Range
            Rlamb=np.random.choice([lamb,random.choice(np.arange(lb,hb,StepSize))],p=[1-1/epsilon,1/epsilon])
        if Rlamb==lamb:
            lamb=lamb+StepSize

        ErrCV=0
        #Durchgehen der einzelnen Bäume mit dem aktuellen Lambda
        for i in np.arange(0,nForest,1):
            ErrAugStar= math.inf    #unendlich
            #Anzahl der Leaves mit dem geringsten ErrAug ermitteln
            for depth in np.arange(LowLeaves,HighLeaves,1):
                ErrAug=mse[i,depth]+Rlamb*depth
                #Auswahl des Baumes mit dem geringsten ErrAug
                if ErrAug < ErrAugStar:
                    ErrAugStar=ErrAug
                    mseTree=mse[i,depth]
                    depthlamb[i]=depth
                    #hier und in ErrCVStar schon Baumauswahl rein: direkt über Tree??? oder ilamb bzw. jlamb/ilambstar...?

            #Ermitteln des ErrCV für das jeweilige Lambda
            ErrCV=ErrCV+mseTree

        #Speichern des Lambda und der Bäume mit dem geringsten ErrCV
        if ErrCV<ErrCVStar:
            lambStar=Rlamb
            ErrCVStar=ErrCV
            depthstar=depthlamb

    #Zähler, ob eine weitere Nachkommastelle sich lohnt
    if ErrCVCheck - ErrCVStar < 0.00000001:
        j = j + 1
        ErrCVCheck=ErrCVStar
    else:
        j = 0

    #Anpassen der neuen Grenzen und der neuen StepSize
    lamb=max(0,lambStar-StepSize)
    hbneu=lambStar+StepSize
    StepSize = StepSize / 10
    Epsilon=Epsilon+1

#Ermitteln des optimalen alpha
lb=0                    #untere Grenze des Alpha
hb=0.1                  #obere Grenze des Alpha
StepSize=0.01           #erste Schrittgröße, in der das Alpha gesucht werden soll
ErrCVStar=math.inf      #unendlich
ErrCVCheck=math.inf     #unendlich
alpha=lb                #Anfangswert des Alpha
hbneu=hb
epsilon=1
j=0
depthlamb={}
while j<3:
    #Durchgehen der einzelnen Lambdas und ermitteln der jeweiligen ErrCV
    while alpha <= hbneu:
        if epsilon==0:
            Ralpha=alpha
        else:
            # Um nicht in lokalem Minimun zu landen Random Nummer aus alter Range
            Ralpha=np.random.choice([alpha,random.choice(np.arange(lb,hb,StepSize))],p=[1-1/epsilon,1/epsilon])
        if Ralpha==alpha:
            alpha=alpha+StepSize

        if alpha!=0:
            ErrCV=0
            #Durchgehen der einzelnen Random Forests mit dem jeweiligen Alpha
            i = 0
            while i <= nAlpha:
                Test = M[(N - KForest) - i*t:N - i * t + 1]         #Test=M[(N-KForest)-i*t:(N-KForest)-i*t+KForest+1]#Durchgehen der einzelnen Bäume des Random Forests zur Ermittlung des ErrCV
                XTest = Test[Test.columns[:len(Test.columns) - 1]]  # Parameter des Testsets
                YTest = Test[Test.columns[len(Test.columns) - 1]]   # Zielwert des Testsets
                YPredictSum=0
                alphaSum=0

                #Prediction des Testsets. Später die Maschine
                #In For-Schleife vorhersage aus den einzelnen Bäumen, um Alpha diskontiert
                #Endgültige Vorhersage mittels YPredictAlpha
                j = 0
                while j <= nForest:
                    YPredict = Alpha**j*Tree[j+i, depthstar[j+i]].predict(XTest) #Vorhersage aus Baum um Alpha diskontiert
                    YPredictSum=YPredictSum+YPredict
                    alphaSum=alphaSum+Alpha**j
                    j = j+1

                YPredictAlpha=YPredictSum/alphaSum #Endgültige Vorhersage

                #Ermitteln des ErrCV für das jeweilige Alpha
                ErrCV=ErrCV+mean_squared_error(YPredictAlpha,YTest)
                i = i+1

            #Speichern des Alpha mit dem geringsten ErrCV
            if ErrCV<ErrCVStar:
                alphaStar=alpha
                ErrCVStar=ErrCV

    #Zähler, ob eine weitere Nachkommastelle sich lohnt
    if ErrCVCheck - ErrCVStar < 0.00000001:
        j = j + 1
        ErrCVCheck=ErrCVStar
    else:
        j = 0

    #Anpassen der neuen Grenzen und der neuen StepSize
    alpha=max(0,alphaStar-StepSize)
    hbneu=alphaStar+StepSize
    StepSize = StepSize / 10
    Epsilon=Epsilon+1

print(lambStar)
print(alphaStar)
print(ErrCVStar)
