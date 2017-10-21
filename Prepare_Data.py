import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pydotplus
import time
from time import strptime

starttime=time.time()
def prepare_igdata(symbol,start_date,end_date):
    Chead=["Date","Clock","Open","High","Low","Close","Volume"]
    df=pd.read_csv("{}.csv".format(symbol),sep=',',names=Chead,na_values=['nan'])
    df['Time']=df['Date'] + ' ' + df['Clock']
    df = df.set_index(['Time'])
    df=df[['Open','High','Low','Close','Volume']]
    dates=pd.date_range(start_date,end_date, freq="1min")
    M=pd.DataFrame(index=dates)
    Clock=M.index.hour+M.index.minute/60                #Convert clock string to float value
    M['Clock']=pd.Series(Clock,index=M.index)
    M=M.join(df)
    M=M.dropna()
    M.index.names=['Date']
    return M

#Gibt +1 zurück, wenn der Preis am oberen Bollinger Band ist und -1 am unteren Bollinger Band.
# Dazwischen bzw. außerhalb gibt es eine kontinuierliche Zahl aus.
def Bollinger_Value(M,wind):
    rm=M.rolling(window=wind).mean()
    std=M.rolling(window=wind).std()
    Bollinger_Value=(M-rm)/(2*std)
    Bollinger_Value=Bollinger_Value.to_frame(name='Bollinger Val')
    return Bollinger_Value

#Gibt einen Wert aus, wo man in der Range ist. +1 beim Hig und 0 beim Low
def HL_Value(M,wind):
    High=M.rolling(window=wind).max()
    Low=M.rolling(window=wind).min()
    HL_Val=(M-Low)/(High-Low)
    HL_Val=HL_Val.to_frame(name='HL Value')
    return HL_Val

def RSI (M, wind):
    Up, Down=M.copy(),M.copy()
    Up[Up<0]=0
    Down[Down>0]=0
    AvgUp=Up.rolling(window=wind).sum()/wind
    AvgDown=Down.rolling(window=wind).sum()*-1/wind
    RSI=100*AvgUp/(AvgUp+AvgDown)
    RSI.columns=['RSI']
    return RSI

start_date='2017-07-01 00:00'
end_date='2017-09-30 23:59'
prepared_igdata= prepare_igdata('USDZAR1Y',start_date,end_date)
prepared_igdata.to_csv("USDZAR1Y_prepared_IG.csv")
Bollinger_Val=Bollinger_Value(prepared_igdata['Close'],20)
HL_Val=HL_Value(prepared_igdata['Close'],1432)
rets=((prepared_igdata['Close']/prepared_igdata['Open'])-1)*10000
rets=rets.to_frame(name='Ret')
RSI_Val=RSI(rets,14)
Clock=prepared_igdata['Clock'].to_frame(name='Clock')
M=Clock.join(HL_Val)
M=M.join(Bollinger_Val)
M=M.join(RSI_Val)
M=M.join(rets)
M['Ret']=M['Ret'].shift(-1)
M=M[:-1]
M=M.dropna()
M.to_csv("USDZAR3M_prepared_Ind.csv")
symbol='USDZAR'
M.columns='{} '.format(symbol)+M.columns.values

prepared_igdata= prepare_igdata('XAUUSD1Y',start_date,end_date)
prepared_igdata.to_csv("XAUUSD1Y_prepared_IG.csv")
Bollinger_Val=Bollinger_Value(prepared_igdata['Close'],20)
HL_Val=HL_Value(prepared_igdata['Close'],1060)
rets=((prepared_igdata['Close']/prepared_igdata['Open'])-1)*10000
rets=rets.to_frame(name='Ret')
RSI_Val=RSI(rets,14)
Clock=prepared_igdata['Clock'].to_frame(name='Clock')
M2=Clock.join(HL_Val)
M2=M2.join(Bollinger_Val)
M2=M2.join(RSI_Val)
M2=M2.join(rets)
M2=M2.dropna()
M2.to_csv("XAUUSD3M_prepared_Ind.csv")
symbol='XAUUSD'
M2.columns='{} '.format(symbol)+M2.columns.values

Mfinish=M[M.columns[:len(M.columns)-1]].join(M2[M2.columns[2:]])
Mfinish=Mfinish.join(M[M.columns[len(M.columns)-1]])
Mfinish.fillna(0,inplace=True)
Mfinish.to_csv("USDZAR-XAUUSD3M_prepared_Ind.csv")
print(Mfinish)

totaltime=time.time()-starttime
print(totaltime)





