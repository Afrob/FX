import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pydotplus

def prepare_igdata(symbol,start_date,end_date):
    Chead=["Date","Clock","Open","High","Low","Close","Volume"]
    df=pd.read_csv("{}.csv".format(symbol),sep=',',names=Chead,na_values=['nan'])
    df['Time']=df[['Date','Clock']].apply(lambda x : '{} {}'.format(x[0],x[1]), axis=1)
    df = df.set_index(['Time'])
    df=df[['Clock','Open','High','Low','Close','Volume']]
    dates=pd.date_range(start_date,end_date, freq="1min")
    M=pd.DataFrame(index=dates)
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

start_date='2017-09-24 00:00'
end_date='2017-09-30 23:59'
prepared_igdata= prepare_igdata('USDZAR1Y',start_date,end_date)
prepared_igdata.to_csv("USDZAR_prepared_IG.csv")
Bollinger_Val=Bollinger_Value(prepared_igdata['Close'],20)
HL_Val=HL_Value(prepared_igdata['Close'],1432)
rets=(prepared_igdata['Close']/prepared_igdata['Open'])-1
rets=rets.to_frame(name='Ret')
RSI_Val=RSI(rets,14)
Clock=prepared_igdata['Clock'].to_frame(name='Clock')
M=Clock.join(HL_Val)
M=M.join(Bollinger_Val)
M=M.join(RSI_Val)
M=M.join(rets)
M['Ret']=M['Ret'].shift(-1)
M=M[:-1]
print(prepared_igdata)#only for test
print(M)#only for test
M=M.dropna()
M.to_csv("USDZAR_prepared_Ind.csv")

