import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
#데이터
data=pd.read_csv(os.getcwd() + '\data.csv')

X=np.array(data['x']).reshape(-1,1)
Y=np.array(data['y']).reshape(-1,1)

#for i in range(len(X)):
#    X[i] += 1960

def costFunction(X,y,a,b):
    return np.sum((X*a+b-y)**2)/2/len(y)
def gradA(X,y,a,b):
    return np.sum((a*X+b-y)*X)/len(y)
def gradB(X,y,a,b):
    return np.sum(a*X+b-y)/len(y)
def train(X,y,ite,lr):
    a,b=0,0
    cost=[]
    temp = 0
    percentage = ite / 100
    print(percentage)
    for _ in range(ite):
        if _ % percentage == 0:
            print(str(temp) + '% has been advanced...')
            temp += 1
        if _%100==0:
            cost.append(costFunction(X,y,a,b))
        ga=gradA(X,y,a,b)
        gb=gradB(X,y,a,b)
        a-=lr*ga
        b-=lr*gb
    return [a,b,cost]

a,b,cost=train(X,Y,10000000,0.001)
print('100%! completed')
plt.scatter(X,Y)
plt.plot([np.min(X),np.max(X)],[a*np.min(X)+b,a*np.max(X)+b])
plt.show()
plt.plot(cost)
plt.show()

print(a,b)
