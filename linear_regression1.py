import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LinearRegression:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.b=[0,0]
    
    def gradient_descent(self,learning_rate):
        y_pred=self.predict()
        y=self.y
        m=len(y)
        self.b[0]=self.b[0]-(learning_rate*((1/m)*np.sum(y_pred-y)))
        self.b[1]=self.b[1]-(learning_rate * ((1/m) *np.sum((y_pred - y) * self.x)))
        
        
    def get_current_accuracy(self, y_pred):
        p, e = y_pred, self.y
        n = len(y_pred)
        return 1-sum(
            [
                abs(p[i]-e[i])/e[i]
                for i in range(n)
                if e[i] != 0]
        )/n

        




    def predict(self):

        
        y_pred=np.array([])
        b=self.b
        for x in self.x:
            y_pred=np.append(y_pred,(b[0])+(b[1]*x))

        return y_pred

    def cost_function(self,y_pred):
        m=len(self.y)
        
        J=((1/2*m)*(np.sum(y_pred-self.y)**2))
        return J


def main():
    x=np.array([i for i in range(11)])
    y=np.array([2*i for i in range(11)])
    regressor=LinearRegression(x,y)
    costs=[]
    learning_rate=0.01
    steps=100
    iterations=0
    while 1:
        y_pred = regressor.predict()
        cost = regressor.cost_function(y_pred)
        costs.append(cost)
        regressor.gradient_descent(learning_rate)
         
        iterations += 1
        if iterations % steps == 0:
            print(iterations, "epochs elapsed")
            print("Current accuracy is :",regressor.get_current_accuracy(y_pred))
 
            stop = input("Do you want to stop (y/*)??")
            if stop == "y":
                break
            
if __name__ == '__main__':
    main()

