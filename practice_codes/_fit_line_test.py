from matplotlib import pyplot as plt
import numpy as np

#linear solver

def linfit(x,y): 

     #a = (len(x) * np.sum(x * y) - (np.sum(x) * np.sum(y))) / (len(x) * np.sum(x * x) - np.sum(x) * np.sum(x))
    #b = (np.sum(y) - a * np.sum(x)) / len(x) 
    a = (len(x) * sum(x*y) - (sum(x) *sum(y))) / (len(x) * sum(x*x) - sum(x)*sum(x))
    b = (sum(y) - a * sum(x)) / len(x)
    print(f"a={a}, b={b}")

    #Testing ready made function for the proof
    #a, b = np.polyfit(x, y, 1)
    return a,b

def main():
    x = np.random.uniform(-2 ,5 ,10)
    y = np.random.uniform(0,3,10)
    print ("THIS: ", x)
    a , b = linfit(x,y)
    plt.plot(x,y,'kx')
    print("Values for x: ", x)
    xp = np.arange(-2,5,0.1)
    plt.plot(xp,a*xp+b ,'r-')
    print (f"My fit : a={a} and b={b}")
    plt.show()
    
main()