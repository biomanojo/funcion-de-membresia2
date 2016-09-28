
# coding: utf-8

# In[1]:






# $$ FUNCIONES - DE - MEMBRESIA $$

# In[3]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,100)
a=10
b=50
c=90

plt.xlabel('Valores de X')
plt.title('Funcion Triangulo')

plt.grid()

def f(x,a,b,c):
    if ((x<a) | (x>=c)):
        ans=0
    if ((a<=x) & (x<b)):
         ans=(x-a)
    if ((b<=x) & (x<=c)):
        ans=c-x
    return ans

f_vec = np.vectorize(f)
func=f_vec(x,a,b,c)
plt.plot(x,f_vec(x,a,b,c))



# 
# 
# $$ FUNCION - TRIANGULAR $$
# 
# $$f(x)=\begin{cases}{0}&\ x\leq a\\\frac{x-a}{b-a} &\ a\leq x \leq b \\\frac{c-x}{c-b} & \ b\leq x \leq c\\\ {0}&\ x\geq a\end{cases}$$
# 
# $$μΑ(x) ={a< m< b}$$
# 
# 
# 

# In[4]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x=[0,10,10,20,100,150,180,200]
x0=10
x1=30
x2=180
x3=200
plt.grid()
def fxTrapezoide(x,x0,x1,x2,x3):
 """
     funcion de membresia Trapezoide
 """
 if x<=x0:
     return 0.0
 if x>x0 and x<=x1:
     return (x/(x1-x0))-(x0/(x1-x0))
 if x>x1 and x<=x2:
     return 1.0
 if x>x2 and x<=x3:
     return -(x/(x3-x2))+(x3/(x3-x2))

     return 0.0
 
f_vec = np.vectorize(fxTrapezoide)
func=f_vec(x,x0,x1,x2,x3)
print("func")
plt.axis([x[0], x[1],-0.0 , 1.5])
plt.plot(f_vec(x,x0,x1,x2,x3))


# $$ FUNCION -PARAIZOIDAL$$
# 
# \begin{cases}{0}&\ x\leq a\\\frac{x-a}{b-a} &\ a\leq x \leq b \\{1}&\ b\leq x\leq c \\\frac{d-x}{d-c} & \ c\leq x \leq d \\\ {0}&\ x\geq d\end{cases}
# 
# \begin{cases}{ a\leq b \leq c \leq d }\end{cases}

# In[5]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

x=[0,10,10,20,100,150,180,200]
x0=10
x1=30
x2=200
x3=200
plt.grid()
def fxTrapezoide(x,x0,x1,x2,x3):
    """
        funcion de membresia Trapezoide
    """
    if x<=x0:
        return 0.0
    if x>x0 and x<=x1:
        return (x/(x1-x0))-(x0/(x1-x0))
    if x>x1 and x<=x2:
        return 1.0
    if x>x2 and x<=x3:
        return -(x/(x3-x2))+(x3/(x3-x2))

        return 0.0
    
f_vec = np.vectorize(fxTrapezoide)
func=f_vec(x,x0,x1,x2,x3)
print("func")
plt.axis([x[0], x[1],-0.1 , 1.5])
plt.plot(f_vec(x,x0,x1,x2,x3))


# $$ FUNCION - HOMBRO$$
# 
# \begin{cases}{0}&\ x\leq a\\\frac{x-a}{b-a} &\ a\leq x \leq b \\{1}&\ b\leq x\leq c \\\frac{d-x}{d-c} & \ c\leq x \leq d \\\ {0}&\ x\geq d\end{cases}
# 
# \begin{cases}{ a\leq b \leq c \leq d }\end{cases}

# In[8]:

import math
plt.grid()
def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)
plt.plot(x,sig)
plt.show()


# $$ FUNCION - SIGMOIDAL $$
# 
# $$F(X)= \frac{1}{1 + e^-a}$$

# In[ ]:



