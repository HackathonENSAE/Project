
# coding: utf-8

# In[78]:


from operator import*
import random
print(ord("Z".lower()))
print("Z".lower())
print(26%26)
print(ord("a".lower()))
def lettresuivante(l):
        return chr(((ord(l)-96)%26)+97)
print (lettresuivante("z"))
print (lettresuivante ("m"))


d={"chat":[30,20], "gorille":[20,10],"aspicot":[122,27], "grotadmorv":[50,22]}

d3=sorted(d.items(), key=lambda kv:max(kv[1]))

print(d3)

mots = ['eddueardo999', 'catelyn999', 'robb', 'sansa', 'arya', 'brandon',
        'rickon', 'theon', 'rorbert', 'cersei', 'tywin', 'jaime',
        'tyrion', 'shae', 'bronn', 'lancel', 'joffrey', 'sandor',
        'varys', 'renly', 'a' ]

def mots_lettre_position (liste, lettre, position) :
    d={}
    L=[]
    for m in mots:
        if len(m)>position-1:
            if m[position-1]==lettre:
                L.append(m)
    return L
print(mots_lettre_position(mots,"y",2))


# In[84]:


def code_vigenere ( message="JENESUISPASCODE", cle="DOP") :
    ord_a=ord("a")
    message=message.lower()
    décalages=[ord(l)-ord_a for l in cle.lower()]
    code=""
    s1=len(cle)
    for i in range(len(message)):
        code+=chr((ord(message[i])+décalages[i%s1]-ord_a)%26+ord_a).upper()
        
    return code

print(code_vigenere())


# chr(69)

# In[80]:


ord("a")


# In[128]:


import random, numpy as np
get_ipython().magic('pylab inline')

X=np.random.random(100)*80
Y=[random.random()*80 for i in range(100)]

plot(X,Y,'o',color='b', linewidth=0.5)
ylabel("some numbers")
xlim(-2,100)

show()




# In[138]:


from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure())
def f(x,y) :
    return x**2 - y**2
X = np.arange(-1, 1, 0.02)
Y = np.arange(-1, 1, 0.02)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
ax.plot_surface(X, Y, Z)
plt.show()

print(np.array([1,2,3,4], shape=(2,2)))


# In[143]:


def creerstructurevide():
    return [False,{}]
    
def ajout(mot,struct):
    """
    on créer structure liste contenant des dicos de la forme L0= [False,{"lettre1":L1,"lettre2":L2}]
    """
    if len(mot)==0:
        struct[0]=True
        return struct
    l=mot[0]
    if l not in struct[1]:
        s=creerstructurevide()
        struct[1][l]=s
        ajout(mot[1:],struct[1][l])
    else:
        
        ajout(mot[1:],struct[1][l])
    return struct
        
    
def recherche (mot,struct):
    if len(mot)==0:
        return struct[0]
    l=mot[0]
    if l not in struct[1]:
        return False
    else:
        return recherche(mot[1:],struct[1][l])
        
struct=creerstructurevide()
ajout("lopus",struct)
print(recherche ("lopus",struct))

