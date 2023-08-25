import pandas as pd
import numpy as np

dataset = pd.read_csv('Credit_Card_Applications.csv')


'''
Identify the Frauds with SOM
'''
#Split the dataset as Y contains approval of credit card & X conatain except that class
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X=sc.fit_transform(X)

# Train SOM
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

# Visualize the Results

from pylab import bone , pcolor , colorbar ,plot , show
bone()
pcolor(som.distance_map().T)

colorbar()
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[Y[i]],markeredgecolor=colors[Y[i]],markerfacecolor=None,markersize=10,markeredgewidth=2)
    
show()

# Finding the Frauds

# Getting Co-ordinates of winning Nodes & rows associated with it
mappings=som.win_map(X)


# getting Co-ordinates of Outliers(winning nodes)
mapping2 = som.distance_map().T
all_coordinates = {}
i_counter = 0
for i in mapping2:
    counter_x = 0
    for x in i:
        value = x
        coordinate = (counter_x,i_counter)
        all_coordinates[coordinate] = value
        counter_x += 1
    i_counter += 1
white = []
for x in all_coordinates.keys():
    if all_coordinates[x] >= 0.90:
        white.append(x)



frauds=[]
for i in white:
    if i in mappings:
        if len(frauds) == 0:
            frauds=mappings[i]
        else:
            frauds=np.concatenate((frauds,mappings[i]))
   


frauds = sc.inverse_transform(frauds)


f = np.concatenate((mappings[(7,4)],mappings[7,5],mappings[(1,8)]))
f = sc.inverse_transform(f)

'''
Going from unsupervised to supervised deep learning Model
'''

# creating the matrix of features
customers = dataset.iloc[:,1:].values

#Creating Dependent Variable
is_fraud = np.zeros(len(customers))
for i in range(len(customers)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]=1



# Feature Scaling 
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
customers=sc.fit_transform(customers)


'''
ANN Model
'''

from keras.layers import Dense
from keras.models import Sequential

classifier = Sequential()
classifier.add(Dense(units=2,init='uniform',activation='relu',input_dim=15))
classifier.add(Dense(units=1,init='uniform',activation='sigmoid'))

#Compile ann
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting ann to trainig set
classifier.fit(customers,is_fraud,batch_size=1,epochs=5)


# predicting the probabilities of frauds
y_pred= classifier.predict(customers)

# Adding customer id along with probabilities
y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)

#Sorting
y_pred = y_pred[y_pred[:,1].argsort()]







