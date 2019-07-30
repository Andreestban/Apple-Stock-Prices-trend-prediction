# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset = pd.read_csv('apple.csv')
#taking Open stock price column 
training_set = dataset.iloc[:-21, 1:2].values

#normalizing data
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler(feature_range = (0,1))
training_set_scaled = ms.fit_transform(training_set)

#creating a empty list
X_train = []
y_train = []
for i in range(150,1741):
    X_train.append(training_set_scaled[i-150:i, 0])
    y_train.append(training_set_scaled[i,0])
#converting X and y into numpy array
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping X_rain array
# Reshaping(batch_size, timesteps, input_dim)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#preparing testing set
testing_set = dataset.iloc[-len(dataset) + len(training_set) :, 1:2].values

#preparing predinting dataset
pred_dataset = dataset.iloc[len(dataset)-len(testing_set) - 150:, 1:2].values

#scaling pred_dataset
pred_dataset = ms.transform(pred_dataset)

X_pred = []
for k in range(100,100 + len(testing_set)):
    X_pred.append(pred_dataset[k - 100:k,0])

#makingX_pred a numpy array
X_pred = np.array(X_pred)

#reshaping X_rain array
# Reshaping(batch_size, timesteps, input_dim)
X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

#importing keras deep learning libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#building model function
def build_model():
    #initializing model
    Model = Sequential()
    
    #adding first Lstm layer with dropout layer
    Model.add(LSTM(units = 128,
                   return_sequences = True,
                   input_shape = (X_train.shape[1],1)))
    Model.add(Dropout(0.20))
    
    #adding 2nd layer
    Model.add(LSTM(units = 128,
                   return_sequences = True))
    Model.add(Dropout(0.20))
    
    #adding 3rd layer
    Model.add(LSTM(units = 128,
                   return_sequences = True))
    Model.add(Dropout(0.20))
    
    #adding 4th layer
    Model.add(LSTM(units = 128,
                   return_sequences = True))
    Model.add(Dropout(0.20))
    
    #adding 5th layer
    Model.add(LSTM(units = 128,
                   return_sequences = True))
    Model.add(Dropout(0.20))
    
    #adding 6th layer
    Model.add(LSTM(units = 128,
                   return_sequences = True))
    Model.add(Dropout(0.20))
    
    #adding 7th layer
    Model.add(LSTM(units = 128,
                   return_sequences = True))
    Model.add(Dropout(0.20))
    
    #adding 8th layer
    Model.add(LSTM(units = 128))
    Model.add(Dropout(0.20))
    
    #adding output dense layer
    Model.add(Dense(units = 1))
    
    #compiling model
    Model.compile(optimizer = "adam",
                  loss = "mean_squared_error",
                  metrics = ["mean_squared_error"])
    
    return Model

#calling function
new_model = build_model()

#saving model architecture
from keras.utils import plot_model
plot_model(new_model,
           to_file = "model_arhitecture.png",
           show_shapes = True)

#training model
history = new_model.fit(x = X_train,
                        y = y_train,
                        batch_size = 100,
                        epochs = 100)  

y_pred  = new_model.predict(X_pred,
                            batch_size = 100) 

#applying inverse tranform to ypred
y_pred = ms.inverse_transform(y_pred)

#plotting testing vs Predicted values of stock prices
plt.title("Stock Price trend analysis") 
plt.xlabel("Time")
plt.ylabel("Stock prices")
plt.plot(y_pred,
         color = "red",
         label = "predicted stock prices")
plt.plot(testing_set,
         color = "blue",
         label = "true stock prices") 
plt.legend()
plt.show() 