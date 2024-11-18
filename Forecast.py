'''
Author: Cooper Lawrenz

Forecast time series data given a csv file with first column being dates and second values (sorted properly!)

'''

import csv
import numpy as np
from keras import layers
import keras
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

def build_model(_input_shape,_units):
    '''
    This method initalizes a keras model with one LSTM layer to a 1 unit dense layer

    Args: the input vector shape
    Returns:  the Model
    '''
    model = keras.Sequential([
        layers.InputLayer(shape = _input_shape),
        layers.LSTM(units=_units),
        layers.Dense(1, name = 'Output' )
    ])
    return model

def format(file, window_size):
    '''
    This method formates data given a window size and file and returns data in more convienient formats
    Note: window size is the length of the input vector for the LSTM model

    Args a csv file, and int representing window size 
    Returns training and test sets as np arrays along with a list of coresponding labels(dates)
    '''
    labels = []
    values = []
    with open(file, 'r') as data: #open the file
        reader = csv.reader(data) #convert to easier format 
        for row in reader: #for each row 
            try:
                values.append(float(row[1])) #add to respective lists 
                labels.append(row[0])
            except: #this exists for any wierd stuff that can end up in a csv file
                pass

    if len(values)<.1*window_size: #not a safe amount to train and plot... I cant guarantee it wont crash is this is not true
        print('This dataset is too small... It must be atleast 10 times greater than than the vector lenght of the input')
        quit() #exit the program 

    values = np.array(values) # convert our values to np arrays because they're better 
    train_range = int(len(values) * 5 // 6) #the first 5/6th of our data will be for training, the rest for testing 
    train_set_x = []
    train_set_y = []

    #the main idea here is that given a window of size n containing previous values lets predict the very next value
    #[march-01,march-02,march-03] ---> LSTM ---> [march-04]

    for i in range(train_range - window_size): 
        train_set_x.append(values[i:i+window_size])#for each element, generate an array of length windowsize with the next values
        train_set_y.append(values[i+window_size])#add the goal value as the one scalar after the above window

    train_set_x = np.array(train_set_x) #numpy arrays make everything easier
    train_set_y = np.array(train_set_y)

    test_set_x = []
    test_set_y = []
    for i in range(train_range - window_size, len(values) - window_size):
        test_set_x.append(values[i:i+window_size]) #now do the same as above just for the test set
        test_set_y.append(values[i+window_size])

    test_set_x = np.array(test_set_x)
    test_set_y = np.array(test_set_y)
    test_labels = labels[train_range:]# subset the labels to just be those spanning test values... this is primarily for plotting purpouses later on

    return train_set_x, train_set_y, test_set_x, test_set_y, test_labels

def fit_model(model, train_set_x, train_set_y, test_set_x, test_set_y, file_name, _epochs):

    '''
    This method prepares data for training and trains the model given a set of hyperparamaters 

    Args: training data in arrays, file name to save model as str, and number of epochs to train for int
    Returns: the fit model
    '''
    #prepare data
    train_set_x = np.reshape(train_set_x, (train_set_x.shape[0], train_set_x.shape[1], 1)) #lets reshape our data for input into an LSTM layer
    train_set_y = np.reshape(train_set_y, (train_set_y.shape[0], 1))

    test_set_x = np.reshape(test_set_x, (test_set_x.shape[0], test_set_x.shape[1], 1)) #once again just prep it for LSTM format
    test_set_y = np.reshape(test_set_y, (test_set_y.shape[0], 1))

    optimizer = Adam(learning_rate=0.001) #Adam because its the best
    model.compile(optimizer=optimizer, loss='mean_squared_error') #compile with adam and MSE becuase this is a regression task 
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=20, restore_best_weights=True)#Early Stopping to help find best model

    model.fit(train_set_x, train_set_y, epochs=_epochs, batch_size=32, shuffle=True, validation_data = (test_set_x,test_set_y),callbacks=[early_stopping]) #we are training on batch size 32 and shuffling the data
    #NOTE: for whatever reason shuffling the time-series data actualy produces better forcasting results... which is somewhat strange....
    
    model.save(file_name+'.keras',include_optimizer=False) #save the model 

    return model

def forecast(model, train_set_y, future_steps, window_size):
    '''
    Now predict future values with a trained model

    Args: a model, data to predict on, how far to predict, and the length of the input data
    Returns: an array of predictions
    '''
    predictions = [] #where we will store our predictions
    previous_window = train_set_y[-window_size:] #lets only train on the last (windowlength) of values
    previous_window = np.reshape(previous_window, (1, window_size, 1))#format the data for the model

    for _ in range(future_steps): #iterate through future steps
        predict = model.predict(previous_window) #predict on the provided data
        predictions.append(predict[0, 0])#add the prediction
        previous_window = np.roll(previous_window, shift=-1, axis=1)#pop out our value at index 0 
        previous_window[:, -1, 0] = predict[0, 0] #add our predicted value to the end of the vector being predicted on
        #continue this process until you have predicted the requested amount

    return predictions

def plot(prediction, test, labels, actual):
    '''
    Plot our predicted values
    
    Args: predicted values, values predictions were trained on, their lables and the actual values
    Returns: None, however prints out a graph
    '''
 
    test = test.tolist() # this is a np array lets make it a normal so we can add them
    total = test + prediction #add these two arrays 

    dates = pd.to_datetime(labels) #this date representation is more friendly to matplot
    dates = dates.to_numpy() #same as prior line 

    plt.plot(dates[:len(test)], total[:len(test)], color='r', label='Test Data') #plot our test data with the coresponding dates
    plt.plot(dates[len(test):len(test) + len(prediction)], prediction, color='b', label='Prediction') #plot our prediction with coresponding dates

    # Plot actual data starting from where the test data ends
    actual_dates = dates[len(test):len(test) + len(actual)]  # Subset of dates corresponding to actual data
    actual = actual[:len(actual_dates)]  # Truncate actual data to match the length of actual_dates
    plt.plot(actual_dates, actual, color='g', label='Actual') #plot actual data with coresponding dates 

    #the MAE provided here is calculated between what the model predicted versus what actualy happened. 
    footnote_text = f'MAE between predicted and actual: {np.mean(np.abs(actual - prediction))}' #because graphs are famously deciving lets have a standardized metrics acompany it
    plt.text(0.5, -0.1, footnote_text, ha='center', va='center', fontsize=8, color='black', transform=plt.gca().transAxes)

    plt.ylabel("Values") #labels and such for plot 
    plt.title("Forecasted Exchange Rates")
    plt.legend()
    plt.show()

def build_New(choice, selection):
    '''
    In a sense a subset of main. if user picks 1 this handels all their requests.

    Args: if they wanted to set parameters or not and the rest of their choices as an array
    Returns" none
    '''
    selection = selection.split(' ')

    if choice == "No": #if you said no we will use default values, there is no theory here... these values just performed best during my tests
        data = selection[0]
        file_name = selection[1]
        future_steps = 20
        epochs = 50 #feel free to extend this value... it will train better, however will take longer
        units = 20
        window_size = 68

    else:
        data = selection[0] #1
        file_name = selection[1] #2
        future_steps = int(selection[2])#3
        epochs = int(selection[3]) #4
        units = int(selection[4])#5
        window_size = int(selection[5])#6

    train_set_x, train_set_y, test_set_x, test_set_y, test_labels= format(data, window_size) #format the data
    model = build_model((window_size,1),units) #build our model
    model = fit_model(model, train_set_x, train_set_y, test_set_x, test_set_y, file_name, epochs) #fit our model
    predictions = forecast(model, train_set_y,future_steps,window_size) #forecast on our model
    plot(predictions,train_set_y[-window_size:], test_labels[:window_size+future_steps], test_set_y) #plot our models predictions

def plot_2(prediction, window, labels): 
    '''
    This is a secondary plotting function for users who chose option 2
    
    Args: predicted values, the window predicted on, and dates
    Returns: None
    '''
    dates = pd.to_datetime(labels) #makes dates for matplot
    last_date = dates[-1]
    next_dates = pd.date_range(start=last_date, periods=30, freq='D') #generate future dates, hardcoded to 30... as is everything for choice 2
    dates = dates.append(next_dates) #add these dates to the list of dates
    
    plt.plot(dates[:len(window)], window, color='r', label='Historical')#plot the window aka historical values
    plt.plot(dates[len(window):len(window)+len(prediction)], prediction, color='b', label='Forecasted Values') #plot our forecasted values 

    plt.xlabel('Date') #labels 
    plt.ylabel('Price')
    plt.title('Historical and Forecasted')
    plt.legend()
    plt.show()

def load_and_run(model,file):
    '''
    This method exists to load up a keras model and forecast off of it

    Args: a model to forecast with, the data to forecast on 
    Returns: None
    '''
    dates = []
    window = []
    with open(file, 'r') as data: #read through data adding to respective arrays 
        reader = csv.reader(data)
        for row in reader:
            try:
                window.append(float(row[1]))
                dates.append(row[0])
            except:
                pass

    prediction = forecast(model,window[-500:],30,68) #forecast on the last 68 seen values 30 periods ahead  
    plot_2(prediction, window[-500:], dates[-500-len(prediction):]) #plot 500 periods of historical data and 30 periods of forecast

def main():
    '''
    A main function which interprets and processes the requests of a user

    Args: None
    Returns: None
    '''
    while True:
        choice = input("Would you like to 1. build a model from scratch or 2. run a prexisting model? choose(1 or 2), otherwise type q to quit] ")
        if choice == '1':
            preset = input("Would You like to use preset model architecture values? (y,n) ")
            if preset == 'n' or preset == 'N':
                selection = input("Please provide, seperated by space in the order, file path to train on (must be local), name to save your new network under, how far forward you would like to predict (<100 steps),the amount of epochs, how many units for the LSTM layer, and the vector length of input: ")
                build_New('n',selection)
            elif preset == 'y' or preset =='Y':
                selection = input("Please provide, seperated by space in the order, file path to train on (must be local) and name to save your new network under ")
                build_New('No', selection)
            elif preset == 'q':
                break

        elif choice == '2':
            selection = input("Please provide a model which has been trained(in .keras format): ")
            model = load_model(selection)
            selection_two = input(f"Please provide a data set to predict on with > (more than 1000 recomended) the window size the model was trained on (default = 68) observations in csv format ")
            load_and_run(model,selection_two)

        elif choice == 'q':
            break

if __name__ == '__main__':
    main()