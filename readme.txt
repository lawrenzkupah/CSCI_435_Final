Ensure you have Matplotlib


This program exists for two primary purposes. 

1)Building and testing forecasting models 
2)Forecasting with an already fit model

To run simply run the file 'Forecast.py' and you will be prompted with application choices.

Note: when providing this program with data take special care to only provide in CSV form. It must consists of two columns the first being dates and the second their respective values. Additionally do not train with less than 1000 observations (all data provided in this folder meets these specifications)

If your original selection == 1:
	type y to use presets (recommended)  ~training on these shouldn't take more than 5/10 minutes 
		then, provide the program with a csv file and name for that file, feel free to use any of the CSV's I have provided as they are in the proper format
	type n if you really want to get under the hood and choose parameters.... 

Be aware that Yen and SP500 data are normalized to make training easier

If your original selection == 2:
	provide a model to be used, once again, feel free to use any provided,
	then on a separate prompt provide data to forecast in csv form. I recommend predicting on the asset which the model was trained on.... 
 
I have images of all four indices being tested/validated if you don't want to run them yourselves.


Ultimately, forecasting this stuff was very difficult. This project has lead me to appreciate simply how much noise exists in this sort of data. 


If curious, this program can predict a straight line! Proof in the aptly named png. Ignore the second graph.