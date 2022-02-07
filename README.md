# Predicting_stock_movement

In this project, the historical   APPLE  stock daily `Close` price data is used to build various machine learning models to predict the stock price movement. The historical data includes all data since APPLE stock's launch in 1980-12-12 until 2022-02-04. The models built include 

- baseline model using mean value

- smoothing model (single, double and triple exponential smoothing models)

- SARIMA models of various orders using `SARIMAX` python library

- SARIMA models of various orders using `pmdarima` python library

- FB Prophet model

- RNN model

- LSTM model. 

The models are trained using the historical data since its launch in 1980-12-12 until 2022-01-21, and  the last ten day's data (from 2022-01-24 to 2022-02-04) were hold out for out-of-sample testing.  Then the models' performances are compared using the metric MAPE (mean absolute percentage error). 


The best model we found is a  deep learning RNN model which  outperforms all other  models in making out-of-sample forecasts  in the metric of MAPE.  The RNN model we found slightly outperforms the LSTM model we built. 

A double exponential smoothing model and a SARIMA model provide slightly worse performance but is much inexpensive to train. 

A FB Prophet model and a SARIMA model built by the python library `pmdarima`   perform significantly worse, which is somewhat surprising. 

Meanwhile we note that there is a significant amount of randomness in training a deep learning model (either RNN or LSTM), even with the same set of hyper-parameters. Of course, training a deep learning model is much more expensive in terms of computing time. 
