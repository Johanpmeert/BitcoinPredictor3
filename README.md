# BitcoinPredictor3
Python deep learning model for Bitcoin price prediction using svr, including test trading with model during months

Data for the model is coming from a .csv file containing historical Bitcoin data since 2014
The dataset is expanded by adding numerous other features like average prices 7/30/365 days, standard deviations, moving averages, ... to end up with 37 features
The goal is to predict the closing price which effectively means 1 day ahead
We test this by simulation trades on a starting wallet and buy/sell/hold according to our model.

Disclaimer: This model is just to show what svr can do, use it for trading at your own risk.
