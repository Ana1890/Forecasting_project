# Forecasting_project

This is a modest project to predict the ammount of passengers of flights in US.

I integrated docker with mlflow to track metrics. The algorithm used to train the model search for the best hiperparameters of a given list. So in the iterate_runs.py script, I am looking to imput a random order of the hiperparameters which the algoritm would search.

Ilustration of the mlflow GUI I achieved:  

  
<img width="707" alt="image" src="https://github.com/Ana1890/Forecasting_project/assets/67620315/0c44fcd5-5f5e-49f9-8b5e-baa95195c320">
  

And the child runs looks like:  

<img width="252" alt="image" src="https://github.com/Ana1890/Forecasting_project/assets/67620315/4fc3fd6e-fca4-4ef4-bfa6-6b932fd99461">

