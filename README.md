# Forecasting_project

This is a modest project to predict the number of passengers an american flight will has.

I used mlflow to track metrics and instead of use a conda enviroment, I choose to create a docker container to host the entire project. The training algorithm searches for the best combination of hyperparameters a list is providing. So in the iterate_runs.py script, I am looking to give a list of proposed lists of hyperparameters along the model will tune. Those 'parent' lists will have a random order of the 'child' lists to look for the best combination.

Ilustration of the results in mlflow GUI:  

  
<img width="707" alt="image" src="https://github.com/Ana1890/Forecasting_project/assets/67620315/0c44fcd5-5f5e-49f9-8b5e-baa95195c320">
  

And the child runs looks like:  

<img width="252" alt="image" src="https://github.com/Ana1890/Forecasting_project/assets/67620315/4fc3fd6e-fca4-4ef4-bfa6-6b932fd99461">  

This can be improved with a bayesian search instead of a random search process

