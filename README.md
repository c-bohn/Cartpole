# OpenAI Gym Cartpole Solutions


## Description
This was a little project of mine as I wanted to get some more practical experience with reinforcement learning. 
I implemented a straightforward Q-Learning solution to the Cartpole game provided in the OpenAI Gym (https://www.gymlibrary.ml/environments/classic_control/cart_pole/). 
The first model is trained to predict the appropriate action at each timestep from the game's internal state, i.e., it receives as input the cart's position and velocity, as well as the pole's angle and angular velocity. With two hidden layers of 128 units each, after training for a while, this model achieves the maximum possible score of 500 steps in most episodes.
Another approach to the problem was to train the model not based on the internal state, but on its rendered output. To that end I implemented a CNN architecture that uses as input the current and two previous rendered frames and predicts the next action based on that. The inclusion of the previous frames is necessary for the model to be able to infer the (angular) velocity. This solution does not perform as well as the first one, but reliably achieves scores above 100 and occasionally reaches the max. score. With some more optimization and longer training, an equal performance should be achievable. 
Finally, I implemented a model that is not trained end-to-end on the problem but instead attempts to estimate the internal state (as used in the input for the first model) from the rendered frames. Using these state predictions as input to the first model works better than a random agent, but the scores achieved that way are only occasionally greater than 100. 




## Directory Structure

The root folder contains the following:

```
cartpole
├──cartpole                     --> The model trained on the internal state
├──cartpole_cnn                 --> The model trained on the rendered frames
└──cartpole_cnn_state_pred      --> The model predicting the state from the rendered frames
```


## Usage

The scripts for each model first ask whether to train the model. Entering 'y' starts the training loop, entering 'n' starts inference from a saved model checkpoint. Then just enter the number of episodes should be played. 

