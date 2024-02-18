# Stochastic Gradient Descent: The Average Problem
*Beauty in Randomness*
Stochastic Gradient Descent (SGD) is the main algorithm behind `Back Propogation` which enables Neural Networks to learn.

## Table of Contents
1. [Problem Introduction](#problem-introduction)
2. [SGD High Level Intuition](#sgd-high-level-intuition)
3. [Defining Terms](#defining-terms)
	1. [Loss Function](#Loss-Function)
	2. [Stochastic](#Stochastic)
	3. [Gradient](#Gradient)
	4. [Gradient Descent](#Gradient-Descent)
5. [Why Stochastic?](#Why-Stochastic)
6. [`SGD` Algorithm](#sgd-algorithm)
7. [SGD in Python](#SGD-in-python)
	1. [Calculating Loss](#Calculating-Loss)
	2. [Generating Next Guess](#Generating-Next-Guess)
	3. [Displaying Data](#Displaying-Data)
8. [Plotting `SGD` With Varying Ranges](#Plotting-SGD-With-Varying-Ranges)
	1. [Tightly Clustered](#Tightly-clustered)
	2. [Small Outlier](#small-outlier)
	3. [Large Outlier](#large-outlier)
___
## Problem Introduction
Although this isn't your average problem, perhaps your `avg()` function is broken and you don't know that `avg()` = `sum of data` / `# of data points`.

However, for each `guess` you make, you are given (1) `a randomly sampled datapoint` and (2) `the distance to that randomly sampled datapoint`.

Note: we will abbreviate Stochastic Gradient Descent as `SGD` 
___
## SGD High Level Intuition
With the information of (1) `random sample` and (2) `distance` of guess to sample, let us try to `tweak our guess` so that the distance is closer.

At a high level, SGD **minimizes the loss/difference** of a predicted and actual value, by **choosing datapoints randomly** and changing the guess by the `gradient` or amount of loss from that random point over many **training iterations**.

Traveling Analogy: 
- We are trying to walk towards USC, however we don't have the address--but hope is not lost, we have a guide!
- After each step, the guide will tell us how close we are to a random classroom
- As the traveler, we will incrementally take small steps and `guess` towards where we that random classroom is (since the classroom is at USC) and adjust our step size/direction based on what the guide tells us

| SGD Process | Traveling Analogy | `Average` Problem |
| ---- | ---- | ---- |
| Target "Ground Truth" Value | USC | Average of the dataset |
| Stochastic | Guide randomly choosing a classroom for us to walk towards | Choosing a random datapoint in the array |
| Gradient | Guide telling us how far away we are from the classroom | Distance of guess from random datapoint |
| Descent | Iteratively walking closer to campus by tweaking our path after each answer from the guide | Changing our guess with respect to the magnitude and direction of the `loss` |
___
## Defining Terms
### Loss Function
- Intuition: How far away are we from our desired outcome
- Generally, `Loss Function` follows some general pattern of `Predicted Value By Algorithm - Actual Value in Training Dataset`
<small>Note: This is sometimes called an `Objective Function`</small>
### Stochastic
- Randomly choosing a datapoint at each training iteration
- Our intuition is that the desired value (`average`) is represented by each datapoint
	- Each datapoint has its own probability of being chosen and the desired value will reflect how prevalent this datapoint is 
### Gradient
- Intuition: A relationship between each `input parameter` and the loss function
- Mathematically represented as a set of numbers related to how each `guess` changes the loss
	- In order to model the relationship of `input parameters` to the `Loss Function`, each component in the vector is a `partial derivative` w.r.t the loss function
	- More information on [Partial Derivates and Vectors ](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors/)

## Gradient Descent
- Intuition: If we now know the relationship between `input paremter` and `loss function`, we can tweak certain parameters to minimize the loss function
- Iteratively walk towards the direction that minimizes the loss (`negative to the gradient`)
___
## Why Stochastic?
- Intuition: If at every step towards USC we calculated our distance from every classroom, we would eventually get to USC, but this would take many extra calculations.
	- Instead by `stochastically` or randomly choosing a classroom, we can get a good approximation of where we need to go
- Since the `Gradient` portion of `Gradient Descent` requires taking the partial derivative of the Loss function:
	- (1) this requires the Loss function to be differentiable
	- (2) may be extremely computationally costly when the loss function takes in many parameters (usually a large weight vector representing the features of the data)
___
## `SGD` Algorithm
```python
1) Loss = Guess - Random Sampled Point
2) Guess.next = Guess.current - learning_rate * Loss
3) Guess.current <- Guess.next and repeat
```

Example: 
`input array` = `[1,2,3]` | `average` = `(1+2+3)/3 = 2`
`learning_rate` = `0.5` (quite high but illustrative for our purposes)

| Iteration | Guess.current | Stochastic Datapoint | Loss | Guess.next |
| ---- | ---- | ---- | ---- | ---- |
| 1 | 0 | `3` | Loss = 0 - 3 | 0 - (0.5)(-3) = 1.5 |
| 2 | 1.5 | `2` | Loss = 1.5 - 2 | 1.5 - (0.5)(0.5) = 1.75 |
| 3 | 1.75 | `1` | Loss = 1.75 - 1 | 1.75 - (0.5)(1) = 1.25 |
| 4 | 1.25 | `3` | Loss = 3 - 1.25 | 1.25 - (0.5)(1.75) = 2.125 |

After 4 iterations, we have approximated the average to be ~2 which is the actual average.

### Impacts of the Learning Rate
- See in iteration 3 that the `guess` value was approaching the correct value of 2 but decreased because of choosing the data point `1`
- Since the learning rate is `0.5`, the actual guess is tweaked by a little bit (imagine if the `learning_rate` approached `1.0`, then the guess would oscilate)
	- However, the same is true for values that increase the guess, the intuition is that the `learning_rate` dampens the effect of each `stochastic datapoint`
- Thus, the Learning Rate is a `speed vs accuracy` tradeoff as seen in the [`Outlier Dataset`](#outlier-dataset) which takes 600 training iterations to descend on the correct average value
### Impacts of the `(-)` sign
- When the loss is negative, that means the `guess` decreases the loss
	- Thus, we should keep doing what we were doing! Increase the guess to decrease the loss
- When the loss is positive, that means the`guess` increases the loss
	- Thus, we should go in the opposite direction of what we were doing! Decrease the guess to decrease the loss

<sup>Note: the loss is based off of a single `input output pair` which we called our `randomly sampled datapoint`, not every single value. Thus, the change that we make based on the `(-)` sign may be in the wrong direction, but in conjunction with the `learning_rate` dampens the effect of the wrong direction datapoints. </sup>

<sup>Then, if the data is highly spread out, we can see how the average will be "pulled" by extreme values but after enough sampling iterations each number should have approx. equal effect on the guess </sup>
___
## SGD In Python
### Calculating Loss
- Use the `np.random.choice(list, # of items)` to extract the stochastic datapoint
```python
def get_loss(guess: float, dataset: list) -> float:
  """
  guess: a starting point on where to approach true average
  dataset: a list representing all the numbers in the dataset
  """
  return guess - np.random.choice(dataset, 1)
```
### Generating Next Guess
- Let the next guess be tweaked by a fraction of the loss (based on the `learning_rate`)
```python
def get_next_step(guess: float, loss: float, learning_rate: float) -> float:
  """
  returns the next guess as a small step towards minimizing the loss
  Since `loss` is defined as `magnitude the guess if off from true average` or `guess - random datapoint`
  """
  return (float) (guess - loss * learning_rate)
```
### Displaying Data
- Data is plotted using `matplotlib` and displays datapoints from all training iterations
- Data is `decmiated`, a concept from Computer Vision, only including every `10th pixel` or in this case `guess` to make table plotting easier
	- Plotting is performed using the `tabulate` library
```python
def decimate_array(long_list: list, interval: int = 10) -> list:
  """Returns every `interval` index of the input list"""
  short_list = []
  for i, val in enumerate(long_list):
    if(i % interval == 0):
      short_list.append(val)
  return short_list
```
___
## Plotting `SGD` With Varying Ranges
Modeling Stochastic Gradient Descent with datasets containing varying ranges and random initializations to solve the task of finding the average of a dataset.

Below are samples of data with varying degrees of distributions with the datasets:
```
low_range = [3, 4, 5]
medium_range = [3, 4, 5, 30]
high_range = [3, 4, 5, 300]
```

| Range Type | Dataset | # of Training Iterations Needed |
| ---- | ---- | ---- |
| [Tightly Clustered](#Tightly-clustered) | `[3, 4, 5]` | 50 |
| [Small Outlier](#small-outlier) | `[3, 4, 5, 30]` | 50 |
| [Large Outlier](#large outlier) | `[3, 4, 5, 300]` | 600 |
___
### Tightly Clustered
![](https://github.com/Ky-Ng/Stochastic-Gradient-Descent-Modeler/blob/main/Assets/SGD_Small.png)
___
### Small Outlier
![](https://github.com/Ky-Ng/Stochastic-Gradient-Descent-Modeler/blob/main/Assets/SGD_Medium.png)
___
### Large Outlier
![](https://github.com/Ky-Ng/Stochastic-Gradient-Descent-Modeler/blob/main/Assets/SGD_600.png)
___
