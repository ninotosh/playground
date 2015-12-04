`simple_perceptron.py` demonstrates an N input simple perceptron.
It was implemented with reference to
http://natureofcode.com/book/chapter-10-neural-networks/

```shell
$ python simple_perceptron.py

run_1d(0.0336570291962, [0.05926078484651032], 0.964251781182)
learning_rate: 0.0336570291962, initial_weights: [0.05926078484651032], initial_bias: 0.964251781182
weights learned: [-0.7373991994829231]
   bias learned: 0.156
error rate: 1 / 1000 => 0.100%

run_and(0.355403915428, [0.8036252216007873, -0.6251097748470287], -0.790074674158)
learning_rate: 0.355403915428, initial_weights: [0.8036252216007873, -0.6251097748470287], initial_bias: -0.790074674158
weights learned: [0.8036252216007873, 0.08569805600867086]
   bias learned: -0.790
error rate: 1 / 4 => 25.000%

run_or(0.826198480346, [0.8088802419149403, 0.14769765069857876], -0.55259593246)
learning_rate: 0.826198480346, initial_weights: [0.8088802419149403, 0.14769765069857876], initial_bias: -0.55259593246
weights learned: [0.8088802419149403, 1.8000946113899554]
   bias learned: -0.553
error rate: 0 / 4 => 0.000%

run_2d(0.681897385171, [-0.543645489627123, 0.8469091199170737], 0.048476699597)
learning_rate: 0.681897385171, initial_weights: [-0.543645489627123, 0.8469091199170737], initial_bias: 0.048476699597
weights learned: [-25.488043520893587, 14.81455345697871]
   bias learned: -4.043
error rate: 56 / 1000 => 5.600%
```

`run_1d` classifies 1 dimensional data.

`run_and` and `run_or` simulates the logical AND / OR operations.

`run_2d` receives 2 dimensional data and outputs a scalar.

The example above shows error rates using randomized data sets.

----

`simple_network.py` demonstrates a neural network with 1 hidden layer using `SimplePerceptron` class in `simple_perceptron.py`.
It may be redundant in that every perceptron has its own learning rate and activation function.
Code to update biases are commented out because the implementation is based on and tested against
http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
