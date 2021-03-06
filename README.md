# Simple neural network

*Markus Dreyer, Feb 2015*

A feedforward neural network for tagging (i.e., sequence labeling).

This code demonstrates that implementing a neural network does not
require much code. The implementation is simpler than CRFs or related
models, since no manual feature engineering is required.

The implementation follows
[this](http://nlp.stanford.edu/~socherr/pa4_ner.pdf) project
description.

Inspired by
[this](https://justindomke.wordpress.com/2009/02/17/automatic-differentiation-the-most-criminally-underused-tool-in-the-potential-machine-learning-toolbox/)
post and others, I am using automatic differentiation here, rather
than implement the gradients by hand. This allows for rapid
prototyping and quick experimentation with different models. Automatic
differentiation is different from and much faster than numerical
differentiation. See
[here](http://en.wikipedia.org/wiki/Automatic_differentiation) for
more information.

The code depends on two external libraries:

* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for
  matrix data structures and matrix multiplication.

* [Adept](http://www.met.reading.ac.uk/clouds/adept/) for automatic
  differentiation.

Adept 1.0 (and earlier versions, presumably) have a bug in the `tanh`
differentiation. To fix it, change line 1486 in `adept.h` from this:

    ADEPT_DEFINE_UNARY_FUNCTION3(Tanh, tanh, Real e=exp(2.0*a_.value()), 4.0*e/((2.0+2)*e+1.0))

to this:

    ADEPT_DEFINE_UNARY_FUNCTION3(Tanh, tanh, Real tmp=tanh(a_.value()), 1.0-tmp*tmp)

I hear that the bugfix will be included in the next Adept version.

To compile `simple-nn`, specify the Eigen and Adept locations in the
Makefile, and type `make`. Then type:

    ./simple-nn -h

Which displays something like this:


    ./simple-nn: A neural network for tagging (experimental).
    Options:
     -h      Show this help text.
     -d dir  Specify the data directory.
     -f [tanh|sigmoid|softplus] Specify the nonlinear function.
     -n num  Specify the window size.
     -H num  Specify the hidden layer size.
     -l num  Specify the gradient descent learning rate.
     -s      Skip test and evaluation at the end.

Simply run with defaults like this:

    ./simple-nn

### Caveats

It seems that Adept slows down the score computation during training
considerably, so this is currently good for prototyping purposes only.
