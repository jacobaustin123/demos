# examples 
Python demos and utilities, mostly for machine learning and image formatting. 

### preprocessing.py

Of particular interest is the preprocessing.py program, which takes images of 
arbitrary size, rebalances the colors, resizes them, and saves them in a variety
of image formats. For a full list of options, use the `python preprocessing.py
-h` command. Partially adapted from user Gauss256's kernel for the Kaggle Docs
vs. Cats Redux Kaggle challenge, available
[here](https://www.kaggle.com/gauss256/preprocess-images).

### python-notes.ipynb and python-module-notes.ipynb

These Jupyter notebooks contain tutorials covering all aspects of Python, from
class design to module usage to a general discussion of types and variable names
in Python.  

### central_limit.py

central_limit.py is a demonstration of the central limit theorem, computing
means from arbitrary distributions and plotting the output as a histogram, which
should approach a normal distribution.

### mandelbrot.py

A simple mandelbrot set generator. The size of the image and the number of
iterations performed on each pixel can be changed.

### name_generator.py

name_generator.py generates names and numbers in a variety of formats, for use
in a variety of programs, like the grading example from Koenig's Accelerated C++
Chapter 9.
