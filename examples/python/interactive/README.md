<!-- DOXYGEN_INTERACTIVE_EXAMPLE_LINK -->

# Interactive live comparison between sklearn.svm and PLSSVM

This directory contains a bokeh application that can be used to compare `sklearn.svm`'s and PLSSVM's classification and 
regression implementation directly besides each other. 
It is possible to change all available hyperparameters, e.g., kernel function, decision function shape, or the 
respective kernel function parameters. Additionally, the number of  classes and datapoints as well as the used dataset 
can be changed on the fly.

![Example of our bokeh application visualization between sklearn.svm and PLSSVM.](https://github.com/SC-SGS/PLSSVM/raw/develop/.figures/plssvm_bokeh.gif)

# Requirements

In order to run our interactive comparison, the following packages must be installed:

```bash
pip install numpy pandas bokeh scikit-learn plssvm
```

# Running

To start the bokeh server locally, it is sufficient to call (in the current directory):

```bash
bokeh serve svm.py
```

This will output something like:

```bash
2025-02-14 17:47:49,341 Starting Bokeh server version 3.6.3 (running on Tornado 6.4.2)
2025-02-14 17:47:49,343 User authentication hooks NOT provided (default user enabled)
2025-02-14 17:47:49,346 Bokeh app running at: http://localhost:5006/svm
2025-02-14 17:47:49,346 Starting Bokeh server with process id: 184614
```

You then simply have to open the prompted URL (in this example `http://localhost:5006/svm`) in a browser and enjoy our 
live comparison between `sklearn.svm` and PLSSVM!