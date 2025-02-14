import numpy as np
import pandas as pd
import sklearn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.palettes import Viridis256

def update_prediction_vs_actual_plot(source, bisector_source, y_true, y_pred):
    """Update the already existing confusion matrix plot using y_true and y_pred."""

    # replace source
    source.data = {'x': y_true, 'y': y_pred }

    # find min and max for the angle bisector
    min_val = np.min(y_true)
    max_val = np.max(y_true)

    # replace bisector source
    bisector_source.data = {'x': [min_val, max_val], 'y': [min_val, max_val]}


def create_prediction_vs_actual_plot(y_true, y_pred):
    """Create a new prediction vs actual plot using y_true and y_pred."""
    # create figure
    fig = figure(toolbar_location=None, title="Prediction vs Actual", x_axis_label="actual", y_axis_label="prediction", sizing_mode='fixed', width=300, height=300)

    # create a ColumnDataSource for dynamic updates
    source = ColumnDataSource(data={'x': y_true, 'y': y_pred})

    # draw data points
    fig.scatter(x="x", y="y", source=source, size=3)
    # disable dragging of the plot
    fig.toolbar.active_drag = None

    # find min and max for the angle bisector
    min_val = np.min(y_true)
    max_val = np.max(y_true)

    # create a ColumnDataSource for dynamic updates
    bisector_source = ColumnDataSource(data={'x': [min_val, max_val], 'y': [min_val, max_val]})

    # draw the angle bisector
    fig.line(x="x", y="y", source=bisector_source, line_width=2, line_alpha=0.6, line_dash="dashed", color="green")

    return fig, source, bisector_source
