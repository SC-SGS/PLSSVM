import numpy as np
import pandas as pd
import sklearn

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper, LinearAxis
from bokeh.palettes import Viridis256


def confusion_matrix_as_dataframe(y_true, y_pred):
    """Compute the confusion matrix using y_true and y_pred and convert it to a Pandas DataFrame usable in bokeh."""
    # calculate the confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

    # get the unique class names
    unique_classes = [str(i) for i in np.unique(np.vstack((y_true, y_pred)))]
    y_true_range = sorted(unique_classes, reverse=True)
    y_pred_range = sorted(unique_classes)

    # convert to DataFrame for easy handling
    df = pd.DataFrame(confusion_matrix, index=unique_classes, columns=unique_classes)
    # reshape for bokeh
    df = df.stack().reset_index(name='value')

    # define a threshold for text color change
    threshold = confusion_matrix.max() * 0.5

    # assign white text for dark backgrounds and black text for bright backgrounds
    df["text_color"] = ["white" if val < threshold else "black" for val in df["value"]]

    # set alpha values -> per default everything is visible
    df["alpha"] = [1] * len(df["value"])

    return df, y_true_range, y_pred_range


def update_confusion_matrix_plot(fig, source, y_true, y_pred):
    """Update the already existing confusion matrix plot using y_true and y_pred."""
    # create the Pandas DataFrame representing a confusion matrix
    df, y_true_range, y_pred_range = confusion_matrix_as_dataframe(y_true, y_pred)

    # check if the number of unique classes has changed
    old_classes = fig.x_range.factors

    if set(y_true_range) != set(old_classes):
        # if classes changed, update axes ranges
        fig.x_range.factors = y_pred_range
        fig.y_range.factors = y_true_range

    # replace entire data dictionary
    source.data = df


def create_confusion_matrix_plot(y_true, y_pred):
    """Create a new confusion matrix plot using y_true and y_pred."""
    # create the Pandas DataFrame representing a confusion matrix
    df, y_true_range, y_pred_range = confusion_matrix_as_dataframe(y_true, y_pred)

    # create a ColumnDataSource for dynamic updates
    source = ColumnDataSource(df)

    # create a color mapper
    mapper = LinearColorMapper(palette=Viridis256)

    # create figure
    fig = figure(x_range=y_pred_range, y_range=y_true_range, toolbar_location=None, title="Confusion Matrix",
                 x_axis_label="y_pred", y_axis_label="y_true", x_axis_location="above",
                 sizing_mode='fixed', width=300, height=300)
    # disable grid lines (only visible if alpha is 0 anyway)
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    # disable dragging of the plot
    fig.toolbar.active_drag = None

    # draw rectangles
    fig.rect(x="level_1", y="level_0", width=1, height=1, source=source,
             fill_color={'field': 'value', 'transform': mapper}, line_color="white", fill_alpha="alpha")

    # add text labels
    fig.text(x="level_1", y="level_0", text="value", source=source,
             text_align="center", text_baseline="middle", text_color="text_color", text_font_size="10pt", text_alpha="alpha")

    fig.axis.major_label_text_font_size = "10pt"
    fig.axis.major_label_standoff = 1

    return fig, source
