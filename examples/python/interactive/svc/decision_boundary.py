import numpy as np

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.palettes import Category10


def compute_decision_boundary(model, X):
    """Computes the decision boundary using a mesh grid and predicts class labels."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.model.predict(grid_points)
    predictions = predictions.reshape(xx.shape)  # Reshape to 2D grid

    return xx, yy, predictions


def update_decision_boundary_plot(model, image, image_source, data_source, X, y):
    """Update the already existing decision boundary."""
    # update underlying data
    data_source.data = {'x': X[:, 0], 'y': X[:, 1], 'class': [str(label) for label in y]}

    # check if the model is defined -> only update plot in this case
    if model is not None:
        image.glyph.global_alpha = 0.5
        # compute the decision boundary
        xx, yy, prediction = compute_decision_boundary(model, X)

        # update source data
        image_source.data = {'image': [prediction], 'x': [xx.min()], 'y': [yy.min()],
                             'dw': [xx.max() - xx.min()], 'dh': [yy.max() - yy.min()]}
    else:
        image.glyph.global_alpha = 0


def create_decision_boundary_plot(title, model, X, y):
    """Create a decision boundary plot."""
    # the used figure
    fig = figure(title=f"{title}", x_axis_label="Feature 1", y_axis_label="Feature 2", sizing_mode='stretch_width',
                 toolbar_location='above')
    fig.title.text_font_size = '16pt'
    fig.toolbar.active_drag = None

    # compute the decision boundary
    xx, yy, prediction = compute_decision_boundary(model, X)

    # create the color mapper used for the data points and decision boundaries
    color_mapper = LinearColorMapper(palette=Category10[4], low=0, high=3)

    # create data source
    data_source = ColumnDataSource(data={'x': X[:, 0], 'y': X[:, 1], 'class': [str(label) for label in y]})

    # decision boundary as filled surface (with alpha for visibility)
    image_source = ColumnDataSource(data={'image': [prediction], 'x': [xx.min()], 'y': [yy.min()],
                                          'dw': [xx.max() - xx.min()], 'dh': [yy.max() - yy.min()]})
    image = fig.image(image='image', x='x', y='y', dw='dw', dh='dh', source=image_source,
                      color_mapper=color_mapper, level="underlay", alpha=0.5)

    # data points
    fig.scatter('x', 'y', source=data_source, color={'field': 'class', 'transform': color_mapper}, legend_field='class',
                size=7)

    # sort the legend entries
    legend = fig.legend[0]
    sorted_legend = sorted(legend.items)
    legend.items = sorted_legend

    return fig, image, image_source, data_source
