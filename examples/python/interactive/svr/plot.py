from bokeh.plotting import figure
from bokeh.models import ColumnDataSource


def update_plot(model, plot_source, data_source, X, y):
    """Update an already existing plot."""
    # update underlying data
    data_source.data = {'x': X, 'y': y}

    if model is not None:
        pred = model.model.predict(X)
        plot_source.data = {'x': X.flatten(), 'y': pred.flatten()}
        return pred
    else:
        plot_source.data = {'x': [], 'y': []}
        return None


def create_plot(title, model, X, y):
    """Create a plot."""
    # the used figure
    fig = figure(title=f"{title}", x_axis_label="Feature", y_axis_label="Target", sizing_mode='stretch_width',
                 toolbar_location='above')
    fig.title.text_font_size = '16pt'
    fig.toolbar.active_drag = None

    # create data source
    data_source = ColumnDataSource(data={'x': X, 'y': y})

    # the predicted plot
    pred = model.model.predict(X)
    plot_source = ColumnDataSource(data={'x': X, 'y': pred})
    plot = fig.line(x='x', y='y', source=plot_source, line_width=2, color='red')

    # data points
    fig.scatter('x', 'y', source=data_source, size=8)

    return fig, plot, plot_source, data_source, pred
