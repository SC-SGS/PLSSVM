from bokeh.models import Slider, NumericInput, CustomJS, Paragraph
from bokeh.layouts import column, row
from bokeh.io import curdoc
import numpy as np


class DiscreteLogSliderWithInput:
    def __init__(self, title, values=None, initial_value=0.0, add_zero=False, low=None):
        """Creates a discrete logarithmic slider with an independent numeric input."""
        # sanity check: iv values are explicitly provided, add_zero may not be true!
        if values is not None and add_zero:
            raise RuntimeError("values are explicitly provided, but 'add_zero' is True!")

        self.title = title

        # set the values if not provided
        self.values = values if values is not None else np.logspace(-4, 4, 9).tolist()  # Default: 10^-4 to 10^4
        if add_zero and values is None:
          self.values = [0.0] + self.values
          
        # get the initial slider position
        # if initial_value is in values, sets the slider accordingly
        self.initial_index = 0 if initial_value not in self.values else self.values.index(initial_value)

        # discrete slider (integer indices mapped to logarithmic values)
        self.slider = Slider(start=0, end=len(self.values) - 1, value=self.initial_index, step=1, width=200, show_value=False, tooltips=False)

        # numeric input for manual value entry
        if low is not None:
            self.input_field = NumericInput(value=initial_value, mode='float', width=100, low=low)
        else:
            self.input_field = NumericInput(value=initial_value, mode='float', width=100)

        # sync slider -> input (update input field when slider moves)
        self.slider.js_on_change("value", CustomJS(args=dict(slider=self.slider, input_field=self.input_field, values=self.values), code="""
            let timeout;
            clearTimeout(timeout);
            timeout = setTimeout(function() {
                // Update the tooltip to show the actual value at the slider's position
                input_field.value = values[slider.value];
            }, 300);  // 300ms delay
        """))

    @property
    def value(self):
        return self.input_field.value

    @property
    def disabled(self):
        return self.slider.disabled

    @disabled.setter
    def disabled(self, is_disabled):
        self.slider.disabled = is_disabled
        self.input_field.disabled = is_disabled

    def layout(self):
        """Returns a row layout with the slider and input field."""
        return column(Paragraph(text=self.title), row(self.slider, self.input_field))
