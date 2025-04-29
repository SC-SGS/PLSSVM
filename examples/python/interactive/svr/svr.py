import numpy as np
import sklearn.svm
from sklearn.metrics import r2_score
import plssvm
import time

from bokeh.models import RadioButtonGroup, Button, Div, Select, NumericInput, Paragraph, Spacer
from bokeh.layouts import column, row, layout

# custom files
from util.controls import DiscreteLogSliderWithInput
from svr.data_generation import generate_noisy_regression_dataset
from svr.plot import update_plot, create_plot
from svr.regression_report import update_regression_report_plot, create_regression_report_plot
from svr.prediction_vs_actual_plot import update_prediction_vs_actual_plot, create_prediction_vs_actual_plot


def create_svr_layout():
    # --- UI elements ---
    kernel_radio = RadioButtonGroup(labels=["linear", "poly", "rbf", "sigmoid", "laplacian"], active=2)
    dataset_select = Select(value="linear", width=160, options=["linear", "quadratic", "cubic", "v", "step", "sine", "tanh", "1/x", "irregular"])
    generate_button = Button(label="Generate New Dataset", button_type="primary")
    noise_input = NumericInput(value=0.0, mode="float", low=0.0, high=1.0)
    n_samples_input = NumericInput(value=40, low=5, high=200)

    # --- SVM parameter inputs ---
    default_params = sklearn.svm.SVR().get_params()
    cost_input = DiscreteLogSliderWithInput("cost:", initial_value=default_params["C"])
    degree_input = DiscreteLogSliderWithInput("degree:", values=[1, 2, 3, 4, 5, 6, 7], initial_value=default_params["degree"], low=1)
    gamma_radio = RadioButtonGroup(labels=["scale", "auto", "float"], active=0)
    gamma_input = DiscreteLogSliderWithInput("gamma-value::", initial_value=0.01, low=1e-10)
    coef0_input = DiscreteLogSliderWithInput("coef0:", values=np.arange(-4.5, 5, 0.5).tolist(), initial_value=default_params["coef0"])

    # --- generate initial dataset ---
    X, y = generate_noisy_regression_dataset(dataset_select.value, n_samples_input.value, noise_input.value)

    def get_params_from_current_options():
        """Return a dictionary containing the currently selected options from the RadioGroups and NumericInputs."""
        return {
            "C": cost_input.value,
            "degree": degree_input.value,
            "gamma": gamma_input.value if gamma_radio.active == 2 else gamma_radio.labels[gamma_radio.active],
            "coef0": coef0_input.value,
            "kernel": kernel_radio.labels[kernel_radio.active]
        }

    def enable_disable_inputs(current_params):
        """Depending on the provided parameters, disable or enable the input fields."""
        degree_input.disabled = current_params["kernel"] != "poly"
        gamma_radio.disabled = current_params["kernel"] == "linear"
        gamma_input.disabled = current_params["kernel"] == "linear" or gamma_radio.active != 2
        coef0_input.disabled = current_params["kernel"] not in ["poly", "sigmoid"]

    class SVRModel:
        """A wrapper class around an arbitrary SVR model. Creates a model and fits it to the data."""

        def __init__(self, model, X_train, y_train):
            """Create a new SVR model and fit the provided training data."""
            # create and fit the model
            tstart = time.time()
            self.model = model
            self.model.fit(X_train, y_train)
            tend = time.time()
            # store the time taken to train the model
            self.time = (tend - tstart) * 1000.0

    def train_models(svm_params, X_train, y_train):
        """Train the SVMs and compute the decision boundaries."""
        # train plssvm.svm.SVR
        trained_plssvm_model = SVRModel(plssvm.svm.SVR(**svm_params), X_train, y_train)

        # if the laplacian kernel is selected, we can't train the sklearn model
        if svm_params["kernel"] == "laplacian":
            return None, trained_plssvm_model

        # train sklearn.svm.SVR
        trained_sklearn_model = SVRModel(sklearn.svm.SVR(**svm_params), X_train, y_train)

        return trained_sklearn_model, trained_plssvm_model

    # --- initial model training ---
    initial_params = get_params_from_current_options()
    enable_disable_inputs(initial_params)
    sklearn_model, plssvm_model = train_models(initial_params, X, y)

    # --- create the initial plots ---
    sklearn_fig, sklearn_plot, sklearn_plot_source, sklearn_data_source, sklearn_pred = create_plot("sklearn.svm.SVR", sklearn_model, X, y)
    sklearn_prediction_vs_actual, sklearn_prediction_vs_actual_source, sklearn_prediction_vs_actual_bisector_source = create_prediction_vs_actual_plot(y, sklearn_pred)
    sklearn_regression_report, sklearn_regression_report_source = create_regression_report_plot(y, sklearn_pred)
    sklearn_text = Div(text=f"score: {r2_score(y, sklearn_pred):.3f}<br>runtime: {sklearn_model.time:.2f}ms",
                       styles={'font-size': '16px', 'color': 'black'})

    plssvm_fig, plssvm_plot, plssvm_plot_source, plssvm_data_source, plssvm_pred = create_plot("plssvm.svm.SVR", plssvm_model, X, y)
    plssvm_prediction_vs_actual, plssvm_prediction_vs_actual_source, plssvm_prediction_vs_actual_bisector_source = create_prediction_vs_actual_plot(y, plssvm_pred)
    plssvm_regression_report, plssvm_regression_report_source = create_regression_report_plot(y, plssvm_pred)
    plssvm_text = Div(text=f"score: {r2_score(y, plssvm_pred):.3f}<br>runtime: {plssvm_model.time:.2f}ms",
                      styles={'font-size': '16px', 'color': 'black'})

    # --- the update function called whenever an input option changes ---
    def update():
        """The update function called whenever an input option changes."""
        # retrain the models
        params = get_params_from_current_options()
        enable_disable_inputs(params)
        sklearn_model, plssvm_model = train_models(params, X, y)

        # update the sklearn plot
        sklearn_pred = update_plot(sklearn_model, sklearn_plot_source, sklearn_data_source, X, y)

        # if the sklearn model could be trained, update the plots, otherwise hide the sklearn plot
        if sklearn_model is None:
            # sklearn_model not trained (kernel = "laplacian")
            sklearn_text.text = "score: -<br>runtime: -"
        else:
            # sklearn_model is trained
            update_prediction_vs_actual_plot(sklearn_prediction_vs_actual_source, sklearn_prediction_vs_actual_bisector_source, y, sklearn_pred)
            update_regression_report_plot(sklearn_regression_report_source, y, sklearn_pred)
            sklearn_text.text = f"score: {r2_score(y, sklearn_pred):.3f}<br>runtime: {sklearn_model.time:.2f}ms"

        # update the plssvm plot
        plssvm_pred = update_plot(plssvm_model, plssvm_plot_source, plssvm_data_source, X, y)
        update_prediction_vs_actual_plot(plssvm_prediction_vs_actual_source, plssvm_prediction_vs_actual_bisector_source, y, plssvm_pred)
        update_regression_report_plot(plssvm_regression_report_source, y, plssvm_pred)
        plssvm_text.text = f"score: {r2_score(y, plssvm_pred):.3f}<br>runtime: {plssvm_model.time:.2f}ms"

    def generate_and_update():
        """Generate new dataset, then update the plots."""
        nonlocal X, y
        X, y = generate_noisy_regression_dataset(dataset_select.value, n_samples_input.value, noise_input.value)
        update()

    # --- register callbacks ---
    kernel_radio.on_change('active', lambda attr, old, new: update())

    cost_input.input_field.on_change('value', lambda attr, old, new: update())
    degree_input.input_field.on_change('value', lambda attr, old, new: update())
    gamma_radio.on_change('active', lambda attr, old, new: update())
    gamma_input.input_field.on_change('value', lambda attr, old, new: update())
    coef0_input.input_field.on_change('value', lambda attr, old, new: update())

    noise_input.on_change('value', lambda attr, old, new: generate_and_update())
    generate_button.on_click(lambda: generate_and_update())

    # --- create the layout ---
    return layout(
        column(
            row(sklearn_fig,
                Spacer(width=15),
                column(sklearn_prediction_vs_actual, sklearn_regression_report, sklearn_text, spacing=15),
                Spacer(width=20),
                Div(text="<hr style='border: 1px solid black; height: 580px;'>"),
                Spacer(width=20),
                plssvm_fig,
                Spacer(width=15),
                column(plssvm_prediction_vs_actual, plssvm_regression_report, plssvm_text, spacing=15), sizing_mode='stretch_width'),
            Div(text="<hr style='border: 1px solid black; width: 97.5vw;'>"),
            row(
                column(Paragraph(text="kernel:"), kernel_radio),
                spacing=20),
            row(
                cost_input.layout(),
                degree_input.layout(),
                column(Paragraph(text="gamma:"), gamma_radio),
                gamma_input.layout(),
                coef0_input.layout()),
            row(
                column(Paragraph(text="dataset type:"), dataset_select),
                column(Paragraph(text="#datapoints:"), n_samples_input),
                column(Paragraph(text="noise:"), noise_input),
                column(Paragraph(text=""), generate_button)),
            spacing=5, sizing_mode='stretch_width'),
        margin=(10, 20, 10, 20), sizing_mode='stretch_width')
