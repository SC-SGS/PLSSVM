import numpy as np
import pandas as pd
import sklearn.svm
from sklearn.metrics import accuracy_score
import plssvm
import time

from bokeh.models import RadioButtonGroup, Button, Div, Select, NumericInput, Paragraph, Spacer
from bokeh.layouts import column, row, layout

# custom files
from util.controls import DiscreteLogSliderWithInput
from svc.data_generation import generate_classification_dataset
from svc.confusion_matrix import update_confusion_matrix_plot, create_confusion_matrix_plot
from svc.classification_report import update_classification_report_plot, create_classification_report_plot
from svc.decision_boundary import update_decision_boundary_plot, create_decision_boundary_plot


def create_svc_layout():
    # --- UI elements ---
    class_radio = RadioButtonGroup(labels=["2 Classes", "3 Classes", "4 Classes"], active=2)
    decision_shape_radio = RadioButtonGroup(labels=["ovo", "ovr"], active=1)
    kernel_radio = RadioButtonGroup(labels=["linear", "poly", "rbf", "sigmoid", "laplacian"], active=2)
    dataset_select = Select(value="classification", width=160, options=[
        "classification", "aniso", "blobs", "varied_density", "outliers_with_clusters", "star_cluster", "checkerboard",
        "concentric_rings", "ball",
        "moons", "wavy_clusters", "s_curves",
        "spiral", "multiple_spirals", "multiarm_spiral"])
    generate_button = Button(label="Generate New Dataset", button_type="primary")
    n_samples_input = NumericInput(value=300, low=10, high=1000)

    # --- SVM parameter inputs ---
    default_params = sklearn.svm.SVC().get_params()
    cost_input = DiscreteLogSliderWithInput("cost:", initial_value=default_params["C"])
    degree_input = DiscreteLogSliderWithInput("degree:", values=[1, 2, 3, 4, 5, 6, 7], initial_value=default_params["degree"], low=1)
    gamma_radio = RadioButtonGroup(labels=["scale", "auto", "float"], active=0)
    gamma_input = DiscreteLogSliderWithInput("gamma-value::", initial_value=0.01, low=1e-10)
    coef0_input = DiscreteLogSliderWithInput("coef0:", values=np.arange(-4.5, 5, 0.5).tolist(), initial_value=default_params["coef0"])

    # --- generate initial dataset ---
    X, y = generate_classification_dataset(dataset_select.value, n_samples_input.value)

    def remap_classes(y, num_classes):
        """Remap the classes dynamically from origianlly 4 classes to 3 or 2 classes."""
        y_mapped = np.copy(y)
        if num_classes == 3:
            y_mapped[y == 3] = 2
        elif num_classes == 2:
            y_mapped[y == 3] = 1
            y_mapped[y == 2] = 0
        return y_mapped

    def get_current_n_classes():
        """Get the current number of classes based on the RadioGroup selection."""
        return class_radio.active + 2

    def get_params_from_current_options():
        """Return a dictionary containing the currently selected options from the RadioGroups and NumericInputs."""
        return {
            "C": cost_input.value,
            "degree": degree_input.value,
            "gamma": gamma_input.value if gamma_radio.active == 2 else gamma_radio.labels[gamma_radio.active],
            "coef0": coef0_input.value,
            "kernel": kernel_radio.labels[kernel_radio.active],
            "decision_function_shape": decision_shape_radio.labels[decision_shape_radio.active]
        }

    def enable_disable_inputs(current_params):
        """Depending on the provided parameters, disable or enable the input fields."""
        degree_input.disabled = current_params["kernel"] != "poly"
        gamma_radio.disabled = current_params["kernel"] == "linear"
        gamma_input.disabled = current_params["kernel"] == "linear" or gamma_radio.active != 2
        coef0_input.disabled = current_params["kernel"] not in ["poly", "sigmoid"]

    class SVCModel:
        """A wrapper class around an arbitrary SVC model. Creates a model and fits it to the data."""

        def __init__(self, model, X_train, y_train):
            """Create a new SVC model and fit the provided training data."""
            # create and fit the model
            tstart = time.time()
            self.model = model
            self.model.fit(X_train, y_train)
            tend = time.time()
            # store the time taken to train the model
            self.time = (tend - tstart) * 1000.0

    def train_models(svm_params, X_train, y_train):
        """Train the SVMs and compute the decision boundaries."""
        # train plssvm.SVC
        trained_plssvm_model = SVCModel(plssvm.SVC(**svm_params), X_train, y_train)

        # if the laplacian kernel is selected, we can't train the sklearn model
        if svm_params["kernel"] == "laplacian":
            return None, trained_plssvm_model

        # train sklearn.svm.SVC
        trained_sklearn_model = SVCModel(sklearn.svm.SVC(**svm_params), X_train, y_train)

        return trained_sklearn_model, trained_plssvm_model

    # --- initial model training ---
    initial_params = get_params_from_current_options()
    enable_disable_inputs(initial_params)
    sklearn_model, plssvm_model = train_models(initial_params, X, y)

    # --- create the initial plots ---
    sklearn_decision_boundary_fig, sklearn_decision_boundary, sklearn_decision_boundary_source, sklearn_data_source = create_decision_boundary_plot("sklearn.svm.SVC",
                                                                                                                                                    sklearn_model, X, y)
    sklearn_pred = sklearn_model.model.predict(X)
    sklearn_confusion_matrix, sklearn_confusion_matrix_source = create_confusion_matrix_plot(y, sklearn_pred)
    sklearn_classification_report, sklearn_classification_report_source = create_classification_report_plot(y, sklearn_pred)
    sklearn_text = Div(text=f"score: {sklearn_model.model.score(X, y) * 100:.2f}%<br>runtime: {sklearn_model.time:.2f}ms", styles={'font-size': '16px', 'color': 'black'})

    plssvm_decision_boundary_fig, plssvm_decision_boundary, plssvm_decision_boundary_source, plssvm_data_source = create_decision_boundary_plot("plssvm.SVC",
                                                                                                                                                plssvm_model, X, y)
    plssvm_pred = plssvm_model.model.predict(X)
    plssvm_confusion_matrix, plssvm_confusion_matrix_source = create_confusion_matrix_plot(y, plssvm_pred)
    plssvm_classification_report, plssvm_classification_report_source = create_classification_report_plot(y, plssvm_pred)
    plssvm_text = Div(text=f"score: {plssvm_model.model.score(X, y) * 100:.2f}%<br>runtime: {plssvm_model.time:.2f}ms", styles={'font-size': '16px', 'color': 'black'})

    def update():
        """The update function called whenever an input option changes."""
        # get the current classes and map the used labels accordingly
        current_classes = get_current_n_classes()
        y_mapped = remap_classes(y, current_classes)

        # retrain the models
        params = get_params_from_current_options()
        enable_disable_inputs(params)
        sklearn_model, plssvm_model = train_models(params, X, y_mapped)

        update_decision_boundary_plot(sklearn_model, sklearn_decision_boundary, sklearn_decision_boundary_source, sklearn_data_source, X, y_mapped)
        # if the sklearn model could be trained, update the plots, otherwise hide the sklearn plot
        if sklearn_model is None:
            # sklearn_model not trained (kernel = "laplacian")
            sklearn_confusion_matrix_source.data["alpha"] = [0] * len(sklearn_confusion_matrix_source.data["value"])
            sklearn_classification_report_source.data = pd.DataFrame()
            sklearn_text.text = "score: -<br>runtime: -"
        else:
            # sklearn_model is trained
            sklearn_pred = sklearn_model.model.predict(X)
            update_confusion_matrix_plot(sklearn_confusion_matrix, sklearn_confusion_matrix_source, y_mapped, sklearn_pred)
            update_classification_report_plot(sklearn_classification_report_source, y_mapped, sklearn_pred)
            sklearn_text.text = f"score: {accuracy_score(y_mapped, sklearn_pred) * 100:.2f}%<br>runtime: {sklearn_model.time:.2f}ms"

        # update the plssvm plot
        plssvm_pred = plssvm_model.model.predict(X)
        update_decision_boundary_plot(plssvm_model, plssvm_decision_boundary, plssvm_decision_boundary_source, plssvm_data_source, X, y_mapped)
        update_confusion_matrix_plot(plssvm_confusion_matrix, plssvm_confusion_matrix_source, y_mapped, plssvm_pred)
        update_classification_report_plot(plssvm_classification_report_source, y_mapped, plssvm_pred)
        plssvm_text.text = f"score: {accuracy_score(y_mapped, plssvm_pred) * 100:.2f}%<br>runtime: {plssvm_model.time:.2f}ms"

    def generate_and_update():
        """Generate new dataset, then update the plots."""
        nonlocal X, y
        X, y = generate_classification_dataset(dataset_select.value, n_samples_input.value)
        update()

    # --- register callbacks ---
    class_radio.on_change('active', lambda attr, old, new: update())
    decision_shape_radio.on_change('active', lambda attr, old, new: update())
    kernel_radio.on_change('active', lambda attr, old, new: update())

    cost_input.input_field.on_change('value', lambda attr, old, new: update())
    degree_input.input_field.on_change('value', lambda attr, old, new: update())
    gamma_radio.on_change('active', lambda attr, old, new: update())
    gamma_input.input_field.on_change('value', lambda attr, old, new: update())
    coef0_input.input_field.on_change('value', lambda attr, old, new: update())

    generate_button.on_click(lambda: generate_and_update())

    # --- create the layout ---
    return layout(
        column(
            row(sklearn_decision_boundary_fig,
                Spacer(width=15),
                column(sklearn_confusion_matrix, sklearn_classification_report, sklearn_text, spacing=15),
                Spacer(width=20),
                Div(text="<hr style='border: 1px solid black; height: 580px;'>"),
                Spacer(width=20),
                plssvm_decision_boundary_fig,
                Spacer(width=15),
                column(plssvm_confusion_matrix, plssvm_classification_report, plssvm_text, spacing=15), sizing_mode='stretch_width'),
            Div(text="<hr style='border: 1px solid black; width: 97.5vw;'>"),
            row(
                column(Paragraph(text="#classes:"), class_radio),
                column(Paragraph(text="kernel:"), kernel_radio),
                column(Paragraph(text="decision_function_shape:"), decision_shape_radio),
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
                column(Paragraph(text=""), generate_button)),
            spacing=5, sizing_mode='stretch_width'),
        margin=(10, 20, 10, 20), sizing_mode='stretch_width')
