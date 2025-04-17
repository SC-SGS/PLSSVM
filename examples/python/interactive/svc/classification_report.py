import pandas as pd
import sklearn

from bokeh.models import ColumnDataSource, DataTable, TableColumn, HTMLTemplateFormatter


def classification_report_as_dataframe(y_true, y_pred):
    """Compute the regression report using y_true and y_pred and convert it to a Pandas DataFrame usable in bokeh."""
    # calculate the classification report
    report_dict = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # convert to DataFrame
    df = pd.DataFrame(report_dict).transpose()

    # convert numeric values and round to 3 decimal places
    numeric_cols = ["precision", "recall", "f1-score"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").round(3)
    df["support"] = df["support"].astype(int)
    df.at['accuracy', 'support'] = len(y_true)

    # format text for display
    df["precision"] = df["precision"].astype(str)
    df["recall"] = df["recall"].astype(str)
    df["f1-score"] = df["f1-score"].astype(str)
    df.at['accuracy', 'precision'] = ""
    df.at['accuracy', 'recall'] = ""
    df.at['accuracy', 'f1-score'] = f"<b>{df.at['accuracy', 'f1-score']}</b>"

    df = df.reset_index().rename(columns={"index": ""})

    return df


def update_classification_report_plot(source, y_true, y_pred):
    """Update the already existing classification report table using y_true and y_pred."""
    source.data = classification_report_as_dataframe(y_true, y_pred)


def create_classification_report_plot(y_true, y_pred):
    """Create a new classification report table using y_true and y_pred."""
    # create the Pandas DataFrame representing a classification report
    df = classification_report_as_dataframe(y_true, y_pred)

    # convert DataFrame to ColumnDataSource
    source = ColumnDataSource(df)

    # define the HTML formatter for the 'Name' column
    name_formatter = HTMLTemplateFormatter(template='<div><%= value %></div>')
    # create table columns (hide "Class" header by setting title to "")
    total_width = 300
    main_column_width = 96  # baded on largest string in column
    minor_column_width = (total_width - main_column_width) // 4
    columns = [TableColumn(field=col, title=col, width=main_column_width if col == "" else minor_column_width, formatter=name_formatter) for col in df.columns]

    # Create DataTable
    classification_table = DataTable(source=source, columns=columns, index_position=None, sizing_mode='fixed', width=total_width, height=200)

    return classification_table, source
