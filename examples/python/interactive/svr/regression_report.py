import pandas as pd
import plssvm

from bokeh.models import ColumnDataSource, DataTable, TableColumn, HTMLTemplateFormatter


def regression_report_as_dataframe(y_true, y_pred):
    """Compute the regression report using y_true and y_pred and convert it to a Pandas DataFrame usable in bokeh."""
    # calculate the regression report
    report_dict = plssvm.regression_report(y_true, y_pred, output_dict=True)

    # convert to DataFrame
    df = pd.DataFrame.from_dict(report_dict, orient="index", columns=["value"])

    # convert numeric values and round to 3 decimal places
    df["value"] = df["value"].apply(pd.to_numeric, errors="coerce").round(3)

    # format text for display
    df["value"] = df["value"].astype(str)
    df.at['r2_score', 'value'] = f"<b>{df.at['r2_score', 'value']}</b>"

    df = df.reset_index().rename(columns={"index": ""})

    return df


def update_regression_report_plot(source, y_true, y_pred):
    """Update the already existing regression report table using y_true and y_pred."""
    source.data = regression_report_as_dataframe(y_true, y_pred)


def create_regression_report_plot(y_true, y_pred):
    """Create a new regression report table using y_true and y_pred."""
    # create the Pandas DataFrame representing a regression report
    df = regression_report_as_dataframe(y_true, y_pred)

    # convert DataFrame to ColumnDataSource
    source = ColumnDataSource(df)

    # define the HTML formatter for the 'Name' column
    name_formatter = HTMLTemplateFormatter(template='<div><%= value %></div>')
    # create table columns (hide "Class" header by setting title to "")
    total_width = 300
    main_column_width = 250  # baded on largest string in column
    minor_column_width = 50
    columns = [TableColumn(field=col, title=col, width=main_column_width if col == "" else minor_column_width, formatter=name_formatter) for col in df.columns]

    # Create DataTable
    regression_table = DataTable(source=source, columns=columns, index_position=None, sizing_mode='fixed', width=total_width, height=200)

    return regression_table, source
