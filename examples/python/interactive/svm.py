from bokeh.models import TabPanel, Tabs
from bokeh.plotting import curdoc
from bokeh.layouts import column

from svr.svr import create_svr_layout
from svc.svc import create_svc_layout

# create tabs with layout-based panels
classification_tab = TabPanel(child=create_svc_layout(), title="Classification")
regression_tab = TabPanel(child=create_svr_layout(), title="Regression")

# Combine the tabs
tabs = Tabs(tabs=[classification_tab, regression_tab], styles={"font-size": "14pt"}, sizing_mode="stretch_both")

# add to document
curdoc().add_root(column(tabs, sizing_mode="stretch_both"))
curdoc().title = "sklearn vs PLSSVM"
