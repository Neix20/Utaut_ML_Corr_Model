import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

def show_bar_graph(df, title = "", x_title = "", y_title = "", y_axis_title = ""):
    fig = go.Figure()
    
    for name in df.index:
        fig.add_trace(go.Bar(name = name, x = df.columns, y = df.loc[name, :]))
        
    fig.update_traces(
        texttemplate='%{value}', 
        textposition='outside'
    )
    
    fig.update_layout(
        title = {
            'text': title,
            'x':0.5,
            'xanchor': 'center'
        },
        xaxis_title = x_title,
        yaxis_title = y_axis_title,
        legend_title = y_title,
    )

    return fig

def show_bar_graph_percentage(df, title = "", x_title = "", y_title = "", y_axis_title = ""):
    fig = go.Figure()
    
    for name in df.index:
        fig.add_trace(go.Bar(name = name, x = df.columns, y = df.loc[name, :]))
        
    fig.update_traces(
        texttemplate='%{value:.2f}%', 
        textposition='outside'
    )
    
    fig.update_layout(
        title = {
            'text': title,
            'x':0.5,
            'xanchor': 'center'
        },
        xaxis_title = x_title,
        yaxis_title = y_axis_title,
        legend_title = y_title,
    )

    return fig

def cmp_result_tbl(clf_report_arr, col_name):
    name_arr, arr_list = [], []
    for name, clf_report, acc_score, mcc, time in clf_report_arr:
        arr_list.append(clf_report.loc[col_name.lower()][:-1].tolist() + [acc_score, mcc, time])
        name_arr.append(name)
        
    df = pd.DataFrame(arr_list, columns = ["Precision", "Recall", "F1-Score", "Accuracy Score", "MCC", "Time Taken"])
    df.index = name_arr
    
    return df

def get_df_type(clf_report_arr, col_name):
    df = pd.DataFrame()

    for name, clf_report in clf_report_arr:
        df[name] = clf_report[col_name.lower()][:-3]
    
    # Replace NA value with 0
    df.fillna(0, inplace=True)
    return df

def pfr_graph(df, x_label, y_label, title):
    fig = px.bar(df.T, x=df.columns, y=df.index, barmode="group")
    fig.update_traces(texttemplate='%{value:.4f}', textposition='outside')
    fig.update_layout(
        title = {
            'text': title,
            'x':0.5,
            'xanchor': 'center'
        },
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title="Legend",
    )
    fig.update_yaxes(range=[0, 1])
    return fig
    
def prc_roc_graph(model_arr, X, Y, x_cor, title, x_title, y_title, func):
    # Empty Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_cor, line=dict(dash='dash'), name="", showlegend=False))
    
    for name, model in model_arr:
        yhat = model.predict_proba(X)
        yhat = yhat[:, -1]
        a, b, c = func(Y, yhat)
        if x_cor == [1, 0]:
            x_val, y_val = b, a
        else:
            x_val, y_val = a, b
        fig.add_trace(
            go.Scatter(
                x = x_val, 
                y = y_val, 
                name = name,
                mode='lines'
            )
        )
        
    fig.update_layout(
        title = {
            'text': title,
            'x': 0.5,
            "y": 0.85,
            'xanchor': 'center'
        },
        xaxis_title = x_title,
        yaxis_title = y_title,
        legend_title = "Legend",
    )
    
    fig.update_xaxes(range=[0,1])
    fig.update_yaxes(range=[0,1])
    return fig
    
def acc_graph(acc_arr, title, x_title, y_title):
    # Empty Plots
    fig = go.Figure()
    
    for name, acc in acc_arr:
        fig.add_trace(
            go.Bar(
                x = [acc], 
                y = [name],
                name = name,
                orientation = 'h'
            )
        )
        
    fig.update_traces(
        texttemplate='%{value:.2f}%', 
        textposition='outside'
    )
    
    fig.update_layout(
        title = {
            'text': title,
            'x': 0.5,
            'y': 0.85,
            'xanchor': 'center'
        },
        xaxis_title = x_title,
        yaxis_title = y_title,
        legend_title = "Legend",
    )
    fig.update_xaxes(range=[0,100])
    return fig