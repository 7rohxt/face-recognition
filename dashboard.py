import plotly.express as px
import pandas as pd
from firebase_configure import ref

def show_attendance_dashboard():

    data = ref.get()

    if not data:
        print("No data found in Firebase.")
        return

    names = []
    attendance_counts = []

    for emp_id, details in data.items():
        names.append(details.get("name", emp_id))
        attendance_counts.append(details.get("total_attendance", 0))

    df = pd.DataFrame({
        'Employee': names,
        'Total Attendance': attendance_counts
    })

    blue_white_scale = [
        [0, '#e0f7ff'],   
        [0.5, '#74b9ff'],
        [1.0, '#1e3799']  
    ]

    fig = px.bar(
        df,
        x='Employee',
        y='Total Attendance',
        color='Total Attendance',
        color_continuous_scale=blue_white_scale,
        title='Employee Attendance Overview',
        labels={'Total Attendance': 'Days Present'},
        hover_data={'Employee': True, 'Total Attendance': True}
    )

    fig.update_layout(
        xaxis_title="Employee",
        yaxis_title="Total Attendance Days",
        title_font_size=24,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=100),
        font=dict(color='#2d3436'),
        height=600
    )

    fig.show()

show_attendance_dashboard()