import streamlit as st
import plotly.express as px
import pandas as pd
from visualization_helpers import render_insight_box, generate_time_series_insight

def render(df, stats_agent):
    st.header("Time Series Analysis")
    
    posts_over_time = stats_agent.get_posts_over_time()
    
    time_agg = st.selectbox(
        "Time Aggregation",
        ["Day", "Week", "Month"],
        index=0
    )
    
    if time_agg == "Day":
        time_data = posts_over_time['day']
    elif time_agg == "Week":
        time_data = posts_over_time['week']
    else:
        time_data = posts_over_time['month']

    if not time_data.empty:
        max_date = time_data.loc[time_data['count'].idxmax(), 'date']
        max_count = time_data['count'].max()
        
        st.metric(
            "Peak Activity", 
            f"{max_count} posts", 
            f"on {max_date.strftime('%Y-%m-%d') if isinstance(max_date, pd.Timestamp) else max_date}"
        )

    fig = px.line(
        time_data,
        x="date",
        y="count",
        title=f"Posts by {time_agg}",
        markers=True
    )
    fig.update_traces(
        line=dict(color="#1e88e5", width=2),
        marker=dict(size=6, color="#1e88e5")
    )
    fig.update_layout(
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    insight_text, _ = generate_time_series_insight(time_data, time_agg)
    render_insight_box(insight_text)
    
    col1, col2 = st.columns(2)
    
    with col1:
        dow_data = stats_agent.get_posts_by_day_of_week()
        most_active_day = dow_data.loc[dow_data['count'].idxmax(), 'day_name']
        most_active_count = dow_data['count'].max()
        least_active_day = dow_data.loc[dow_data['count'].idxmin(), 'day_name']
        least_active_count = dow_data['count'].min()
        
        fig = px.bar(
            dow_data,
            x="day_name",
            y="count",
            title="Posts by Day of Week",
            color="count",
            color_continuous_scale="Blues"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        
        weekday_weekend_pattern = "higher on weekdays than weekends" if dow_data.loc[dow_data['day_name'].isin(['Saturday', 'Sunday']), 'count'].mean() < dow_data.loc[~dow_data['day_name'].isin(['Saturday', 'Sunday']), 'count'].mean() else "higher on weekends than weekdays"
        
        insight_text = f"""
        **Interpretation:** This chart shows posting activity by day of week. 
        **{most_active_day}** is the most active day with **{most_active_count} posts**, while **{least_active_day}** has the lowest activity (**{least_active_count} posts**).
        Activity is generally {weekday_weekend_pattern}, suggesting {'users are more engaged during the work week' if weekday_weekend_pattern == 'higher on weekdays than weekends' else 'users are more active during leisure time'}.
        """
        render_insight_box(insight_text)
    
    with col2:
        hour_data = stats_agent.get_posts_by_hour()
        peak_hour = hour_data.loc[hour_data['count'].idxmax(), 'hour']
        peak_hour_count = hour_data['count'].max()
        quiet_hour = hour_data.loc[hour_data['count'].idxmin(), 'hour']
        quiet_hour_count = hour_data['count'].min()
        
        fig = px.line(
            hour_data,
            x="hour",
            y="count",
            title="Posts by Hour of Day",
            markers=True
        )
        fig.update_traces(
            line=dict(color="#1e88e5", width=2),
            marker=dict(size=6)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        peak_hour_12h = f"{peak_hour if peak_hour <= 12 else peak_hour - 12} {'AM' if peak_hour < 12 or peak_hour == 24 else 'PM'}"
        quiet_hour_12h = f"{quiet_hour if quiet_hour <= 12 else quiet_hour - 12} {'AM' if quiet_hour < 12 or quiet_hour == 24 else 'PM'}"
        
        morning_peak = hour_data[(hour_data['hour'] >= 6) & (hour_data['hour'] < 12)]['count'].max()
        evening_peak = hour_data[(hour_data['hour'] >= 17) & (hour_data['hour'] < 23)]['count'].max()
        peak_pattern = "both morning and evening peaks" if morning_peak > 0.7 * evening_peak and evening_peak > 0.7 * morning_peak else "primarily evening activity" if evening_peak > morning_peak else "primarily morning activity"
        
        insight_text = f"""
        **Interpretation:** This graph shows posting activity throughout the day. 
        Peak activity occurs around **{peak_hour_12h}** (**{peak_hour_count} posts**), with the least activity at **{quiet_hour_12h}** (**{quiet_hour_count} posts**).
        The pattern shows {peak_pattern}, suggesting {'users engage both before and after work hours' if peak_pattern == 'both morning and evening peaks' else 'users are most active during evening leisure hours' if peak_pattern == 'primarily evening activity' else 'users tend to post early in the day'}.
        """
        render_insight_box(insight_text)
    st.markdown("""
    <style>
    /* Reset any custom styling that might affect text color */
    .stMarkdown {
        color: inherit !important;
    }
    </style>
    """, unsafe_allow_html=True)
