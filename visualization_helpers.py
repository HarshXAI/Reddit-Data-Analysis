import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

def generate_time_series_insight(time_data: pd.DataFrame, time_agg: str) -> Tuple[str, Dict[str, Any]]:

    if time_data.empty:
        return "No time series data available for analysis.", {}

    max_date = time_data.loc[time_data['count'].idxmax(), 'date']
    max_count = time_data['count'].max()
    min_date = time_data.loc[time_data['count'].idxmin(), 'date']
    min_count = time_data['count'].min()
    avg_count = time_data['count'].mean()
    

    max_date_str = max_date.strftime('%Y-%m-%d') if isinstance(max_date, pd.Timestamp) else str(max_date)
    min_date_str = min_date.strftime('%Y-%m-%d') if isinstance(min_date, pd.Timestamp) else str(min_date)

    if len(time_data) > 5:
        first_half_avg = time_data['count'].iloc[:len(time_data)//2].mean()
        second_half_avg = time_data['count'].iloc[len(time_data)//2:].mean()
        change_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        if change_percent > 5:
            trend_text = f"increasing (by {change_percent:.1f}%)"
        elif change_percent < -5:
            trend_text = f"decreasing (by {abs(change_percent):.1f}%)"
        else:
            trend_text = "stable"
    else:
        trend_text = "not determinable due to limited data points"
        change_percent = 0
    

    insight = f"""
    **Interpretation:** This time series shows the frequency of Reddit posts by {time_agg.lower()}. 
    Peak activity occurred on **{max_date_str}** with **{max_count} posts**, while the lowest activity was **{min_count} posts** on **{min_date_str}**.
    The overall trend appears to be {trend_text} over this time period, with an average of {avg_count:.1f} posts per {time_agg.lower()}.
    """
    
    metrics = {
        'max_date': max_date,
        'max_count': max_count,
        'min_date': min_date,
        'min_count': min_count,
        'avg_count': avg_count,
        'change_percent': change_percent,
        'trend_text': trend_text
    }
    
    return insight, metrics

def generate_distribution_insight(df: pd.DataFrame, column: str, title: str = "distribution") -> str:
    if column not in df.columns:
        return f"No data available for {title} analysis."
    
    series = df[column]
    avg_value = series.mean()
    median_value = series.median()
    max_value = series.max()
    min_value = series.min()
    
    # Determine skew
    skew_value = series.skew()
    if skew_value > 0.5:
        skew_description = "right-skewed (most values are lower with a few high outliers)"
        pattern_explanation = "This pattern is typical where a small percentage receives significant attention."
    elif skew_value < -0.5:
        skew_description = "left-skewed (most values are higher with a few low outliers)"
        pattern_explanation = "This unusual pattern suggests most content receives higher-than-average attention."
    else:
        skew_description = "relatively balanced around the average"
        pattern_explanation = "This suggests more uniform distribution than typically seen in social media data."
    
    insight = f"""
    **Interpretation:** This histogram displays the {title} distribution, which is {skew_description}. 
    The average is **{avg_value:.1f}** (median: {median_value:.1f}) with a range from {min_value} to {max_value}. 
    {pattern_explanation}
    """
    
    return insight

def generate_category_insight(category_data: pd.DataFrame, 
                             count_col: str = 'count', 
                             category_col: str = 'category',
                             description: str = "categories") -> str:
    if category_data.empty:
        return f"No {description} data available for analysis."
    
    # Extract key metrics
    total_count = category_data[count_col].sum()
    top_category = category_data.iloc[0][category_col]
    top_count = category_data.iloc[0][count_col]
    top_percent = (top_count / total_count) * 100
    
    # Calculate concentration metrics
    top3_percent = (category_data.head(3)[count_col].sum() / total_count) * 100
    
    # Determine pattern description
    if top_percent > 30:
        distribution = f"heavily dominated by {top_category}"
        implication = "showing a strong concentration of attention"
    elif top3_percent > 60:
        distribution = "concentrated among a few top categories"
        implication = "suggesting a narrow focus of interest"
    else:
        distribution = "relatively evenly distributed"
        implication = "indicating diverse interests across multiple topics"
    
    insight = f"""
    **Interpretation:** This graph shows the distribution across {description}. 
    **{top_category}** is the most frequent with **{top_count:,}** ({top_percent:.1f}% of total).
    The distribution is {distribution}, {implication}.
    """
    
    return insight

def render_insight_box(insight_text: str) -> None:
    st.markdown(insight_text)

def render_metric_card(title, value):
    """Render a metric using plain text."""
    st.markdown(f"**{title}**")
    st.markdown(value)

def render_custom_insight_box(insight_text: str, title: str = "Insight", icon: str = "💡") -> None:
    st.markdown(f"**{icon} {title}**")
    st.markdown(insight_text)