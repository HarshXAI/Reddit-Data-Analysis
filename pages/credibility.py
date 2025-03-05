import streamlit as st
import plotly.express as px
from visualization_helpers import render_metric_card

def render_credibility_meter(score, factors):
    if score >= 70:
        color = "#4caf50"  
    elif score >= 40:
        color = "#ff9800"  
    else:
        color = "#f44336" 
        
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between;">
            <span>Low Credibility</span>
            <span>High Credibility</span>
        </div>
        <div style="height: 8px; width: 100%; background-color: #e9ecef; border-radius: 4px; overflow: hidden; margin: 4px 0;">
            <div style="height: 100%; width: {score}%; background-color: {color}; border-radius: 4px;"></div>
        </div>
        <div style="text-align: center; margin-top: 0.5rem;">
            <span style="font-weight: 600; color: {color};">Score: {score}/100</span>
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem;">
            <b>Factors:</b> {factors if factors else "No specific factors detected"}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render(df, advanced_agent):
    st.header("Content Credibility Analysis")
    
    with st.spinner("Analyzing content credibility..."):
        credibility_df = advanced_agent.score_credibility()
        
        if 'error' in credibility_df.columns:
            st.error(f"Error in credibility analysis: {credibility_df['error'].iloc[0]}")
        else:
            col1, col2, col3 = st.columns(3)
            
            avg_score = credibility_df['credibility_score'].mean()
            with col1:
                render_metric_card("Average Credibility Score", f"{avg_score:.1f}/100")
            
            high_cred = (credibility_df['credibility_score'] >= 70).sum()
            high_percent = (high_cred / len(credibility_df)) * 100
            with col2:
                render_metric_card("High Credibility Posts", f"{high_cred} ({high_percent:.1f}%)")
            
            low_cred = (credibility_df['credibility_score'] < 40).sum()
            low_percent = (low_cred / len(credibility_df)) * 100
            with col3:
                render_metric_card("Low Credibility Posts", f"{low_cred} ({low_percent:.1f}%)")
            
            # Display credibility distribution
            st.subheader("Credibility Score Distribution")
            
            fig = px.histogram(
                credibility_df,
                x="credibility_score",
                nbins=20,
                color_discrete_sequence=["#1e88e5"]
            )
            fig.update_layout(
                xaxis_title="Credibility Score",
                yaxis_title="Number of Posts",
                xaxis=dict(range=[0, 100]),
            )
            fig.add_shape(
                type="line", line=dict(dash="dash", color="red"),
                x0=40, x1=40, y0=0, y1=1, yref="paper"
            )
            fig.add_shape(
                type="line", line=dict(dash="dash", color="green"),
                x0=70, x1=70, y0=0, y1=1, yref="paper"
            )
            fig.add_annotation(x=20, y=0.95, yref="paper", text="Low Credibility", showarrow=False)
            fig.add_annotation(x=55, y=0.95, yref="paper", text="Medium Credibility", showarrow=False)
            fig.add_annotation(x=85, y=0.95, yref="paper", text="High Credibility", showarrow=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Posts with Low Credibility Scores")
            
            low_cred_posts = credibility_df.sort_values('credibility_score').head(5)
            
            for _, post in low_cred_posts.iterrows():
                with st.expander(f"{post['title']}"):
                    st.markdown(f"**Subreddit**: r/{post['subreddit']}")
                    st.markdown(f"**Credibility Score**: {post['credibility_score']}/100")
                    
                    render_credibility_meter(
                        int(post['credibility_score']),
                        post['credibility_factors']
                    )
            with st.expander("How credibility scores are calculated"):
                st.markdown("""
                ### Credibility Scoring Methodology
                
                Posts are analyzed based on several factors:
                
                1. **Source Credibility**: Links to reputable sources increase score, while suspicious domains lower it
                2. **Content Analysis**: Posts with misinformation patterns or conspiracy terms receive lower scores
                3. **Presentation Style**: Excessive capitalization, punctuation, or alarmist language lowers credibility
                4. **User History**: Author's historical credibility impacts score (when available)
                
                Scores range from 0-100, with higher scores indicating higher credibility.
                """)
