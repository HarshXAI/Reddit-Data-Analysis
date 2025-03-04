import streamlit as st

# Tab indices for navigation
TAB_INDICES = {
    "overview": 0,
    "time_series": 1,
    "text_analysis": 2,
    "advanced_topics": 3,
    "credibility": 4,
    "ai_insights": 5
}

def switch_tab(tab_name):
    """
    Switch to the specified tab by updating session state.
    
    Args:
        tab_name: Name of the tab to switch to
    """
    if tab_name in TAB_INDICES:
        st.session_state.active_tab = TAB_INDICES[tab_name]

def create_tab_links():
    """
    Create HTML links for tab navigation.
    Returns HTML code for tab links that can be used anywhere.
    """
    tabs = [
        ("Overview & Stats", "overview"),
        ("Time Series Analysis", "time_series"),
        ("Text Analysis", "text_analysis"),
        ("Advanced Topics", "advanced_topics"),
        ("Credibility Analysis", "credibility"),
        ("AI Insights", "ai_insights")
    ]
    
    links_html = '<div style="display: flex; gap: 10px; margin-bottom: 20px;">'
    for tab_title, tab_id in tabs:
        # Create a button that will trigger a form submission
        links_html += f"""
        <form action="javascript:void(0);" onsubmit="this.btnSubmit.click()">
            <input type="hidden" name="tab" value="{tab_id}">
            <button type="submit" 
                style="background: {'#1E90FF' if st.session_state.active_tab == TAB_INDICES[tab_id] else '#f0f2f6'}; 
                       color: {'white' if st.session_state.active_tab == TAB_INDICES[tab_id] else '#31333F'}; 
                       border: none; 
                       padding: 5px 15px; 
                       border-radius: 5px; 
                       cursor: pointer;"
                name="btnSubmit" 
                onclick="window.parent.postMessage({{tab: '{tab_id}'}}, '*')">
                {tab_title}
            </button>
        </form>
        """
    links_html += '</div>'
    
    # Add JavaScript to handle the message
    links_html += """
    <script>
        // Create a callback function that changes the query parameter
        window.addEventListener('message', function(event) {
            if (event.data && event.data.tab) {
                const tab = event.data.tab;
                const url = new URL(window.location);
                url.searchParams.set('tab', tab);
                window.history.pushState({}, '', url);
                window.location.reload();
            }
        });
    </script>
    """
    
    return links_html
