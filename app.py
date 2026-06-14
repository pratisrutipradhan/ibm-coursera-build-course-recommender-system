import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration to wide mode for an expansive dashboard feel
st.set_page_config(page_title="Course Recommender Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS styling to make course cards pop beautifully
st.markdown("""
    <style>
    .course-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #4F81BD;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# DATA LOADING UTILITY
# =================================================================
@st.cache_data
def load_course_inventory():
    # Full catalog matching your Capstone project tracks
    courses = {
        'COURSE_ID': [0, 1, 2, 4, 5, 6, 7, 10, 11, 17],
        'Title': [
            'Python for Data Science', 'Introduction to Data Science', 'Big Data 101',
            'Data Analysis with Python', 'Data Science Methodology', 'Machine Learning with Python',
            'Spark Fundamentals I', 'Data Visualization with Python', 'Deep Learning 101',
            'SQL and Relational Databases 101'
        ],
        'Python':   [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        'ML':       [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        'Analysis': [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
        'Database': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'BigData':  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    }
    return pd.DataFrame(courses)

df_courses = load_course_inventory()
feature_cols = ['Python', 'ML', 'Analysis', 'Database', 'BigData']

# =================================================================
# SIDEBAR CONTROL PANEL
# =================================================================
st.sidebar.header("👤 Learner Profile Builder")
st.sidebar.write("Select topics you have already mastered:")

# User background checkbox inputs
user_inputs = {}
for col in feature_cols:
    user_inputs[col] = st.sidebar.checkbox(f"Completed {col} Track", value=False)

# Convert selections into a clear 2D mathematical vector
user_vector = np.array([1 if user_inputs[c] else 0 for c in feature_cols]).reshape(1, -1)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Architecture Settings")
sim_threshold = st.sidebar.slider("Minimum Similarity Threshold Score", 0.1, 0.9, 0.4, 0.1)
top_n = st.sidebar.slider("Max Cards to Render", 1, 5, 3)

# =================================================================
# APP HEADER LAYOUT
# =================================================================
st.title("🎓 Course Recommender Engine Dashboard")
st.write("An Explainable AI (XAI) approach to content-based similarity matching using **Vector Space Dot Products**.")

# Check for a cold-start state (no features selected yet)
if user_vector.sum() == 0:
    st.info("💡 **Welcome to your profile builder!** Check one or more background skill tracks in the sidebar to generate your personal preference vector and initialize recommendations.")
else:
    # RUN MATHEMATICAL MODEL MATCHING
    course_matrix = df_courses[feature_cols].values
    scores = cosine_similarity(user_vector, course_matrix).flatten()
    
    # Pack data into an evaluation results matrix
    df_eval = df_courses.copy()
    df_eval['SimilarityScore'] = scores
    
    # Apply user-defined threshold filters
    df_filtered = df_eval[df_eval['SimilarityScore'] >= sim_threshold]
    df_recs = df_filtered.sort_values(by='SimilarityScore', ascending=False).head(top_n)

    # CREATE VIEW TABS
    tab1, tab2, tab3 = st.tabs(["🎯 Personalized Recommendations", "📊 Explainable AI Vectors", "🗂️ Master Repository Lookup"])

    # -------------------------------------------------------------
    # TAB 1: RECOMMENDATIONS RENDERING (COURSE CARDS)
    # -------------------------------------------------------------
    with tab1:
        if df_recs.empty:
            st.warning(f"⚠️ No uncompleted courses pass your minimum similarity constraint threshold of **{sim_threshold}**. Loosen the filter parameters in the sidebar panel.")
        else:
            st.subheader(f"🚀 Top {len(df_recs)} Recommended Tracks matched to your profile:")
            
            # Use columns to distribute cards side by side
            cols = st.columns(len(df_recs))
            for i, row in enumerate(df_recs.itertuples()):
                with cols[i]:
                    match_percent = int(row.SimilarityScore * 100)
                    
                    # HTML Container Injection for card aesthetics
                    st.markdown(f"""
                        <div class="course-card">
                            <h3>🟢 {match_percent}% Match</h3>
                            <p style="font-size:1.15rem; font-weight:bold; color:#1F2937;">{row.Title}</p>
                            <p style="color:#6B7280; font-size:0.85rem;">System Reference ID: <code>COURSE_ID {row.COURSE_ID}</code></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.button("Launch Syllabus & Enroll", key=f"btn_{row.COURSE_ID}")

    # -------------------------------------------------------------
    # TAB 2: EXPLAINABLE AI VISUALIZATIONS (PLOTLY RADAR GRAPH)
    # -------------------------------------------------------------
    with tab2:
        st.subheader("📊 Vector Overlap Analysis")
        st.write("See exactly why these courses are being surfaced. The chart below graphs your preference vector against the top recommendation's attributes.")
        
        if not df_recs.empty:
            top_rec_row = df_recs.iloc[0]
            
            # Setup Plotly Horizontal Bar Visualizer to compare attributes
            fig = go.Figure()
            
            # Add user profile attributes
            fig.add_trace(go.Bar(
                y=feature_cols,
                x=user_vector[0],
                name='Your Preference Profile',
                orientation='h',
                marker_color='#2C3E50',
                opacity=0.85
            ))
            
            # Add targeted course attributes
            fig.add_trace(go.Bar(
                y=feature_cols,
                x=[top_rec_row[c] for c in feature_cols],
                name=f"Course: {top_rec_row['Title']}",
                orientation='h',
                marker_color='#4F81BD',
                opacity=0.85
            ))
            
            fig.update_layout(
                barmode='group',
                title=dict(text="Feature Overlap Breakdown (User Profile vs. Top Recommendation)", font=dict(size=14)),
                xaxis=dict(title="Attribute Weight Baseline", tickvals=[0, 1]),
                yaxis=dict(title="Track Categories"),
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Extra analytical summary
            st.markdown(f"""
                > **Model Inference Breakdown:** **`{top_rec_row['Title']}`** reached a high cosine matching rank because of shared overlapping parameters in key domain tracks.
            """)

    # -------------------------------------------------------------
    # TAB 3: MASTER DATAFRAME LOOKUP
    # -------------------------------------------------------------
    with tab3:
        st.subheader("🗂️ Global Model Computation Matrix")
        st.write("This table shows the raw calculations across the entire inventory, sorted by total similarity score.")
        st.dataframe(
            df_eval.sort_values(by='SimilarityScore', ascending=False),
            use_container_width=True,
            hide_index=True
        )
