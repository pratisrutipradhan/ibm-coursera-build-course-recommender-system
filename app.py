import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set web page configurations
st.set_page_config(page_title="Course Recommender Engine", layout="wide", initial_sidebar_state="expanded")

# =================================================================
# 1. LOAD & PREPARE COURSE METADATA (VECTORS)
# =================================================================
@st.cache_data
def load_course_data():
    # Simulating the course inventory dataset mapped with binary genre feature vectors
    # Features: [Python, Machine Learning, Data Analysis, Databases, Big Data]
    courses = {
        'COURSE_ID': [0, 1, 2, 4, 5, 6, 7, 10, 11, 17],
        'Title': [
            'Python for Data Science',
            'Introduction to Data Science',
            'Big Data 101',
            'Data Analysis with Python',
            'Data Science Methodology',
            'Machine Learning with Python',
            'Spark Fundamentals I',
            'Data Visualization with Python',
            'Deep Learning 101',
            'SQL and Relational Databases 101'
        ],
        'Python':   [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        'ML':       [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        'Analysis': [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
        'Database': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'BigData':  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    }
    return pd.DataFrame(courses)

df_courses = load_course_data()
feature_cols = ['Python', 'ML', 'Analysis', 'Database', 'BigData']

# =================================================================
# 2. SIDEBAR - CONTROL PANEL & USER INTERACTION PROFILE
# =================================================================
st.sidebar.header("👤 User Learning Profile")
st.sidebar.write("Select the skills or topics you have already completed to build your interest profile:")

# Collect real-time training inputs from user interaction checkboxes
user_selections = {}
for feature in feature_cols:
    user_selections[feature] = st.sidebar.checkbox(f"Completed {feature} courses", value=False)

# Build the dynamic User Preference Vector based on active inputs
user_vector = np.array([1 if user_selections[f] else 0 for f in feature_cols]).reshape(1, -1)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Tuning Parameters")
# Similarity Threshold tuning slider
sim_threshold = st.sidebar.slider("Cosine Similarity Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
top_n = st.sidebar.slider("Maximum recommendations to show", min_value=1, max_value=5, value=3)

# =================================================================
# 3. RECOMMENDATION ENGINE LOGIC (Vector Space Modeling)
# =================================================================
st.title("🎓 IBM Machine Learning Capstone: Course Recommender")
st.write("This app uses **Content-Based Filtering via Cosine Similarity** to recommend courses matching your profile.")

# Ensure the user has checked at least one background skill box
if user_vector.sum() == 0:
    st.info("💡 Please select at least one completed skill category in the left sidebar panel to initialize your user vector profile.")
else:
    # Isolate the course feature space matrix
    course_matrix = df_courses[feature_cols].values
    
    # Mathematical Inference: Calculate cosine similarity between user vector and all courses
    similarity_scores = cosine_similarity(user_vector, course_matrix).flatten()
    
    # Append results back into a temporary evaluation matrix
    df_results = df_courses[['COURSE_ID', 'Title']].copy()
    df_results['Similarity Score'] = similarity_scores
    
    # Filter by user-defined absolute threshold constraints
    df_filtered = df_results[df_results['Similarity Score'] >= sim_threshold]
    
    # Sort in descending order to locate top matches
    df_final_recs = df_filtered.sort_values(by='Similarity Score', ascending=False).head(top_n)
    
    # =================================================================
    # 4. RENDER GRAPHICAL OUTPUT INTERFACE
    # =================================================================
    st.subheader("🚀 Top Personalized Course Matches for You")
    
    if df_final_recs.empty:
        st.warning(f"⚠️ No courses in the database meet your strict similarity threshold of **{sim_threshold}**. Try lowering the threshold bar in the sidebar.")
    else:
        # Loop through matches and render clean card elements using columns
        cols = st.columns(len(df_final_recs))
        for idx, row in enumerate(df_final_recs.itertuples()):
            with cols[idx]:
                st.metric(label=f"Match Score: {int(row._3 * 100)}%", value=f"ID: {row.COURSE_ID}")
                st.markdown(f"**{row.Title}**")
                st.caption("Recommended based on vector text alignment profile tags.")
                st.button("Enroll Now", key=f"btn_{row.COURSE_ID}")

    # Display underlying mathematical calculation dataset for peer-review inspection
    with st.expander("📊 Inspect Vector Matching Computation Matrix"):
        st.dataframe(
            df_results.sort_values(by='Similarity Score', ascending=False),
            use_container_width=True,
            hide_index=True
        )