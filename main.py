import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit config
st.set_page_config(page_title="Social Media Trend Analysis", layout="wide")

# Load dataset and pipeline
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_social media influencers - instagram.csv")

@st.cache_resource
def load_pipeline():
    return joblib.load("logistic_regression_model.pkl")  # combined preprocessor + model

df = load_data()
pipeline = load_pipeline()

# Sidebar
with st.sidebar:
    st.title("📌 Dashboard")
    st.markdown("Navigate to explore data or predict engagement.")
    st.info("Built with ❤️ using Streamlit")

tab1, tab2 = st.tabs(["📊 Data Insights", "🤖 Predict Engagement"])

# -------------------------
# TAB 1: Data Insights
# -------------------------
with tab1:
    st.header("📊 Instagram Influencer Analysis")

    st.sidebar.subheader("🔍 Filter Influencers")

    selected_categories = st.sidebar.multiselect(
        "Select Category(s)", sorted(df['Category'].dropna().unique()), default=None)

    selected_countries = st.sidebar.multiselect(
        "Select Top Audience Country(s)", sorted(df['Top audience country'].dropna().unique()), default=None)

    follower_min, follower_max = st.sidebar.slider(
        "Follower Range", min_value=int(df["Followers"].min()), max_value=int(df["Followers"].max()),
        value=(int(df["Followers"].min()), int(df["Followers"].max()))
    )

    filtered_df = df[
        (df["Followers"] >= follower_min) &
        (df["Followers"] <= follower_max)
    ]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
    if selected_countries:
        filtered_df = filtered_df[filtered_df["Top audience country"].isin(selected_countries)]

    with st.expander("📄 View Filtered Data"):
        st.dataframe(filtered_df.head(20), use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode()
    st.download_button("📥 Download Filtered Data", csv, "filtered_influencers.csv", "text/csv")

    st.subheader("🏷️ Top Categories")
    top_cat = filtered_df['Category'].value_counts().head(10).reset_index()
    top_cat.columns = ['Category', 'Count']
    fig1 = px.bar(top_cat, x='Count', y='Category', orientation='h', color='Count',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("🌍 Top Countries")
    top_country = filtered_df['Top audience country'].value_counts().head(10).reset_index()
    top_country.columns = ['Country', 'Count']
    fig2 = px.bar(top_country, x='Count', y='Country', orientation='h', color='Count',
                  color_continuous_scale='Greens')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📍 Followers vs Engagement Average")
    fig3 = px.scatter(
        filtered_df,
        x='Followers',
        y='Engagement average',
        size='Authentic engagement',
        color='Category',
        hover_name='Influencer insta name',
        title='Followers vs Engagement',
        size_max=60
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📈 Engagement Metric Correlations")
    numeric_cols = ['Followers', 'Authentic engagement', 'Engagement average']
    if not filtered_df[numeric_cols].empty:
        corr = filtered_df[numeric_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)
    else:
        st.warning("Not enough data for correlation heatmap.")

# -------------------------
# TAB 2: Engagement Prediction
# -------------------------
with tab2:
    st.header("🎯 Predict Engagement Level")
    st.markdown("Fill in the influencer details to predict engagement level:")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            influencer_name = st.selectbox("Influencer insta name (info only)", sorted(df['Influencer insta name'].dropna().unique()))
            followers = st.number_input("Total Followers", min_value=0.0, value=100000.0, step=10000.0)
            authentic_engagement = st.number_input("Authentic Engagement", min_value=0.0, value=5000.0, step=500.0)
        with col2:
            engagement_avg = st.number_input("Engagement Average", min_value=0.0, value=8000.0, step=500.0)
            category = st.selectbox("Category", sorted(df['Category'].dropna().unique()))
            country = st.selectbox("Top Audience Country", sorted(df['Top audience country'].dropna().unique()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            input_dict = {
                'Followers': followers,
                'Authentic engagement': authentic_engagement,
                'Engagement average': engagement_avg,
                'Category': category,
                'Top audience country': country
            }

            input_df = pd.DataFrame([input_dict])

            pred = pipeline.predict(input_df)[0]
            prob = pipeline.predict_proba(input_df).max()

            st.success(f"📌 Predicted Engagement Level: **{pred}**")
            st.info(f"Model Confidence: **{prob:.2%}**")

        except Exception as e:
            st.error(f"Prediction Failed: {e}")
