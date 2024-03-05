import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to visualize features
def visualize_features(df, feature):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Function to visualize outliers
def visualize_outliers(df, feature):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=feature, data=df, ax=ax)
    ax.set_title(f'Boxplot of {feature}')
    ax.set_xlabel(feature)
    st.pyplot(fig)

def main():
    st.title('Machine Learning Project Dashboard')
    st.sidebar.title('Navigation')

    # Load data
    file_path = 'train.csv'
    df = load_data(file_path)

    # Sidebar options
    option = st.sidebar.selectbox('Select Option', ['Feature Visualization', 'Outlier Visualization'])

    # if option == 'Overview':
    #     st.write(df.head())

    if option == 'Feature Visualization':
        st.subheader('Feature Visualization')

        # Dropdown to select feature
        feature_to_visualize = st.selectbox('Select Feature', df.columns)

        # Visualize selected feature
        visualize_features(df, feature_to_visualize)
        st.image('visulization_featureSelect.png', caption='Feature Selection Visualization', use_column_width=True)

    elif option == 'Outlier Visualization':
        st.subheader('Outlier Visualization')

        # Dropdown to select feature
        feature_to_visualize = st.selectbox('Select Feature', df.columns)

        # Visualize outliers for selected feature
        visualize_outliers(df, feature_to_visualize)
        st.image('visulization_outlier.png', caption='Outlier Visualization', use_column_width=True)


if __name__ == '__main__':
    main()
