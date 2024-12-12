import streamlit as st
import os
from PIL import Image

def display_results():
    image_path_1 = 'output/q_values.png'
    image_path_2 = 'output/states.png'
    video_path = 'output/video.mp4'

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(image_path_1):
            img = Image.open(image_path_1)
            st.image(img, caption='Image 1', use_container_width=True)
        else:
            st.write("Image 1 not found.")
    
    with col2:
        if os.path.exists(image_path_2):
            img = Image.open(image_path_2)
            st.image(img, caption='Image 2', use_container_width=True)
        else:
            st.write("Image 2 not found.")
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.write("Video not found.")

st.sidebar.title('Simulation Settings')

map_type = st.sidebar.radio("Select Map Type", ('custom', 'random'))

if map_type == 'custom':
    custom_map = st.sidebar.selectbox("Select Custom Map", ['custom1', 'custom2'])
else:
    map_size = st.sidebar.slider("Select Map Size", 5, 20, 10)

algorithm = st.sidebar.selectbox("Choose Algorithm", ['Q Learning', 'SARSA'])

total_episodes = st.sidebar.number_input("Total Episodes", min_value=1, value=1000, step=100)
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
probability_frozen = st.sidebar.slider("Probability Frozen", 0.0, 1.0, 0.5)

if st.sidebar.button("Run Simulation"):
    display_results()