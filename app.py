import streamlit as st
import cv2
import numpy as np
import time
import os
import base64
from io import BytesIO
from camera_utils import CameraCapture
from asl_model import ASLModel
from image_preprocessing import preprocess_frame
import tempfile
from typing import Tuple, Optional, List
import database as db
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="ASL Sign Language to Text Converter",
    page_icon="👋",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'camera_initialized' not in st.session_state:
    st.session_state.camera_initialized = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Recognition"
if 'stats_data' not in st.session_state:
    st.session_state.stats_data = None

def image_to_base64(img):
    """Convert an image to base64 string for database storage"""
    if img is None:
        return None
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_image(base64_str):
    """Convert a base64 string back to an image"""
    if base64_str is None:
        return None
    img_data = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Title and description
st.title("ASL Sign Language to Text Converter")
st.markdown("""
This application uses computer vision and machine learning to convert American Sign Language (ASL) 
signs into text in real-time using your webcam.
""")

# Main navigation tabs
tabs = ["Recognition", "History", "Statistics"]
st.session_state.current_tab = st.radio("Navigation", tabs, horizontal=True)

# Sidebar for model loading and settings
with st.sidebar:
    st.header("Settings")
    
    # Model loading section
    if not st.session_state.model_loaded:
        st.info("Model is being loaded. This may take a moment...")
        
        # Load the model
        try:
            model = ASLModel()
            model.load_model()
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.success("Model is loaded and ready!")
    
    # Camera settings - only show if on Recognition tab
    if st.session_state.current_tab == "Recognition":
        st.subheader("Camera Settings")
        camera_source = st.selectbox("Camera Source", ["Webcam", "Upload Video"], index=0)
        force_real_camera = st.checkbox("Force Real Camera (No Mock Mode)", False)
        
        # Show info about camera modes
        if force_real_camera:
            st.info("Real camera mode enabled. This will try to access your physical webcam and fail if not available. Use this when running the app locally on your computer.")
        else:
            st.info("Mock camera mode will be used as a fallback if a real camera is not available.")
        
        # Prediction settings
        st.subheader("Recognition Settings")
        prediction_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        prediction_smoothing = st.checkbox("Enable Prediction Smoothing", True)
        show_processed_image = st.checkbox("Show Processed Image", False)
        save_to_database = st.checkbox("Save Recognitions to Database", True)
        
        # Reset button
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.session_state.predictions = []
    
    # Database settings - only show if on History or Statistics tab
    elif st.session_state.current_tab in ["History", "Statistics"]:
        st.subheader("Database Settings")
        
        if st.button("Refresh Data"):
            if st.session_state.current_tab == "History":
                with st.spinner("Loading recognition history..."):
                    # This will refresh the data when the user returns to the History tab
                    st.session_state.recognition_history = None
            
            elif st.session_state.current_tab == "Statistics":
                with st.spinner("Calculating statistics..."):
                    # Force refresh of statistics data
                    st.session_state.stats_data = None

# RECOGNITION TAB
if st.session_state.current_tab == "Recognition":
    # Main content area with two columns
    col1, col2 = st.columns([2, 1])
    
    # Camera feed and prediction in left column
    with col1:
        st.subheader("Camera Feed")
        camera_placeholder = st.empty()
        
        if camera_source == "Webcam":
            start_camera = st.button("Start Camera")
            stop_camera = st.button("Stop Camera")
            
            if start_camera:
                st.session_state.camera_initialized = True
            
            if stop_camera:
                st.session_state.camera_initialized = False
                if 'camera' in st.session_state:
                    st.session_state.camera.release()
                    del st.session_state.camera
                    
            # Initialize camera if not already initialized
            if st.session_state.camera_initialized and 'camera' not in st.session_state:
                try:
                    # Initialize camera with force_real_camera option
                    st.session_state.camera = CameraCapture(force_real_camera=force_real_camera)
                    
                    # Show success message based on mode
                    if st.session_state.camera.mock_mode:
                        st.success("Camera initialized in mock mode! Real camera not available.")
                    else:
                        st.success("Real camera initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing camera: {e}")
                    st.session_state.camera_initialized = False
        
        elif camera_source == "Upload Video":
            uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
            if uploaded_file is not None:
                # Save the uploaded file to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
                temp_file.close()
                
                try:
                    # Use force_real_camera parameter here too
                    st.session_state.camera = CameraCapture(source=temp_file_path, force_real_camera=force_real_camera)
                    st.session_state.camera_initialized = True
                    
                    # Show appropriate message
                    if st.session_state.camera.mock_mode:
                        st.warning("Video could not be loaded. Using mock mode instead.")
                    else:
                        st.success("Video loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading video: {e}")
                    st.session_state.camera_initialized = False
                    os.unlink(temp_file_path)
        
        # Display the current prediction
        st.subheader("Current Prediction")
        current_prediction = st.empty()
        
        # Show processed image if enabled
        if show_processed_image:
            st.subheader("Processed Image")
            processed_img_placeholder = st.empty()
    
    # Prediction history and additional info in right column
    with col2:
        st.subheader("Recognition History")
        history_placeholder = st.empty()
        
        # Display information about the application
        with st.expander("About ASL Recognition"):
            st.markdown("""
            ### How it works
            1. The application captures frames from your camera
            2. Each frame is processed to detect hands and extract relevant features
            3. A machine learning model (Random Forest) classifies the hand sign
            4. The recognized sign is converted to text
            
            ### Tips for better recognition
            - Ensure good lighting conditions
            - Position your hand in the center of the frame
            - Make clear, distinct signs
            - Hold the sign steady for a moment
            - Use a plain background if possible
            """)
    
    # Main processing loop
    if st.session_state.camera_initialized and st.session_state.model_loaded:
        try:
            # Get frame from camera
            frame = st.session_state.camera.get_frame()
            
            if frame is not None:
                # Preprocess the frame for the model - now returns both processed image and annotated original
                processed_img, annotated_frame = preprocess_frame(frame)
                
                # Store processed image in session state for display if needed
                st.session_state.processed_img = processed_img
                
                # Get prediction from model
                prediction, confidence = st.session_state.model.predict(processed_img)
                
                # Add prediction text to the annotated frame
                cv2.putText(
                    annotated_frame, 
                    f"{prediction} ({confidence:.2f})", 
                    (50, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Apply confidence threshold
                if confidence >= prediction_threshold:
                    # Apply smoothing if enabled
                    if prediction_smoothing:
                        st.session_state.predictions.append(prediction)
                        # Keep only the last 5 predictions for smoothing
                        if len(st.session_state.predictions) > 5:
                            st.session_state.predictions.pop(0)
                        
                        # Get the most common prediction from the last few frames
                        if st.session_state.predictions:
                            from collections import Counter
                            prediction = Counter(st.session_state.predictions).most_common(1)[0][0]
                    
                    # Add to history if it's a new prediction
                    if not st.session_state.prediction_history or st.session_state.prediction_history[-1] != prediction:
                        st.session_state.prediction_history.append(prediction)
                        # Keep history manageable
                        if len(st.session_state.prediction_history) > 20:
                            st.session_state.prediction_history.pop(0)
                        
                        # Save to database if enabled
                        if save_to_database:
                            try:
                                # Convert frame to base64 for storage
                                img_base64 = image_to_base64(annotated_frame)
                                # Save to database
                                db.save_recognition(prediction, confidence, img_base64)
                            except Exception as e:
                                st.sidebar.error(f"Database error: {e}")
                
                # Display the frame with annotations
                camera_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                
                # Display the processed image if enabled
                if show_processed_image and st.session_state.processed_img is not None:
                    processed_img_placeholder.image(
                        st.session_state.processed_img, 
                        caption="Processed Hand Image", 
                        use_column_width=True
                    )
                
                # Display current prediction
                current_prediction.markdown(f"### Recognized Sign: {prediction} (Confidence: {confidence:.2f})")
                
                # Update prediction history display
                history_placeholder.markdown("\n".join([f"- {pred}" for pred in reversed(st.session_state.prediction_history)]))
        
        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.session_state.camera_initialized = False
            if 'camera' in st.session_state:
                st.session_state.camera.release()
                del st.session_state.camera
    
    elif not st.session_state.camera_initialized and camera_source == "Webcam":
        camera_placeholder.info("Camera is not started. Click 'Start Camera' to begin recognition.")
    elif not st.session_state.camera_initialized and camera_source == "Upload Video":
        camera_placeholder.info("Please upload a video file to begin recognition.")
    elif not st.session_state.model_loaded:
        camera_placeholder.warning("Model is still loading. Please wait...")

# HISTORY TAB
elif st.session_state.current_tab == "History":
    st.subheader("Recognition History")
    
    # Load the recognition history from the database
    if 'recognition_history' not in st.session_state or st.session_state.recognition_history is None:
        with st.spinner("Loading recognition history..."):
            # Get recognition history from the database
            recognitions = db.get_recent_recognitions(limit=100)
            st.session_state.recognition_history = recognitions
    
    # Display the recognition history
    if not st.session_state.recognition_history:
        st.info("No recognition history available. Try recognizing some ASL signs first.")
    else:
        # Create columns for the history display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a table of history data
            history_data = []
            for rec in st.session_state.recognition_history:
                history_data.append({
                    "ID": rec.id,
                    "Letter": rec.letter,
                    "Confidence": f"{rec.confidence:.2f}",
                    "Timestamp": rec.timestamp,
                    "Feedback": rec.user_feedback or "None"
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            # Select a recognition to view details
            if st.session_state.recognition_history:
                selected_id = st.selectbox(
                    "Select a recognition to view details:",
                    options=[rec.id for rec in st.session_state.recognition_history],
                    format_func=lambda x: f"ID {x}: {next((r.letter for r in st.session_state.recognition_history if r.id == x), '')}"
                )
                
                # Get the selected recognition
                selected_rec = next((r for r in st.session_state.recognition_history if r.id == selected_id), None)
                
                if selected_rec:
                    st.write(f"**Letter:** {selected_rec.letter}")
                    st.write(f"**Confidence:** {selected_rec.confidence:.2f}")
                    st.write(f"**Timestamp:** {selected_rec.timestamp}")
                    
                    # Display the image if available
                    if selected_rec.image_data:
                        try:
                            img = base64_to_image(selected_rec.image_data)
                            if img is not None:
                                st.image(img, channels="BGR", caption=f"Recognition of letter {selected_rec.letter}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    
                    # Feedback section
                    st.subheader("Feedback")
                    current_feedback = selected_rec.user_feedback or "None"
                    st.write(f"Current feedback: {current_feedback}")
                    
                    # Feedback options
                    feedback_options = ["Correct", "Incorrect", "Unclear", "None"]
                    new_feedback = st.selectbox("Provide feedback:", options=feedback_options, index=feedback_options.index(current_feedback) if current_feedback in feedback_options else 3)
                    
                    if st.button("Submit Feedback") and new_feedback != current_feedback:
                        if new_feedback == "None":
                            new_feedback = None
                        
                        # Save the feedback to the database
                        success = db.save_user_feedback(selected_id, new_feedback)
                        if success:
                            st.success("Feedback saved successfully!")
                            # Force refresh of the recognition history
                            st.session_state.recognition_history = None
                            st.rerun()
                        else:
                            st.error("Failed to save feedback.")

# STATISTICS TAB
elif st.session_state.current_tab == "Statistics":
    st.subheader("Recognition Statistics")
    
    # Load statistics data if not already loaded
    if st.session_state.stats_data is None:
        with st.spinner("Calculating statistics..."):
            st.session_state.stats_data = db.get_recognition_stats()
    
    # Display statistics
    if st.session_state.stats_data and st.session_state.stats_data['total_count'] > 0:
        total_count = st.session_state.stats_data['total_count']
        
        # Create tabs for different statistics views
        stat_tabs = ["Overview", "By Letter", "Confidence Analysis"]
        selected_stat_tab = st.radio("Statistics View", stat_tabs, horizontal=True)
        
        if selected_stat_tab == "Overview":
            # Show overall stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Recognitions", total_count)
            
            with col2:
                unique_letters = len(st.session_state.stats_data['by_letter'])
                st.metric("Unique Letters", unique_letters)
            
            with col3:
                avg_confidence = sum(item['avg_confidence'] for item in st.session_state.stats_data['by_letter']) / len(st.session_state.stats_data['by_letter'])
                st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
            
            # Create a timeline of recognitions
            st.subheader("Recent Activity")
            # Get recent 100 recognitions for the timeline
            recent_recs = db.get_recent_recognitions(limit=100)
            if recent_recs:
                # Convert to DataFrame for easier manipulation
                df_timeline = pd.DataFrame([
                    {"timestamp": rec.timestamp, "letter": rec.letter, "confidence": rec.confidence}
                    for rec in recent_recs
                ])
                
                # Group by day and count
                df_timeline['date'] = df_timeline['timestamp'].dt.date
                daily_counts = df_timeline.groupby('date').size().reset_index(name='count')
                
                # Plot the timeline
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(daily_counts['date'], daily_counts['count'], color='skyblue')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Recognitions')
                ax.set_title('Recognition Activity Timeline')
                fig.autofmt_xdate()  # Rotate date labels
                
                st.pyplot(fig)
                
        elif selected_stat_tab == "By Letter":
            # Create a DataFrame for letter stats
            letter_stats = pd.DataFrame(st.session_state.stats_data['by_letter'])
            
            # Sort by count
            letter_stats = letter_stats.sort_values('count', ascending=False)
            
            # Plot letter distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(letter_stats['letter'], letter_stats['count'], color='skyblue')
            ax.set_xlabel('Letter')
            ax.set_ylabel('Count')
            ax.set_title('Recognition Counts by Letter')
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Show the data in a table
            st.dataframe(letter_stats, use_container_width=True)
            
        elif selected_stat_tab == "Confidence Analysis":
            # Create a DataFrame for confidence analysis
            conf_stats = pd.DataFrame(st.session_state.stats_data['by_letter'])
            
            # Sort by average confidence
            conf_stats = conf_stats.sort_values('avg_confidence', ascending=False)
            
            # Plot confidence distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(conf_stats['letter'], conf_stats['avg_confidence'], color='lightgreen')
            ax.set_xlabel('Letter')
            ax.set_ylabel('Average Confidence')
            ax.set_title('Average Confidence by Letter')
            ax.set_ylim(0, 1.0)  # Confidence is between 0 and 1
            
            # Add confidence labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Show the data in a table
            st.dataframe(conf_stats, use_container_width=True)
    
    else:
        st.info("No recognition data available yet. Try recognizing some ASL signs first.")

# Footer always shown at the bottom
st.markdown("---")
st.markdown("ASL Sign Language to Text Converter | Developed with Streamlit, OpenCV, and scikit-learn")
