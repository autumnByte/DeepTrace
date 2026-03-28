"""
Streamlit web application for deepfake detection
"""
import streamlit as st
import tempfile
import os
from pathlib import Path
import time
import plotly.graph_objects as go
import pandas as pd
import cv2  # Added for video preview

# Import modules
from main import run_pipeline

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .timestamp-box {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff4757 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
    .segment-container {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff4757;
    }
    .warning-text {
        color: #ff4444;
        font-weight: bold;
    }
    .success-text {
        color: #00C851;
        font-weight: bold;
    }
    .metric-card {
        background: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'><h1>🎥 Deepfake Detection & Manipulation Timestamp Localization</h1></div>", 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙ Settings")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="Scores above this threshold are considered fake"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system detects deepfake manipulations in videos and identifies 
    specific timestamps where manipulation occurs.
    
    **Features:**
    - Frame extraction
    - Face detection
    - Deepfake classification
    - Timestamp localization
    """)

# Main content
st.markdown("Upload a video to detect **deepfake manipulation** and identify suspicious timestamps.")

# File uploader
uploaded_video = st.file_uploader(
    "Upload Video", 
    type=["mp4", "mov", "avi", "mkv"],
    help="Supported formats: MP4, MOV, AVI, MKV"
)

if uploaded_video is not None:
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name
    
    # Display video and controls
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📹 Uploaded Video")
        st.video(video_path)
    
    with col2:
        st.subheader("🔍 Detection Control")
        
        # In the detection section (around line 85-120), update this part:

    if st.button("🚀 Run Deepfake Detection", type="primary", use_container_width=True):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Run pipeline with progress updates
            status_text.text("Step 1/4: Extracting frames...")
            progress_bar.progress(25)
            
            results = run_pipeline(video_path, threshold)  # Make sure threshold is passed
            
            status_text.text("Step 2/4: Analyzing faces...")
            progress_bar.progress(50)
            time.sleep(0.5)
            
            status_text.text("Step 3/4: Running deepfake detection (CNN + Random Forest)...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            status_text.text("Step 4/4: Localizing timestamps...")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            status_text.text("✅ Detection Completed!")
            progress_bar.empty()
            
            # Store results in session state - MAKE SURE ALL DATA IS STORED
            st.session_state['results'] = results
            st.session_state['threshold'] = threshold
            st.session_state['video_path'] = video_path
            
            # Also store comparison data separately for easier access
            if 'comparison_report' in results:
                st.session_state['comparison_report'] = results['comparison_report']
            if 'comparison_data' in results:
                st.session_state['comparison_data'] = results['comparison_data']
            
            st.success("Detection Complete! View results in the tabs below.")
            st.rerun()  # Force refresh to show results
            
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            status_text.empty()
            progress_bar.empty()

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    threshold = st.session_state['threshold']
    
    segments = results["segments"]
    faces = results.get("faces", {})
    fake_scores = results.get("fake_scores", {})
    
    # ===== PROMINENT TIMESTAMP DISPLAY AT TOP =====
    st.markdown("---")
    st.markdown("## ⏱️ DETECTED MANIPULATION TIMESTAMPS")
    
    if len(segments) == 0:
        st.success("✅ No manipulation detected in this video")
    else:
        # Show total manipulated duration
        total_duration = sum(seg["end"] - seg["start"] for seg in segments)
        
        # Summary card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Manipulated Segments", len(segments))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Manipulated Time", f"{total_duration:.2f}s")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            fake_count = sum(1 for s in fake_scores.values() if s >= threshold)
            confidence = (fake_count / len(fake_scores)) * 100 if fake_scores else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Detection Confidence", f"{confidence:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visual timeline of segments
        st.markdown("### 📊 Manipulation Timeline")
        
        # Create a custom progress bar for each segment
        if 'timestamps' in results and fake_scores:
            timestamps = results['timestamps']
            if timestamps:
                video_duration = max(timestamps.values())
                
                # Create a visual representation
                timeline_html = "<div style='background: #2d2d2d; padding: 1rem; border-radius: 10px;'>"
                timeline_html += "<div style='display: flex; height: 40px; width: 100%; background: #444; border-radius: 5px; overflow: hidden;'>"
                
                for seg in segments:
                    start_percent = (seg["start"] / video_duration) * 100
                    width_percent = ((seg["end"] - seg["start"]) / video_duration) * 100
                    
                    timeline_html += f"""
                    <div style='width: {start_percent}%; background: #444;'></div>
                    <div style='width: {width_percent}%; background: #ff4757; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;'>
                        {seg["start"]:.1f}s-{seg["end"]:.1f}s
                    </div>
                    """
                
                timeline_html += "</div></div>"
                st.markdown(timeline_html, unsafe_allow_html=True)
        
        # Detailed segment cards
        st.markdown("### 📋 Detailed Segments")
        for i, seg in enumerate(segments, 1):
            start = seg["start"]
            end = seg["end"]
            duration = round(end - start, 2)
            
            with st.container():
                st.markdown(f"""
                <div class='segment-container'>
                    <h4>Segment {i}</h4>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='font-size: 1.5rem; font-weight: bold; color: #ff4757;'>{start:.2f}s</span>
                            <span style='font-size: 1.2rem;'> → </span>
                            <span style='font-size: 1.5rem; font-weight: bold; color: #ff4757;'>{end:.2f}s</span>
                        </div>
                        <div style='background: #2d2d2d; padding: 0.5rem 1rem; border-radius: 20px;'>
                            ⏱️ Duration: <strong>{duration}s</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a mini progress bar for this segment
                st.progress(1.0, text=f"Manipulated Region {i}")
    
    st.markdown("---")
    
    # Create tabs for additional views
    tab1, tab2, tab3, tab4 = st.tabs(["⚠ Manipulation Analysis", "📊 Timeline View", "🔬 Frame Analysis", "🤖 Model Comparison"])
    
    with tab1:
        st.subheader("Confidence Over Time")
        
        if fake_scores and 'timestamps' in results:
            # Prepare data for plotting
            timestamps = results['timestamps']
            df = pd.DataFrame([
                {"time": timestamps[frame], "confidence": score}
                for frame, score in fake_scores.items()
                if frame in timestamps
            ]).sort_values("time")
            
            if not df.empty:
                # Create interactive plot
                fig = go.Figure()
                
                # Add confidence line
                fig.add_trace(go.Scatter(
                    x=df["time"],
                    y=df["confidence"],
                    mode='lines',
                    name='Confidence',
                    line=dict(color='#3498db', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.2)'
                ))
                
                # Add threshold line
                fig.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({threshold})",
                    annotation_position="top right"
                )
                
                # Highlight manipulated regions
                for seg in segments:
                    fig.add_vrect(
                        x0=seg["start"],
                        x1=seg["end"],
                        fillcolor="red",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        annotation_text="Manipulated"
                    )
                
                fig.update_layout(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Fake Confidence",
                    hovermode='x',
                    height=400,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistics below chart
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_confidence = df["confidence"].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.2f}")
                with col2:
                    max_confidence = df["confidence"].max()
                    st.metric("Peak Confidence", f"{max_confidence:.2f}")
                with col3:
                    min_confidence = df["confidence"].min()
                    st.metric("Minimum Confidence", f"{min_confidence:.2f}")
            else:
                st.info("No timeline data available")
    
    with tab2:
        st.subheader("Suspicious Frames")
        
        if not faces:
            st.info("No faces detected in video")
        else:
            # Collect suspicious frames
            suspicious = []
            for frame, data in faces.items():
                score = fake_scores.get(frame)
                if score is not None and score >= threshold:
                    suspicious.append((frame, score, data.get("face_path", "")))
            
            if not suspicious:
                st.info("No suspicious frames detected above threshold")
            else:
                st.write(f"Showing top {min(9, len(suspicious))} of {len(suspicious)} suspicious frames")
                
                # Sort by highest fake score
                suspicious = sorted(suspicious, key=lambda x: x[1], reverse=True)
                
                # Create grid of images
                cols = st.columns(3)
                for idx, (frame, score, path) in enumerate(suspicious[:9]):
                    col = cols[idx % 3]
                    with col:
                        if path and os.path.exists(path):
                            st.image(path, use_container_width=True)
                            st.caption(f"Score: {score:.2f} | {frame}")
                            # Add timestamp if available
                            if 'timestamps' in results and frame in results['timestamps']:
                                st.caption(f"Time: {results['timestamps'][frame]:.2f}s")
                        else:
                            st.warning("Image not found")
    
    with tab3:
        st.subheader("Detailed Statistics")
        
        if fake_scores:
            # Create a dataframe with all results
            stats_data = []
            for frame, score in fake_scores.items():
                timestamp = results['timestamps'].get(frame, 0) if 'timestamps' in results else 0
                is_fake = score >= threshold
                stats_data.append({
                    "Frame": frame,
                    "Timestamp (s)": round(timestamp, 2),
                    "Confidence Score": round(score, 3),
                    "Classification": "FAKE" if is_fake else "REAL"
                })
            
            df_stats = pd.DataFrame(stats_data)
            
            # Show summary stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Distribution")
                fake_count = len(df_stats[df_stats["Classification"] == "FAKE"])
                real_count = len(df_stats[df_stats["Classification"] == "REAL"])
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Fake', 'Real'],
                    values=[fake_count, real_count],
                    marker_colors=['#ff4757', '#00C851']
                )])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Score Distribution")
                fig = go.Figure(data=[go.Histogram(
                    x=df_stats["Confidence Score"],
                    nbinsx=20,
                    marker_color='#3498db'
                )])
                fig.update_layout(
                    xaxis_title="Confidence Score",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show full table
            st.markdown("### All Frames Data")
            st.dataframe(df_stats, use_container_width=True, height=400)
        else:
            st.info("No data available")
    with tab4:
        st.subheader("🤖 Model Comparison: CNN vs Random Forest")
        
        # Get the comparison report
        comparison_df = None
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            comparison_df = results.get('comparison_report')
        
        if comparison_df is not None and not comparison_df.empty:
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_cnn = comparison_df["CNN Score"].mean()
                st.metric("CNN Average Score", f"{avg_cnn:.3f}")
            with col2:
                avg_rf = comparison_df["Random Forest Score"].mean()
                st.metric("Random Forest Avg Score", f"{avg_rf:.3f}")
            with col3:
                agreement = (comparison_df["Agreement"] == "Yes").sum()
                agreement_pct = (agreement / len(comparison_df)) * 100
                st.metric("Model Agreement", f"{agreement_pct:.1f}%")
            
            # Scatter plot
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=comparison_df["CNN Score"],
                y=comparison_df["Random Forest Score"],
                mode='markers',
                marker=dict(
                    size=8,
                    color=comparison_df["Difference"],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Difference"),
                    reversescale=True
                ),
                text=comparison_df["Frame"],
                hovertemplate="Frame: %{text}<br>CNN: %{x:.3f}<br>RF: %{y:.3f}<extra></extra>"
            ))
            
            # Add diagonal line (perfect agreement)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect Agreement'
            ))
            
            fig.update_layout(
                title="CNN vs Random Forest Predictions",
                xaxis_title="CNN Score (Higher = More Likely Fake)",
                yaxis_title="Random Forest Score (Higher = More Likely Fake)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show where models disagree
            st.subheader("⚠️ Frames Where Models Disagree")
            disagreements = comparison_df[comparison_df["Agreement"] == "No"]
            if not disagreements.empty:
                st.dataframe(disagreements, use_container_width=True)
                st.caption(f"Total disagreements: {len(disagreements)} frames ({len(disagreements)/len(comparison_df)*100:.1f}%)")
            else:
                st.success("✅ Models agree on all frames! Good consistency.")
            
            # Full data in expander
            with st.expander("📋 View Full Comparison Data"):
                st.dataframe(comparison_df, use_container_width=True)
            
            # Summary statistics
            with st.expander("📊 How to Interpret This Comparison"):
                st.markdown(f"""
                ### Model Comparison Summary
                
                - **Total Frames Analyzed**: {len(comparison_df)}
                - **Model Agreement**: {agreement_pct:.1f}%
                - **CNN Average Score**: {avg_cnn:.3f}
                - **RF Average Score**: {avg_rf:.3f}
                
                ### Interpretation Guide
                
                - Points **above** the diagonal: Random Forest thinks it's more fake than CNN
                - Points **below** the diagonal: CNN thinks it's more fake than Random Forest
                - Points **far from diagonal**: Significant disagreement between models
                - **Green points**: Models agree (difference < 0.2)
                - **Red points**: Models disagree (difference ≥ 0.2)
                
                ### What This Means
                
                When both models agree on a frame, it increases confidence in the classification.
                When they disagree, it suggests the frame may be ambiguous or one model may be making an error.
                """)
        
        else:
            st.info("No comparison data available. Run detection first to see model comparison.")

# Cleanup temporary file on session end
if 'video_path' in st.session_state:
    try:
        os.unlink(st.session_state['video_path'])
    except:
        pass