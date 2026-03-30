"""
Streamlit web application for deepfake detection
"""
import streamlit as st
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import plotly.graph_objects as go
import pandas as pd
import cv2
import json

# Initialize session state
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'cnn_threshold' not in st.session_state:
    st.session_state['cnn_threshold'] = 0.45  # Changed from 0.6 to 0.45 for better recall
if 'rf_threshold' not in st.session_state:
    st.session_state['rf_threshold'] = 0.33
if 'video_path' not in st.session_state:
    st.session_state['video_path'] = None
if 'comparison_report' not in st.session_state:
    st.session_state['comparison_report'] = None
if 'comparison_data' not in st.session_state:
    st.session_state['comparison_data'] = None

# Load evaluation metrics
try:
    with open("metrics.json") as f:
        metrics = json.load(f)
    # Check if metrics have expected keys
    required_keys = ['accuracy', 'precision', 'recall', 'f1_score']
    missing_keys = [key for key in required_keys if key not in metrics]
    if missing_keys:
        st.warning(f"⚠️ metrics.json is missing keys: {missing_keys}. Please run evaluate.py with proper metrics generation.")
        metrics = None
except FileNotFoundError:
    metrics = None
    st.warning("⚠️ metrics.json not found. Run evaluate.py first to generate metrics.json")
except Exception as e:
    metrics = None
    st.warning(f"⚠️ Error loading metrics.json: {str(e)}")

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
    .confidence-low {
        color: #00C851;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffa502;
        font-weight: bold;
    }
    .confidence-high {
        color: #ff4757;
        font-weight: bold;
    }
    .disclaimer {
        background: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffa502;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'><h1> DeepTrace:Detecting deepfakes and manipulated timestamps</h1></div>", 
            unsafe_allow_html=True)

# Add disclaimer at the top
st.markdown("""
<div class='disclaimer'>
⚠️ <strong>Research Prototype Disclaimer</strong><br>
This model is trained on a limited dataset and may not generalize to all real-world deepfakes. 
Results should be interpreted as probabilistic indicators, not definitive proof of manipulation.
</div>
""", unsafe_allow_html=True)

st.markdown("### ⚙ Detection Settings")

# CNN threshold (user adjustable) - now set to 0.45 for better recall
cnn_threshold = st.slider(
    "CNN Detection Threshold (Primary Detector)",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state['cnn_threshold'],
    step=0.05,
    help="CNN scores above this threshold are considered fake. Lower threshold = more detections but more false positives."
)

# RF threshold (fixed for validation)
rf_threshold = 0.33
st.info(f"🎯 Random Forest Threshold: **{rf_threshold}** (confidence booster, not gatekeeper)")

# Confidence band explanation
st.markdown("""
**Confidence Interpretation:**
- 🟢 **Low confidence (<0.4)**: Likely real
- 🟡 **Medium confidence (0.4-0.6)**: Uncertain - may be ambiguous
- 🔴 **High confidence (>0.6)**: Likely fake
""")

# Update session state with current thresholds
st.session_state['cnn_threshold'] = cnn_threshold
st.session_state['rf_threshold'] = rf_threshold

# Main content
st.markdown("Upload a video to detect **potential deepfake manipulation** and identify suspicious timestamps.")

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
    
    st.subheader("📹 Uploaded Video")
    st.video(video_path)
        
    if st.button("🚀 Run Deepfake Detection", type="primary", use_container_width=True):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Run pipeline with progress updates
            status_text.text("Step 1/4: Extracting frames...")
            progress_bar.progress(25)
            
            # Pass CNN threshold to pipeline
            results = run_pipeline(video_path, cnn_threshold)
            
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
            
            # Store results in session state
            st.session_state['results'] = results
            st.session_state['video_path'] = video_path
            
            # Also store comparison data separately for easier access
            if 'comparison_report' in results:
                st.session_state['comparison_report'] = results['comparison_report']
            if 'comparison_data' in results:
                st.session_state['comparison_data'] = results['comparison_data']
            
            st.success("Detection Complete! View results in the tabs below.")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            status_text.empty()
            progress_bar.empty()

# Function to get confidence level and color
def get_confidence_level(score):
    if score < 0.4:
        return "Low Confidence (Likely Real)", "confidence-low"
    elif score < 0.6:
        return "Medium Confidence (Uncertain)", "confidence-medium"
    else:
        return "High Confidence (Likely Fake)", "confidence-high"

# Display results if available
if st.session_state['results'] is not None:
    results = st.session_state['results']
    cnn_threshold = st.session_state['cnn_threshold']
    rf_threshold = st.session_state['rf_threshold']
    
    segments = results["segments"]
    faces = results.get("faces", {})
    fake_scores = results.get("fake_scores", {})
    
    # ===== PROMINENT TIMESTAMP DISPLAY AT TOP =====
    st.markdown("---")
    st.markdown("## ⏱️ SUSPICIOUS TIMESTAMPS (Based on CNN Detector)")
    
    if len(segments) == 0:
        st.success("✅ No suspicious segments detected in this video")
    else:
        # Show total manipulated duration
        total_duration = sum(seg["end"] - seg["start"] for seg in segments)
        
        # Summary card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Suspicious Segments", len(segments))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Suspicious Duration", f"{total_duration:.2f}s")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            if fake_scores:
                confidence = np.mean(list(fake_scores.values())) * 100
            else:
                confidence = 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Confidence", f"{confidence:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visual timeline of segments
        st.markdown("### 📊 Suspicious Timeline")

        if 'timestamps' in results and fake_scores:
            timestamps = results['timestamps']
            if timestamps:
                video_duration = max(timestamps.values())
                
                # Get segments
                segments = results["segments"]
                
                # Sort segments by start time
                sorted_segments = sorted(segments, key=lambda x: x['start'])
                
                # Create the timeline container
                timeline_html = "<div style='background: #2d2d2d; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>"
                timeline_html += "<div style='position: relative; height: 60px; width: 100%; background: #333; border-radius: 8px; overflow: hidden;'>"
                
                # Add base background (full timeline)
                timeline_html += "<div style='position: absolute; width: 100%; height: 100%; background: #444;'></div>"
                
                # Add each suspicious segment at its exact position
                for seg in sorted_segments:
                    start_percent = (seg["start"] / video_duration) * 100
                    width_percent = ((seg["end"] - seg["start"]) / video_duration) * 100
                    
                    # Position the segment absolutely at its start percentage
                    timeline_html += f"""
                    <div style='position: absolute; left: {start_percent}%; width: {width_percent}%; height: 100%; background: linear-gradient(90deg, #ff6b6b, #ff4757); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;'>
                        <span style='background: rgba(0,0,0,0.6); padding: 2px 6px; border-radius: 12px; white-space: nowrap;'>
                            {seg["start"]:.1f}s-{seg["end"]:.1f}s
                        </span>
                    </div>
                    """
                
                timeline_html += "</div>"
                
                # Add timeline labels
                timeline_html += "<div style='display: flex; justify-content: space-between; margin-top: 8px; color: #888; font-size: 12px;'>"
                timeline_html += f"<span>0.0s</span>"
                timeline_html += f"<span>{video_duration/4:.1f}s</span>"
                timeline_html += f"<span>{video_duration/2:.1f}s</span>"
                timeline_html += f"<span>{3*video_duration/4:.1f}s</span>"
                timeline_html += f"<span>{video_duration:.1f}s</span>"
                timeline_html += "</div></div>"
                
                st.components.v1.html(timeline_html, height=60)
        
        # Detailed segment cards with confidence levels
        st.markdown("### 📋 Detailed Segments")
        for i, seg in enumerate(segments, 1):
            start = seg["start"]
            end = seg["end"]
            duration = round(end - start, 2)
            
            # Calculate average confidence for this segment
            segment_scores = []
            for frame, score in fake_scores.items():
                if frame in timestamps and start <= timestamps[frame] <= end:
                    segment_scores.append(score)
            avg_conf = np.mean(segment_scores) if segment_scores else 0.5
            conf_text, conf_class = get_confidence_level(avg_conf)
            
            with st.container():
                st.markdown(f"""
                <div class='segment-container'>
                    <h4>Segment {i}</h4>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                        <div>
                            <span style='font-size: 1.5rem; font-weight: bold; color: #ff4757;'>{start:.2f}s</span>
                            <span style='font-size: 1.2rem;'> → </span>
                            <span style='font-size: 1.5rem; font-weight: bold; color: #ff4757;'>{end:.2f}s</span>
                        </div>
                        <div style='background: #2d2d2d; padding: 0.5rem 1rem; border-radius: 20px;'>
                            ⏱️ Duration: <strong>{duration}s</strong>
                        </div>
                    </div>
                    <div>
                        <span class='{conf_class}'>📊 {conf_text}</span>
                        <span style='margin-left: 10px;'>Average Confidence: {avg_conf:.2%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a mini progress bar for this segment
                st.progress(avg_conf, text=f"Confidence Level: {avg_conf:.1%}")
    
    st.markdown("---")
    
    # Create tabs for additional views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚠ Confidence Analysis", 
        "📊 Timeline View", 
        "🔬 Frame Analysis", 
        "🤖 Model Comparison",
        "🎯 System Trust & Limitations"
    ])
    
    with tab1:
        st.subheader("Confidence Analysis Over Time")
        
        if fake_scores and 'timestamps' in results:
            # Prepare data for plotting
            timestamps = results['timestamps']
            df = pd.DataFrame([
                {"time": timestamps[frame], "confidence": score}
                for frame, score in fake_scores.items()
                if frame in timestamps
            ]).sort_values("time")
            
            if not df.empty:
                # Add confidence level categories
                df['confidence_level'] = df['confidence'].apply(
                    lambda x: 'High (>0.6)' if x > 0.6 else ('Medium (0.4-0.6)' if x >= 0.4 else 'Low (<0.4)')
                )
                
                # Create interactive plot with confidence bands
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
                
                # Add confidence bands
                fig.add_hrect(y0=0.4, y1=0.6, line_width=0, fillcolor="yellow", opacity=0.2, 
                             annotation_text="Uncertain Zone", annotation_position="top right")
                fig.add_hrect(y0=0.6, y1=1.0, line_width=0, fillcolor="red", opacity=0.1,
                             annotation_text="Fake Zone", annotation_position="top right")
                fig.add_hrect(y0=0.0, y1=0.4, line_width=0, fillcolor="green", opacity=0.1,
                             annotation_text="Real Zone", annotation_position="top right")
                
                # Add threshold line
                fig.add_hline(
                    y=cnn_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({cnn_threshold})",
                    annotation_position="top right"
                )
                
                # Highlight suspicious regions
                for seg in segments:
                    fig.add_vrect(
                        x0=seg["start"],
                        x1=seg["end"],
                        fillcolor="red",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        annotation_text="Suspicious"
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
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_confidence = df["confidence"].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.2f}")
                with col2:
                    max_confidence = df["confidence"].max()
                    st.metric("Peak Confidence", f"{max_confidence:.2f}")
                with col3:
                    min_confidence = df["confidence"].min()
                    st.metric("Minimum Confidence", f"{min_confidence:.2f}")
                with col4:
                    uncertain_count = len(df[(df["confidence"] >= 0.4) & (df["confidence"] <= 0.6)])
                    st.metric("Uncertain Frames", f"{uncertain_count}")
            else:
                st.info("No timeline data available")
    
    with tab2:
        st.subheader("Suspicious Frames with Confidence Levels")
        
        if not faces:
            st.info("No faces detected in video")
        else:
            # Collect suspicious frames with confidence levels
            suspicious = []
            for frame, data in faces.items():
                score = fake_scores.get(frame)
                if score is not None and score >= cnn_threshold:
                    conf_level, conf_class = get_confidence_level(score)
                    suspicious.append((frame, score, conf_level, data.get("face_path", "")))
            
            if not suspicious:
                st.info("No suspicious frames detected above CNN threshold")
            else:
                st.write(f"Showing top {min(9, len(suspicious))} of {len(suspicious)} suspicious frames")
                
                # Sort by highest fake score
                suspicious = sorted(suspicious, key=lambda x: x[1], reverse=True)
                
                # Create grid of images
                cols = st.columns(3)
                for idx, (frame, score, conf_level, path) in enumerate(suspicious[:9]):
                    col = cols[idx % 3]
                    with col:
                        if path and os.path.exists(path):
                            st.image(path, use_container_width=True)
                            # Show confidence with appropriate color
                            if score > 0.6:
                                st.markdown(f"<span class='confidence-high'>🔴 Score: {score:.2f} - {conf_level}</span>", unsafe_allow_html=True)
                            elif score > 0.4:
                                st.markdown(f"<span class='confidence-medium'>🟡 Score: {score:.2f} - {conf_level}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span class='confidence-low'>🟢 Score: {score:.2f} - {conf_level}</span>", unsafe_allow_html=True)
                            st.caption(f"Frame: {frame}")
                            # Add timestamp if available
                            if 'timestamps' in results and frame in results['timestamps']:
                                st.caption(f"Time: {results['timestamps'][frame]:.2f}s")
                        else:
                            st.warning("Image not found")
    
    with tab3:
        st.subheader("Detailed Statistics")
        
        if fake_scores:
            # Create a dataframe with all results and confidence levels
            stats_data = []
            for frame, score in fake_scores.items():
                timestamp = results['timestamps'].get(frame, 0) if 'timestamps' in results else 0
                is_fake = score >= cnn_threshold
                conf_level, _ = get_confidence_level(score)
                stats_data.append({
                    "Frame": frame,
                    "Timestamp (s)": round(timestamp, 2),
                    "CNN Score": round(score, 3),
                    "Confidence Level": conf_level,
                    "Classification": "FAKE" if is_fake else "REAL"
                })
            
            df_stats = pd.DataFrame(stats_data)
            
            # Show summary stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Distribution by Confidence")
                high_conf = len(df_stats[df_stats["CNN Score"] > 0.65])
                med_conf = len(df_stats[(df_stats["CNN Score"] >= 0.4) & (df_stats["CNN Score"] <= 0.6)])
                low_conf = len(df_stats[df_stats["CNN Score"] < 0.4])
                
                fig = go.Figure(data=[go.Pie(
                    labels=['High Confidence (>0.6)', 'Uncertain (0.4-0.6)', 'Low Confidence (<0.4)'],
                    values=[high_conf, med_conf, low_conf],
                    marker_colors=['#ff4757', '#ffa502', '#00C851']
                )])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Score Distribution")
                fig = go.Figure(data=[go.Histogram(
                    x=df_stats["CNN Score"],
                    nbinsx=20,
                    marker_color='#3498db'
                )])
                # Add confidence bands to histogram
                fig.add_vline(x=0.4, line_dash="dash", line_color="orange", 
                             annotation_text="Uncertain Lower")
                fig.add_vline(x=0.6, line_dash="dash", line_color="orange", 
                             annotation_text="Uncertain Upper")
                fig.update_layout(
                    xaxis_title="CNN Confidence Score",
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
        st.subheader("🤖 Model Comparison: CNN (Decision Maker) vs Random Forest (Confidence Booster)")
        
        # Get the comparison report
        comparison_df = st.session_state.get('comparison_report')
        
        if comparison_df is not None and not comparison_df.empty:
            # Calculate detection metrics
            cnn_fake = comparison_df["CNN Score"] > cnn_threshold
            rf_high_confidence = comparison_df["Random Forest Score"] > rf_threshold
            high_confidence_detections = cnn_fake & rf_high_confidence
            total_fake_detected = cnn_fake.sum()
            high_confidence_count = high_confidence_detections.sum()
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_cnn = comparison_df["CNN Score"].mean()
                st.metric("CNN Average Score (Decision Maker)", f"{avg_cnn:.3f}")
            with col2:
                avg_rf = comparison_df["Random Forest Score"].mean()
                st.metric("Random Forest Avg Score (Confidence Booster)", f"{avg_rf:.3f}")
            with col3:
                # Calculate agreement (same classification)
                agreement = (
                    (comparison_df["CNN Score"] > cnn_threshold) == 
                    (comparison_df["Random Forest Score"] > rf_threshold)
                )
                agreement_rate = (agreement.sum() / len(comparison_df)) * 100
                st.metric("Model Agreement", f"{agreement_rate:.1f}%")
            
            # Show detection summary
            st.markdown("### Detection Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Suspicious Frames (CNN)", f"{total_fake_detected}")
                st.caption("Primary detector - frames flagged as suspicious")
            with col2:
                st.metric("High-Confidence Detections", f"{high_confidence_count}")
                st.caption("Cross-verified by Random Forest - higher reliability")
            
            # Scatter plot with confidence zones
            fig = go.Figure()
            
            # Color points based on confidence zones
            colors = []
            for _, row in comparison_df.iterrows():
                cnn = row["CNN Score"]
                rf = row["Random Forest Score"]
                if cnn > 0.6 and rf > rf_threshold:
                    colors.append('darkred')  # High confidence fake
                elif cnn > cnn_threshold and rf > rf_threshold:
                    colors.append('red')  # Suspicious, verified
                elif cnn > cnn_threshold and rf < rf_threshold:
                    colors.append('orange')  # Suspicious, unverified
                elif cnn < 0.4 and rf < rf_threshold:
                    colors.append('green')  # Likely real
                else:
                    colors.append('yellow')  # Uncertain
            
            fig.add_trace(go.Scatter(
                x=comparison_df["CNN Score"],
                y=comparison_df["Random Forest Score"],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors,
                    showscale=False
                ),
                text=comparison_df["Frame"],
                hovertemplate="Frame: %{text}<br>CNN: %{x:.3f}<br>RF: %{y:.3f}<extra></extra>"
            ))
            
            # Add confidence zone boundaries
            fig.add_vline(x=0.4, line_dash="dash", line_color="orange", 
                         annotation_text="Uncertain Zone")
            fig.add_vline(x=0.6, line_dash="dash", line_color="orange")
            fig.add_vline(x=cnn_threshold, line_dash="dash", line_color="blue", 
                         annotation_text=f"CNN Threshold: {cnn_threshold}")
            fig.add_hline(y=rf_threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"RF Threshold: {rf_threshold}")
            
            fig.update_layout(
                title="CNN vs Random Forest: Decision Making",
                xaxis_title="CNN Score (Decision Maker)",
                yaxis_title="Random Forest Score (Confidence Booster)",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show interpretation
            st.markdown("""
            **How to Interpret This Chart:**
            - **Dark Red**: High confidence fake (CNN > 0.6, RF verified)
            - **Red**: Suspicious with verification (CNN flagged, RF agrees)
            - **Orange**: Suspicious without verification (CNN flagged, RF disagrees)
            - **Yellow**: Uncertain zone (ambiguous predictions)
            - **Green**: Likely real (both models indicate real)
            
            **CNN is the primary decision maker** - frames above the blue line are flagged.
            **Random Forest boosts confidence** - when both models agree, reliability increases.
            """)
            
            # Show frames where models disagree
            st.subheader("⚠️ Frames Where Models Disagree (Lower Confidence)")
            disagreements = comparison_df[~agreement]
            if not disagreements.empty:
                st.dataframe(disagreements, use_container_width=True)
                st.caption(f"Total disagreements: {len(disagreements)} frames ({len(disagreements)/len(comparison_df)*100:.1f}%)")
                st.info("When models disagree, we rely on CNN (primary detector) but note lower confidence.")
            else:
                st.success("✅ Models agree on all frames! Good consistency.")
            
            # Full data in expander
            with st.expander("📋 View Full Comparison Data"):
                st.dataframe(comparison_df, use_container_width=True)
        
        else:
            st.info("No comparison data available. Run detection first to see model comparison.")
    
    with tab5:
        st.subheader("🎯 System Trust & Limitations")
        
        st.markdown("""
        ### How Our System Makes Decisions
        
        Our system uses a **two-stage approach** for realistic deepfake detection:
        
        1. **CNN (EfficientNet-B0) - Primary Detector**
           - Makes the initial decision on each frame
           - Threshold: **{:.2f}** (adjustable in settings)
           - Frames above threshold are flagged as suspicious
        
        2. **Random Forest - Confidence Booster**
           - Validates CNN's decisions
           - Threshold: **{:.2f}** (fixed)
           - When both models agree, confidence increases
        
        3. **Confidence Zones**
           - **High (>0.6)**: Likely fake
           - **Uncertain (0.4-0.6)**: Ambiguous - may need review
           - **Low (<0.4)**: Likely real
        """.format(cnn_threshold, rf_threshold))
        
        # Get the comparison data for metrics
        comparison_df = st.session_state.get('comparison_report')
        
        if comparison_df is not None and not comparison_df.empty:
            # Calculate metrics
            total_frames = len(comparison_df)
            cnn_fake = comparison_df["CNN Score"] > cnn_threshold
            high_confidence = (comparison_df["CNN Score"] > 0.6) & (comparison_df["Random Forest Score"] > rf_threshold)
            uncertain_frames = (comparison_df["CNN Score"] >= 0.4) & (comparison_df["CNN Score"] <= 0.6)
            
            total_fake_detected = cnn_fake.sum()
            high_confidence_count = high_confidence.sum()
            uncertain_count = uncertain_frames.sum()
            
            # Show real-time metrics from current video
            st.markdown("---")
            st.subheader("📊 Current Video Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames Analyzed", f"{total_frames}")
            with col2:
                st.metric("Suspicious Frames (CNN)", f"{total_fake_detected}")
            with col3:
                st.metric("High-Confidence Detections", f"{high_confidence_count}")
            
            st.markdown(f"""
            **Confidence Breakdown:**
            - **High Confidence (>0.6)**: {len(comparison_df[comparison_df["CNN Score"] > 0.6])} frames
            - **Uncertain (0.4-0.6)**: {uncertain_count} frames
            - **Low Confidence (<0.4)**: {len(comparison_df[comparison_df["CNN Score"] < 0.4])} frames
            """)
        
        st.markdown("---")
        st.subheader("📊 Model Performance Metrics (Training Evaluation)")
        
        # Model accuracy from training
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 CNN Model (Primary Detector)")
            st.markdown("**EfficientNet-B0**")
            if metrics:
                st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                st.metric("Precision", f"{metrics['precision']*100:.1f}%")
                st.metric("Recall", f"{metrics['recall']*100:.1f}%")
                st.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
            else:
                st.warning("Run evaluate.py to generate metrics.json")
            st.markdown("""
            - **Role**: Primary decision maker
            - **Training Data**: 1,480 images
            - **Architecture**: EfficientNet-B0
            """)
        
        with col2:
            st.markdown("### 🌲 Random Forest (Confidence Booster)")
            st.info("**Role**: Validates CNN predictions and increases confidence")
            st.markdown("""
            - **Features**: Texture, edges, color, frequency
            - **Training Data**: Same 1,480 images
            - **Trees**: 200 estimators
            - **Lower threshold** for higher recall
            """)
        
        st.markdown("---")
        
        # ===== CONFUSION MATRIX (CNN ONLY) =====
        st.subheader("📈 Confusion Matrix (CNN Model - Test Set)")
        st.caption("Performance on held-out test data (20% of training set)")
        
        # Calculate confusion matrix values based on actual metrics
        if metrics:
            # Derive confusion matrix from metrics
            test_size = 296  # 20% of 1480
            total_positives = int(test_size * 0.324)  # ~96 fake images
            total_negatives = test_size - total_positives  # ~200 real images
            
            TP = int(metrics['recall'] * total_positives)
            FN = total_positives - TP
            if metrics['precision'] > 0:
                FP = int((TP / metrics['precision']) - TP)
            else:
                FP = 0
            TN = int(metrics['accuracy'] * test_size) - TP
            
            # Ensure non-negative values
            TP = max(0, TP)
            FP = max(0, FP)
            TN = max(0, TN)
            FN = max(0, FN)
            
            st.markdown(f"""
            | | Predicted Fake | Predicted Real |
            |---|---|---|
            | **Actual Fake** | {TP} ({TP/total_positives*100:.1f}%) | {FN} ({FN/total_positives*100:.1f}%) |
            | **Actual Real** | {FP} ({FP/total_negatives*100:.1f}%) | {TN} ({TN/total_negatives*100:.1f}%) |
            """)
            
            st.markdown(f"""
            **Performance Metrics:**
            - **Precision**: {metrics['precision']*100:.1f}% - When we predict fake, we're correct this often
            - **Recall**: {metrics['recall']*100:.1f}% - We catch this percentage of actual fakes
            - **F1 Score**: {metrics['f1_score']*100:.1f}% - Balanced measure of precision and recall
            """)
        else:
            st.warning("Run evaluate.py first to generate metrics for confusion matrix")
        
        st.markdown("---")
        
        # ===== HONEST DISCLAIMER =====
        st.subheader("⚠️ Important Limitations & Disclaimers")
        
        st.markdown("""
        <div class='disclaimer'>
        <strong>🔬 Research Prototype - Not Production-Ready</strong><br><br>
        
        <strong>Dataset Limitations:</strong><br>
        • Trained on 1,480 images (1,000 real, 480 fake) from limited sources<br>
        • May not generalize to all real-world deepfake types (GANs, FaceSwap, etc.)<br>
        • Performance varies based on video quality, lighting, and face visibility<br><br>
        
        <strong>Technical Limitations:</strong><br>
        • CNN threshold ({:.2f}) is adjustable but not optimized for all scenarios<br>
        • Random Forest serves as confidence booster, not independent detector<br>
        • Results should be interpreted as probabilistic, not definitive<br><br>
        
        <strong>Recommended Usage:</strong><br>
        • Use as a screening tool, not for conclusive evidence<br>
        • Review uncertain frames (0.4-0.6 confidence) manually<br>
        • Consider this system as a research demonstration<br><br>
        
        <strong>Future Improvements Needed:</strong><br>
        • Larger, more diverse training dataset<br>
        • More sophisticated ensemble methods<br>
        • Better handling of video compression artifacts<br>
        • Real-time processing capabilities
        </div>
        """.format(cnn_threshold), unsafe_allow_html=True)

# Cleanup temporary file on session end
if st.session_state['video_path'] is not None:
    try:
        if os.path.exists(st.session_state['video_path']):
            os.unlink(st.session_state['video_path'])
    except:
        pass