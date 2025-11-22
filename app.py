import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from models import AnomalyDetectionEnsemble, calculate_feature_importance
from datetime import datetime
import os
import io

st.set_page_config(
    page_title="AI-Powered Network Anomaly Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_or_create_model():
    """Load existing model or create new one."""
    model = AnomalyDetectionEnsemble(contamination=0.15)
    if os.path.exists('models/scaler.pkl'):
        try:
            model.load_models('models')
            return model, True
        except:
            return model, False
    return model, False

def train_models(df, feature_cols):
    """Train the ensemble models on the dataset."""
    X = df[feature_cols].values
    
    model = AnomalyDetectionEnsemble(contamination=0.15)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Training Isolation Forest...")
    progress_bar.progress(20)
    
    with st.spinner("Training ensemble models... This may take a minute."):
        history = model.train(X, epochs=30, batch_size=32, verbose=0)
    
    progress_bar.progress(100)
    status_text.text("Training complete!")
    
    model.save_models('models')
    
    return model

def plot_anomaly_timeline(df):
    """Create timeline visualization of anomalies."""
    fig = go.Figure()
    
    normal_data = df[df['prediction'] == 1]
    anomaly_data = df[df['prediction'] == -1]
    
    fig.add_trace(go.Scatter(
        x=normal_data.index,
        y=normal_data['confidence'],
        mode='markers',
        name='Normal',
        marker=dict(color='green', size=6, opacity=0.6),
        hovertemplate='<b>Normal Traffic</b><br>Index: %{x}<br>Confidence: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=anomaly_data.index,
        y=anomaly_data['confidence'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10, symbol='x'),
        hovertemplate='<b>Anomaly Detected</b><br>Index: %{x}<br>Confidence: %{y:.2f}<br>Severity: %{text}<extra></extra>',
        text=anomaly_data['severity']
    ))
    
    fig.update_layout(
        title="Anomaly Detection Timeline",
        xaxis_title="Sample Index",
        yaxis_title="Confidence Score",
        hovermode='closest',
        height=400
    )
    
    return fig

def plot_feature_distributions(df, feature_cols):
    """Plot feature distributions for normal vs anomaly."""
    anomaly_df = df[df['prediction'] == -1]
    normal_df = df[df['prediction'] == 1]
    
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=feature_cols,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, feature in enumerate(feature_cols):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(x=normal_df[feature], name='Normal', marker_color='green', opacity=0.6, showlegend=(idx==0)),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Histogram(x=anomaly_df[feature], name='Anomaly', marker_color='red', opacity=0.6, showlegend=(idx==0)),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Feature Distributions: Normal vs Anomaly",
        height=300 * n_rows,
        showlegend=True,
        barmode='overlay'
    )
    
    return fig

def plot_model_comparison(individual_preds, ensemble_total):
    """Compare individual model predictions with ensemble result."""
    model_names = list(individual_preds.keys()) + ['Ensemble (≥2 votes)']
    anomaly_counts = [np.sum(preds == -1) for preds in individual_preds.values()] + [ensemble_total]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#e377c2']
    
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=anomaly_counts,
            marker_color=colors,
            text=anomaly_counts,
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial Black'),
        )
    ])
    
    fig.update_layout(
        title="Model Comparison: Anomalies Detected by Each Model",
        xaxis_title="Model",
        yaxis_title="Number of Anomalies Detected",
        height=450,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

def plot_voting_breakdown(df):
    """Show detailed voting breakdown."""
    anomaly_df = df[df['prediction'] == -1]
    
    vote_counts = anomaly_df['severity'].value_counts()
    vote_mapping = {'low': '2 models', 'medium': '3 models', 'high': '4 models'}
    
    labels = [vote_mapping.get(sev, sev) for sev in vote_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=vote_counts.values,
            marker_color=['#ffeda0', '#feb24c', '#f03b20'],
            text=vote_counts.values,
            textposition='auto',
            textfont=dict(size=14, color='black', family='Arial Black'),
        )
    ])
    
    fig.update_layout(
        title="Voting Breakdown: How Many Models Agreed on Each Anomaly",
        xaxis_title="Number of Models in Agreement",
        yaxis_title="Count of Anomalies",
        height=400,
        font=dict(size=12)
    )
    
    return fig

def plot_severity_distribution(df):
    """Plot severity distribution of anomalies."""
    severity_counts = df[df['prediction'] == -1]['severity'].value_counts()
    
    colors = {'low': '#ffeda0', 'medium': '#feb24c', 'high': '#f03b20'}
    
    fig = go.Figure(data=[
        go.Pie(
            labels=severity_counts.index,
            values=severity_counts.values,
            marker=dict(colors=[colors.get(s, '#gray') for s in severity_counts.index]),
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="Anomaly Severity Distribution",
        height=400
    )
    
    return fig

def main():
    st.markdown('<div class="main-header">🔒 AI-Powered Network Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ensemble Classical Models + Deep Autoencoder</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This system uses an **ensemble of 4 machine learning models** to detect network anomalies:
    - **Isolation Forest** - Detects anomalies by isolating outliers
    - **One-Class SVM** - Learns the boundary of normal behavior
    - **Local Outlier Factor** - Identifies local density deviations
    - **Deep Autoencoder** - Neural network reconstruction-based detection
    """)
    
    if 'model_trained' not in st.session_state:
        model, is_loaded = load_or_create_model()
        if is_loaded:
            st.session_state['model'] = model
            st.session_state['model_trained'] = True
        else:
            st.session_state['model_trained'] = False
    
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home & Upload", "🤖 Model Training", "📊 Detection & Analysis", "🔍 Explainability", "📝 Feedback & Retraining", "📖 Documentation"]
    )
    
    feature_cols = ['packet_size', 'duration', 'src_bytes', 'dst_bytes', 
                   'protocol_type', 'flag', 'src_port', 'dst_port', 
                   'packet_rate', 'error_rate']
    
    if page == "🏠 Home & Upload":
        st.header("📁 Data Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Network Traffic Data")
            uploaded_file = st.file_uploader(
                "Upload CSV file with network traffic data",
                type=['csv'],
                help="Upload a CSV file containing network traffic features"
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state['data'] = df
                st.success(f"✅ Loaded {len(df)} samples")
                st.dataframe(df.head(10))
        
        with col2:
            st.subheader("Use Sample Dataset")
            if st.button("📥 Load Sample Network Traffic Data"):
                if os.path.exists('sample_network_traffic.csv'):
                    df = pd.read_csv('sample_network_traffic.csv')
                    st.session_state['data'] = df
                    st.success(f"✅ Loaded {len(df)} sample records")
                    st.dataframe(df.head(10))
                else:
                    st.error("Sample data not found. Please generate it first using generate_sample_data.py")
        
        if 'data' in st.session_state:
            st.subheader("📊 Dataset Overview")
            df = st.session_state['data']
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Samples", len(df))
            col2.metric("Features", len(feature_cols))
            
            if 'label' in df.columns:
                col3.metric("Normal Traffic", sum(df['label'] == 0))
                col4.metric("Known Anomalies", sum(df['label'] == 1))
    
    elif page == "🤖 Model Training":
        st.header("🤖 Model Training")
        
        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first from the Home page")
            return
        
        df = st.session_state['data']
        
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            contamination = st.slider(
                "Expected Anomaly Ratio",
                min_value=0.05,
                max_value=0.3,
                value=0.15,
                step=0.05,
                help="Expected proportion of anomalies in the dataset"
            )
        
        with col2:
            use_full_data = st.checkbox(
                "Train on full dataset",
                value=True,
                help="Train on all data or just normal traffic (if labels available)"
            )
        
        if st.session_state.get('model_trained', False) and 'model' in st.session_state:
            st.info("ℹ️ Pre-trained models are loaded. You can use them for detection or retrain with new data.")
        
        if st.button("🚀 Train Ensemble Models", type="primary"):
            if use_full_data or 'label' not in df.columns:
                train_df = df
            else:
                train_df = df[df['label'] == 0]
            
            st.info(f"Training on {len(train_df)} samples...")
            
            model = train_models(train_df, feature_cols)
            st.session_state['model'] = model
            st.session_state['model_trained'] = True
            
            st.success("✅ All models trained successfully!")
            st.balloons()
    
    elif page == "📊 Detection & Analysis":
        st.header("📊 Anomaly Detection & Analysis")
        
        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first")
            return
        
        if not st.session_state.get('model_trained', False):
            st.warning("⚠️ No trained model found. Please train the model first in the Model Training page.")
            return
        
        model = st.session_state['model']
        df = st.session_state['data'].copy()
        
        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Analyzing network traffic..."):
                X = df[feature_cols].values
                
                predictions, confidence, severity, individual_preds = model.predict_ensemble(X)
                
                df['prediction'] = predictions
                df['confidence'] = confidence
                df['severity'] = severity
                
                st.session_state['predictions'] = df
                st.session_state['individual_preds'] = individual_preds
        
        if 'predictions' in st.session_state:
            df = st.session_state['predictions']
            
            st.subheader("🎯 Detection Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_anomalies = sum(df['prediction'] == -1)
            anomaly_rate = (total_anomalies / len(df)) * 100
            avg_confidence = df['confidence'].mean()
            
            high_severity = sum((df['prediction'] == -1) & (df['severity'] == 'high'))
            
            col1.metric("Total Anomalies (Ensemble)", total_anomalies)
            col2.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
            col3.metric("Avg Confidence", f"{avg_confidence:.2f}")
            col4.metric("High Severity", high_severity, delta_color="inverse")
            
            st.info("""
            **📊 How Ensemble Works:** The final anomaly count ({}) shown above comes from **majority voting** where at least 2 out of 4 models must agree.  
            Each individual model may detect different numbers - check the Model Comparison tab to see!
            """.format(total_anomalies))
            
            st.subheader("📈 Visualizations")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Timeline", "Model Comparison", "Voting Breakdown", "Severity", "Feature Distributions"])
            
            with tab1:
                st.markdown("**Anomaly Timeline:** Shows when anomalies were detected and their confidence scores")
                st.plotly_chart(plot_anomaly_timeline(df), use_container_width=True)
            
            with tab2:
                st.markdown("**Individual vs Ensemble:** Each model detects different anomalies independently. The ensemble combines their votes.")
                st.plotly_chart(plot_model_comparison(st.session_state['individual_preds'], total_anomalies), use_container_width=True)
                
                individual_preds = st.session_state['individual_preds']
                st.markdown("#### 📋 Model Detection Summary:")
                cols = st.columns(5)
                for idx, (model_name, preds) in enumerate(individual_preds.items()):
                    count = np.sum(preds == -1)
                    cols[idx].metric(model_name.replace('_', ' ').title(), count)
                cols[4].metric("**Ensemble Final**", total_anomalies, help="At least 2 models agreed")
            
            with tab3:
                st.markdown("**Voting Breakdown:** Shows how many models agreed on each detected anomaly")
                if total_anomalies > 0:
                    st.plotly_chart(plot_voting_breakdown(df), use_container_width=True)
                    
                    vote_2 = sum((df['prediction'] == -1) & (df['severity'] == 'low'))
                    vote_3 = sum((df['prediction'] == -1) & (df['severity'] == 'medium'))
                    vote_4 = sum((df['prediction'] == -1) & (df['severity'] == 'high'))
                    
                    st.markdown(f"""
                    - **2 models agreed:** {vote_2} anomalies (Low severity)
                    - **3 models agreed:** {vote_3} anomalies (Medium severity)  
                    - **4 models agreed:** {vote_4} anomalies (High severity - all models detected!)
                    - **Total:** {vote_2 + vote_3 + vote_4} anomalies
                    """)
                    
                    verification_sum = vote_2 + vote_3 + vote_4
                    if verification_sum == total_anomalies:
                        st.success(f"✅ **Verification Passed:** Math checks out! {vote_2} + {vote_3} + {vote_4} = {total_anomalies}")
                    else:
                        st.error(f"❌ Mismatch detected: {verification_sum} ≠ {total_anomalies}")
                else:
                    st.info("No anomalies detected")
            
            with tab4:
                st.markdown("**Severity Distribution:** Breakdown of anomaly severity levels")
                if total_anomalies > 0:
                    st.plotly_chart(plot_severity_distribution(df), use_container_width=True)
                else:
                    st.info("No anomalies detected")
            
            with tab5:
                st.markdown("**Feature Distributions:** Compare how features differ between normal and anomalous traffic")
                st.plotly_chart(plot_feature_distributions(df, feature_cols), use_container_width=True)
            
            st.subheader("🔴 Detected Anomalies")
            anomaly_df = df[df['prediction'] == -1].copy()
            
            if len(anomaly_df) > 0:
                st.dataframe(
                    anomaly_df[['timestamp'] + feature_cols + ['confidence', 'severity'] if 'timestamp' in anomaly_df.columns 
                              else feature_cols + ['confidence', 'severity']].head(50),
                    use_container_width=True
                )
                
                csv_buffer = io.StringIO()
                anomaly_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Anomaly Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No anomalies detected in the dataset")
            
            if 'label' in df.columns:
                st.subheader("📊 Performance Metrics")
                
                from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
                
                y_true = np.where(df['label'] == 1, -1, 1)
                y_pred = df['prediction'].values
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
                recall = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
                f1 = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.3f}")
                col2.metric("Precision", f"{precision:.3f}")
                col3.metric("Recall", f"{recall:.3f}")
                col4.metric("F1-Score", f"{f1:.3f}")
                
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Normal', 'Anomaly'],
                           yticklabels=['Normal', 'Anomaly'])
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
    
    elif page == "🔍 Explainability":
        st.header("🔍 Model Explainability")
        
        if 'predictions' not in st.session_state:
            st.warning("⚠️ Please run anomaly detection first")
            return
        
        st.markdown("""
        Understanding **why** a sample was classified as an anomaly is crucial for:
        - Building trust in the system
        - Identifying attack patterns
        - Reducing false positives
        """)
        
        df = st.session_state['predictions']
        model = st.session_state.get('model')
        
        if model is None:
            st.warning("⚠️ Model not loaded")
            return
        
        st.subheader("📊 Feature Importance Analysis")
        
        if st.button("🔬 Calculate Feature Importance"):
            with st.spinner("Analyzing feature importance..."):
                sample_size = min(200, len(df))
                sample_df = df.sample(n=sample_size, random_state=42)
                
                importance_dict = calculate_feature_importance(
                    model,
                    sample_df[feature_cols].values,
                    feature_cols
                )
                
                st.session_state['feature_importance'] = importance_dict
        
        if 'feature_importance' in st.session_state:
            importance_dict = st.session_state['feature_importance']
            
            importance_df = pd.DataFrame(
                list(importance_dict.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Feature Importance for Anomaly Detection",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 💡 Interpretation")
            top_features = importance_df.head(3)
            st.write(f"The top 3 most important features for anomaly detection are:")
            for idx, row in top_features.iterrows():
                st.write(f"- **{row['Feature']}**: {row['Importance']:.4f}")
        
        st.subheader("🔎 Individual Sample Analysis")
        
        anomaly_df = df[df['prediction'] == -1]
        if len(anomaly_df) > 0:
            selected_idx = st.selectbox(
                "Select an anomaly to analyze",
                anomaly_df.index,
                format_func=lambda x: f"Sample {x} - Severity: {df.loc[x, 'severity']}, Confidence: {df.loc[x, 'confidence']:.2f}"
            )
            
            sample = df.loc[selected_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Sample Details")
                st.write(f"**Confidence:** {sample['confidence']:.3f}")
                st.write(f"**Severity:** {sample['severity']}")
                
                if 'individual_preds' in st.session_state:
                    st.markdown("### Model Votes")
                    for model_name, preds in st.session_state['individual_preds'].items():
                        pred_label = "🔴 Anomaly" if preds[selected_idx] == -1 else "🟢 Normal"
                        st.write(f"**{model_name}**: {pred_label}")
            
            with col2:
                st.markdown("### Feature Values")
                feature_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Value': [sample[f] for f in feature_cols]
                })
                st.dataframe(feature_df, use_container_width=True)
    
    elif page == "📝 Feedback & Retraining":
        st.header("📝 Feedback & Retraining")
        
        st.markdown("""
        Improve model performance by:
        1. Labeling detected anomalies as true or false positives
        2. Retraining the model with corrected labels
        3. Iteratively improving detection accuracy
        """)
        
        if 'predictions' not in st.session_state:
            st.warning("⚠️ Please run anomaly detection first")
            return
        
        df = st.session_state['predictions'].copy()
        
        st.subheader("🏷️ Label Anomalies")
        
        anomaly_df = df[df['prediction'] == -1].copy()
        
        if len(anomaly_df) > 0:
            st.write(f"Found {len(anomaly_df)} anomalies to review")
            
            if 'feedback_labels' not in st.session_state:
                st.session_state['feedback_labels'] = {}
            
            for idx in anomaly_df.index[:10]:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Sample {idx}** - Severity: {df.loc[idx, 'severity']}, Confidence: {df.loc[idx, 'confidence']:.2f}")
                
                with col2:
                    if st.button(f"✅ True Positive", key=f"tp_{idx}"):
                        st.session_state['feedback_labels'][idx] = 1
                        st.success("Marked as True Positive")
                
                with col3:
                    if st.button(f"❌ False Positive", key=f"fp_{idx}"):
                        st.session_state['feedback_labels'][idx] = 0
                        st.success("Marked as False Positive")
            
            st.markdown("---")
            st.subheader("📊 Feedback Summary")
            
            if len(st.session_state['feedback_labels']) > 0:
                true_positives = sum(1 for v in st.session_state['feedback_labels'].values() if v == 1)
                false_positives = sum(1 for v in st.session_state['feedback_labels'].values() if v == 0)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Labeled Samples", len(st.session_state['feedback_labels']))
                col2.metric("True Positives", true_positives)
                col3.metric("False Positives", false_positives)
                
                if st.button("🔄 Retrain Model with Feedback", type="primary"):
                    st.info("🚧 Retraining with user feedback...")
                    
                    corrected_df = df.copy()
                    for idx, label in st.session_state['feedback_labels'].items():
                        corrected_df.loc[idx, 'label'] = label
                    
                    normal_data = corrected_df[corrected_df['label'] == 0]
                    
                    if len(normal_data) > 50:
                        model = train_models(normal_data, feature_cols)
                        st.session_state['model'] = model
                        st.success("✅ Model retrained with your feedback!")
                        st.session_state['feedback_labels'] = {}
                    else:
                        st.error("Not enough normal samples to retrain. Please label more samples.")
            else:
                st.info("No feedback provided yet. Label some anomalies above.")
        else:
            st.info("No anomalies detected to review")
    
    elif page == "📖 Documentation":
        st.header("📖 Project Documentation")
        
        st.markdown("""
        ## AI-Powered Network Anomaly Detection System
        
        ### 🎯 Overview
        This project implements a comprehensive network anomaly detection system using an **ensemble of machine learning models** 
        combined with a **deep autoencoder** for robust, accurate detection of malicious network activities.
        
        ### 🔧 System Architecture
        
        #### 1. **Ensemble Components**
        
        **Isolation Forest**
        - Tree-based anomaly detection algorithm
        - Isolates anomalies by randomly selecting features
        - Fast and efficient for high-dimensional data
        - Works well with contaminated datasets
        
        **One-Class SVM**
        - Learns the boundary of normal behavior
        - Uses kernel methods for non-linear patterns
        - Effective for complex decision boundaries
        - Good generalization to unseen data
        
        **Local Outlier Factor (LOF)**
        - Density-based anomaly detection
        - Identifies local density deviations
        - Detects anomalies in varying density regions
        - Captures local anomalies missed by global methods
        
        **Deep Autoencoder**
        - Neural network with encoder-decoder architecture
        - Learns compressed representation of normal traffic
        - Detects anomalies via reconstruction error
        - Captures complex non-linear patterns
        
        #### 2. **Ensemble Voting System**
        - **Majority Voting**: Anomaly if ≥2 models agree
        - **Confidence Score**: Proportion of models agreeing (0-1)
        - **Severity Levels**:
          - **Low**: 2 models detect anomaly
          - **Medium**: 3 models detect anomaly
          - **High**: All 4 models detect anomaly
        
        ### 📊 Features
        
        The system analyzes 10 network traffic features:
        1. **packet_size**: Network packet size in bytes
        2. **duration**: Connection duration in seconds
        3. **src_bytes**: Bytes sent from source
        4. **dst_bytes**: Bytes sent to destination
        5. **protocol_type**: Network protocol (TCP/UDP/ICMP)
        6. **flag**: Connection flag status
        7. **src_port**: Source port number
        8. **dst_port**: Destination port number
        9. **packet_rate**: Packets per second
        10. **error_rate**: Percentage of error packets
        
        ### 🚀 How to Use
        
        #### Step 1: Upload Data
        - Navigate to **Home & Upload** page
        - Upload your CSV file with network traffic data
        - Or load the provided sample dataset
        
        #### Step 2: Train Models
        - Go to **Model Training** page
        - Configure contamination ratio (expected anomaly proportion)
        - Click "Train Ensemble Models"
        - Wait for training to complete (~1-2 minutes)
        
        #### Step 3: Detect Anomalies
        - Navigate to **Detection & Analysis** page
        - Click "Detect Anomalies"
        - View results, visualizations, and metrics
        - Download anomaly reports
        
        #### Step 4: Understand Results
        - Go to **Explainability** page
        - Calculate feature importance
        - Analyze individual anomaly samples
        - Understand why samples were flagged
        
        #### Step 5: Provide Feedback
        - Navigate to **Feedback & Retraining** page
        - Label detected anomalies as true/false positives
        - Retrain model with corrected labels
        - Improve detection accuracy iteratively
        
        ### 📈 Performance Metrics
        
        The system provides comprehensive metrics when ground truth labels are available:
        - **Accuracy**: Overall correctness
        - **Precision**: True positives / (True positives + False positives)
        - **Recall**: True positives / (True positives + False negatives)
        - **F1-Score**: Harmonic mean of precision and recall
        - **Confusion Matrix**: Detailed breakdown of predictions
        
        ### 🔍 Explainability
        
        The system uses **permutation-based feature importance** to explain:
        - Which features contribute most to anomaly detection
        - Why specific samples were classified as anomalies
        - How each model voted on individual samples
        
        ### 🔄 Feedback Loop
        
        Continuous improvement through:
        1. **User Labeling**: Mark anomalies as true/false positives
        2. **Data Correction**: Update dataset with verified labels
        3. **Model Retraining**: Train on corrected data
        4. **Performance Improvement**: Reduce false positives over time
        
        ### 💡 Key Benefits
        
        ✅ **Robust Detection**: Ensemble approach reduces false positives
        ✅ **Explainable**: Understand why anomalies are detected
        ✅ **Adaptive**: Improves with user feedback
        ✅ **Real-time**: Fast prediction on streaming data
        ✅ **Comprehensive**: Multiple visualization and analysis tools
        
        ### 🛠️ Technical Stack
        
        - **Frontend**: Streamlit
        - **ML Models**: Scikit-learn (IF, SVM, LOF)
        - **Deep Learning**: TensorFlow/Keras (Autoencoder)
        - **Visualization**: Plotly, Matplotlib, Seaborn
        - **Data Processing**: Pandas, NumPy
        
        ### 📝 Dataset Format
        
        Your CSV should contain these columns:
        ```
        packet_size, duration, src_bytes, dst_bytes, protocol_type, 
        flag, src_port, dst_port, packet_rate, error_rate, [label]
        ```
        
        The `label` column is optional (0=normal, 1=anomaly) and used for evaluation.
        
        ### 🎓 For College Project Submission
        
        This project demonstrates:
        - **Machine Learning**: Multiple algorithms and ensemble methods
        - **Deep Learning**: Autoencoder architecture
        - **Data Science**: Feature engineering, visualization, analysis
        - **Software Engineering**: Modular design, clean code
        - **UI/UX**: Interactive dashboard with Streamlit
        - **Explainable AI**: Feature importance and interpretability
        - **Continuous Learning**: Feedback loop and retraining
        
        ### 📚 References
        
        - Isolation Forest: Liu et al., 2008
        - One-Class SVM: Schölkopf et al., 2001
        - Local Outlier Factor: Breunig et al., 2000
        - Autoencoders for Anomaly Detection: Sakurada & Yairi, 2014
        
        ### 👨‍💻 Development
        
        **Author**: College Minor Project
        **Purpose**: Network Security & Anomaly Detection
        **Year**: 2025
        
        ---
        
        ### 🚀 Deployment
        
        This app is ready for deployment on Streamlit Cloud:
        1. Push code to GitHub
        2. Connect repository to Streamlit Cloud
        3. Deploy with one click
        4. Share with anyone via public URL
        
        """)

if __name__ == "__main__":
    main()
