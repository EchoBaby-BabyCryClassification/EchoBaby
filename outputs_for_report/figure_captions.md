# Applied Machine Learning Final Report: Figure Captions

**Figure 1: Class Distribution After Deduplication**
Distribution of the 634 unique audio samples across the five target acoustic classes after MD5 hash deduplication. The chart highlights the severe class imbalance present in the curated dataset, with the "Hungry" class dominating the distribution and "Tired" representing a significant minority.

**Figure 2: Impact of MD5-Based Deduplication**
Comparison of the total dataset size before and after the programmatic data cleaning phase. The application of MD5 hash-based deduplication revealed that approximately 77% of the original 1,816 audio files were exact binary copies, reducing the dataset to 634 unique, valid samples.

**Figure 3: Cross-Validation Macro-F1 and Macro-Recall by Model**
Performance comparison of five baseline tabular machine learning models utilizing a 5-fold `StratifiedGroupKFold` cross-validation strategy. The Extra Trees classifier demonstrated the highest individual predictive capability across both Macro-F1 and Macro-Recall metrics.

**Figure 4: Final Soft Voting Ensemble Confusion Matrix**
Hold-out evaluation confusion matrix for the deployed Soft Voting Ensemble (70% Extra Trees, 30% SVC). The matrix visualizes the model's predictive performance and highlights persistent error patterns, primarily the over-prediction of the majority "Hungry" class and deliberate threshold-induced sensitivity towards the "Belly Pain" class.

**Figure 5: Version 1 Pipeline Architecture**
End-to-end architectural flowchart of the completed Version 1 applied machine learning pipeline. The process sequentially illustrates the transition from raw, noisy audio data through strict hash-based duplicate detection, label conflict resolution, and robust feature extraction, culminating in grouped cross-validation and ensemble model training.

**Figure 6: Version 2 Planned Roadmap**
Proposed strategic roadmap for the upcoming Version 2 architecture. The planned workflow introduces critical improvements including the acquisition of new data, expert pediatric relabeling, raw audio augmentation (e.g., pitch shifting, time stretching), and comprehensive retraining to facilitate a transition towards deep convolutional neural networks.
