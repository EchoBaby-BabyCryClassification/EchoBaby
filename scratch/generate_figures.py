import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# Ensure the output directory exists
out_dir = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_report\figures"
os.makedirs(out_dir, exist_ok=True)

# Set common style
plt.style.use('ggplot')

# 1. class_distribution_after_deduplication.png
classes = ['Belly Pain', 'Burping', 'Discomfort\n/ Cold-Hot', 'Hungry', 'Tired']
counts = [126, 78, 166, 239, 25]

plt.figure(figsize=(10, 6))
sns.barplot(x=classes, y=counts, palette='viridis')
plt.title("Class Distribution After Deduplication", fontsize=16)
plt.ylabel("Number of Samples", fontsize=12)
plt.xlabel("Acoustic Class", fontsize=12)
for i, v in enumerate(counts):
    plt.text(i, v + 2, str(v), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "class_distribution_after_deduplication.png"), dpi=300)
plt.close()

# 2. deduplication_impact.png
labels = ['Raw Audio Files\nBefore Cleaning', 'Unique Valid Files\nAfter Deduplication']
values = [1816, 634]

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=values, palette='mako')
plt.title("Impact of MD5-Based Deduplication", fontsize=16)
plt.ylabel("Number of Files", fontsize=12)
for i, v in enumerate(values):
    plt.text(i, v + 20, str(v), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "deduplication_impact.png"), dpi=300)
plt.close()

# 3. model_comparison_macro_metrics.png
models = ['Extra Trees', 'Random Forest', 'HistGradient\nBoosting', 'SVC', 'Logistic\nRegression']
macro_f1 = [0.468, 0.450, 0.425, 0.409, 0.346]
macro_recall = [0.448, 0.430, 0.410, 0.419, 0.391]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, macro_f1, width, label='Macro-F1', color='#4c72b0')
rects2 = ax.bar(x + width/2, macro_recall, width, label='Macro-Recall', color='#dd8452')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Cross-Validation Macro-F1 and Macro-Recall by Model', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend()
ax.set_ylim(0, 0.6)

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "model_comparison_macro_metrics.png"), dpi=300)
plt.close()

# 4. final_confusion_matrix.png
cm = np.array([
    [14, 0, 0, 10, 1],
    [4, 6, 0, 6, 0],
    [4, 0, 8, 21, 0],
    [11, 0, 0, 37, 0],
    [2, 0, 1, 0, 2]
])
cm_labels = ['Belly Pain', 'Burping', 'Discomfort', 'Hungry', 'Tired']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels, cbar=False, annot_kws={"size": 14, "weight": "bold"})
plt.title("Final Soft Voting Ensemble Confusion Matrix", fontsize=16)
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "final_confusion_matrix.png"), dpi=300)
plt.close()

# Function to draw horizontal flowcharts
def draw_flowchart(steps, title, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    plt.title(title, fontsize=16, y=0.9)
    
    n_steps = len(steps)
    x_positions = np.linspace(0.1, 0.9, n_steps)
    box_width = 0.75 / n_steps
    
    for i, (x, step) in enumerate(zip(x_positions, steps)):
        words = step.split()
        if len(words) > 2:
            step_text = "\\n".join([" ".join(words[:len(words)//2]), " ".join(words[len(words)//2:])])
        else:
            step_text = "\\n".join(words)
            
        box = patches.FancyBboxPatch((x - box_width/2, 0.4), box_width, 0.2,
                                     boxstyle="round,pad=0.05",
                                     ec="#2b2b2b", fc="#e1e1e1", lw=1.5)
        ax.add_patch(box)
        ax.text(x, 0.5, step_text, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw arrows
        if i < n_steps - 1:
            ax.annotate('', xy=(x_positions[i+1] - box_width/2 - 0.01, 0.5), 
                        xytext=(x + box_width/2 + 0.01, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#2b2b2b"))
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()

# 5. v1_pipeline_diagram.png
v1_steps = [
    "Raw Audio Files",
    "MD5 Duplicate Detection",
    "Label Conflict Resolution",
    "Audio Standardization",
    "Feature Extraction",
    "StratifiedGroup KFold",
    "Model Training",
    "Evaluation"
]
draw_flowchart(v1_steps, "Version 1 Pipeline Architecture", "v1_pipeline_diagram.png")

# 6. v2_roadmap_diagram.png
v2_steps = [
    "New Dataset Collection",
    "Audio Format Standardization",
    "Cross-Dataset Deduplication",
    "Expert Relabeling",
    "Audio Augmentation",
    "Retraining",
    "V1 vs V2 Comparison"
]
draw_flowchart(v2_steps, "Version 2 Planned Roadmap", "v2_roadmap_diagram.png")

print("All figures generated successfully.")
