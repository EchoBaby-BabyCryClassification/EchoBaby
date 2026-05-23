import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

out_dir = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_report\figures"
os.makedirs(out_dir, exist_ok=True)

def draw_vertical_flowchart(steps, title, filename):
    n_steps = len(steps)
    fig, ax = plt.subplots(figsize=(6, n_steps * 1.5 + 1))
    ax.axis('off')
    plt.title(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Calculate positions
    y_positions = [1 - (i+1.5)/(n_steps+1.5) for i in range(n_steps)]
    
    box_height = 0.5 / n_steps
    box_width = 0.7
    
    for i, (y, step) in enumerate(zip(y_positions, steps)):
        box = patches.FancyBboxPatch(
            (0.5 - box_width/2, y - box_height/2), 
            box_width, box_height,
            boxstyle="round,pad=0.08",
            ec="#2b2b2b", fc="#f2f2f2", lw=2
        )
        ax.add_patch(box)
        
        # Word wrap manually if too long
        words = step.split()
        if len(words) > 3:
            step_text = "\n".join([" ".join(words[:len(words)//2]), " ".join(words[len(words)//2:])])
        else:
            step_text = step
            
        ax.text(0.5, y, step_text, ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Draw arrows downwards
        if i < n_steps - 1:
            ax.annotate(
                '', 
                xy=(0.5, y_positions[i+1] + box_height/2 + 0.02), 
                xytext=(0.5, y - box_height/2 - 0.02),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2b2b2b")
            )
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
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
draw_vertical_flowchart(v1_steps, "Version 1 Pipeline Architecture", "v1_pipeline_diagram.png")

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
draw_vertical_flowchart(v2_steps, "Version 2 Planned Roadmap", "v2_roadmap_diagram.png")

print("Flowcharts generated successfully in vertical layout.")
