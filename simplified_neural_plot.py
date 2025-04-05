import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from matplotlib.patches import ConnectionPatch
import scipy.stats as stats

# Create model-dataset color scheme
def create_model_type_color_scheme():
    # Model type colors
    model_type_colors = {
        'CNN': '#4CAF50',          # Green
        'ViT': '#1E88E5',          # Blue
        'RNN': '#F44336',          # Red
        'Adversarial': '#fbbc05',  # Yellow
        'ConvViT': '#9C27B0',      # Purple
        'Unknown': '#999999'       # Gray
    }
    
    return model_type_colors

# Create the main plotting function
def create_plot(df, x_column, y_column='MajajHong2015.IT-pls', title=None, save_path=None, ylim=None, xlim=None, annotate_outliers=False):
    # Set publication-quality style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 14,
        'axes.linewidth': 1.0,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'lines.linewidth': 2.0,
        'patch.linewidth': 1.0,
        'savefig.dpi': 600,
        'savefig.format': 'pdf'
    })
    
    # Create figure with two subplots - main plot and legend
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
    
    # Main plot subplot
    ax_main = plt.subplot(gs[0])
    
    # Add human accuracy patch for ImageNet-top1 plots if needed
    if x_column == 'ImageNet-top1':
        # Get current y-axis limits
        y_min, y_max = ax_main.get_ylim() if ylim is None else ylim
        
        # Create a light gray patch for human accuracy range
        human_acc_min = 0.919
        human_acc_max = 0.973
        
        # Create the patch
        rect = plt.Rectangle(
            (human_acc_min, y_min), 
            human_acc_max - human_acc_min, 
            y_max - y_min,
            facecolor='#E0E0E0',
            alpha=0.3,
            edgecolor=None,
            zorder=1
        )
        ax_main.add_patch(rect)
        
        # Add text label
        text_x = (human_acc_min + human_acc_max) / 2
        text_y = y_max * 0.95
        ax_main.text(
            text_x,
            text_y,
            'Human accuracy',
            fontsize=12,
            color='#505050',
            ha='center',
            va='top',
            style='italic',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=None, pad=2),
            zorder=1
        )
    
    # Create color mapping based on model type
    color_dict = create_model_type_color_scheme()
    
    # Define marker styles
    marker_styles = {
        'vision': 's',  # Square for vision-only models
        'VLM': 'd',     # Diamond for vision & language models
        'unknown': 'o'  # Circle as fallback
    }
    
    # Group by model_type for plotting
    for model_type, group in df.groupby('model_type'):
        if len(group) > 0:
            ax_main.scatter(
                group[x_column],
                group[y_column],
                color=color_dict.get(model_type, '#999999'),
                marker=marker_styles.get(group['vision_or_VLM'].iloc[0], 'o'),
                alpha=0.8,  
                s=80,
                label=model_type,
                edgecolor='black',  
                linewidth=0.5,
                zorder=5
            )
    
    # Add regression line
    valid_data = df[[x_column, y_column]].dropna()
    if len(valid_data) >= 2:
        x = valid_data[x_column].values.reshape(-1, 1)
        y = valid_data[y_column].values
        model = LinearRegression().fit(x, y)
        r_squared = model.score(x, y)
        
        # Plot regression line
        x_range = np.linspace(valid_data[x_column].min(), valid_data[x_column].max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        ax_main.plot(x_range, y_pred, color='black', linestyle='--', linewidth=2)
        
        # Add R² annotation
        ax_main.text(
            0.05, 0.95, 
            f'R² = {r_squared:.2f}', 
            transform=ax_main.transAxes,
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=None, pad=3)
        )
    
    # Annotate outliers if requested
    if annotate_outliers:
        # Identify outliers as top models in terms of y-value
        top_indices = df.nlargest(5, y_column).index
        
        # Define a function to shorten model names for annotation
        def get_short_name(full_name):
            # Remove everything after first underscore or hyphen
            name = str(full_name).split('_')[0].split('-')[0]
            return name
        
        # Annotate outliers
        for idx in top_indices:
            row = df.loc[idx]
            ax_main.annotate(
                get_short_name(row['model_name']),
                (row[x_column], row[y_column]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    
    # Set labels with better descriptive names
    if x_column == 'ImageNet-top1':
        x_label = 'ImageNet Accuracy'
    else:
        x_label = x_column.replace('_', ' ').title()
        
    if y_column == 'MajajHong2015.IT-pls':
        y_label = 'Brain Score (IT)'
    else:
        y_label = y_column.replace('_', ' ').title()
        
    ax_main.set_xlabel(x_label, fontsize=16)
    ax_main.set_ylabel(y_label, fontsize=16)
    
    # Set title if provided
    if title:
        ax_main.set_title(title, fontsize=18)
    else:
        ax_main.set_title(f'{y_label} vs {x_label}', fontsize=18)
    
    # Set axis limits if provided
    if xlim:
        ax_main.set_xlim(xlim)
    if ylim:
        ax_main.set_ylim(ylim)
    
    # Create legend subplot
    ax_legend = plt.subplot(gs[1])
    ax_legend.axis('off')
    
    # Create custom legend handles and labels
    handles = []
    labels = []
    
    # Group by model type for legend
    model_types = sorted(df['model_type'].unique())
    
    for model_type in model_types:
        group_data = df[df['model_type'] == model_type]
        if len(group_data) > 0:
            # Get marker style based on vision_or_VLM
            vlm_value = group_data['vision_or_VLM'].iloc[0]
            marker = marker_styles.get(vlm_value, 'o')
            
            # Add to handles and labels
            handles.append(plt.Line2D(
                [0], [0], 
                marker=marker, 
                color='w', 
                markerfacecolor=color_dict.get(model_type, '#999999'),
                markeredgecolor='black',
                markersize=10, 
                alpha=0.8, 
                linewidth=0
            ))
            labels.append(model_type)
    
    # Create the legend
    ax_legend.legend(
        handles, 
        labels, 
        loc='center left',
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        handletextpad=0.5,
        markerscale=1.5,
        title='Model Types',
        title_fontsize=14
    )
    
    # Remove spines
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # Add a light grid
    ax_main.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    
    plt.show()

# Main script execution
if __name__ == "__main__":
    # Load the data
    neural_data = pd.read_csv('csvs/brainscore_coded.csv')
    
    # Create output directory if it doesn't exist
    output_dir = 'simplified_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert 'X' values in ImageNet-top1 to NaN
    neural_data['ImageNet-top1'] = pd.to_numeric(neural_data['ImageNet-top1'], errors='coerce')
    
    # Create scatter plot of Brain Score vs ImageNet Accuracy
    create_plot(
        neural_data,
        'ImageNet-top1', 
        'MajajHong2015.IT-pls',
        title='Brain Score vs ImageNet Accuracy', 
        save_path=os.path.join(output_dir, 'brain_score_vs_imagenet.png'),
        ylim=(0.5, 0.6),
        annotate_outliers=True
    )
    
    # Add analysis by model type
    print("\nAverage Brain Score by Model Type:")
    model_type_stats = neural_data.groupby('model_type')['MajajHong2015.IT-pls'].agg(['mean', 'std', 'count'])
    print(model_type_stats)
    
    # Calculate correlation between ImageNet accuracy and Brain Score
    valid_data = neural_data[['ImageNet-top1', 'MajajHong2015.IT-pls']].dropna()
    correlation = valid_data.corr().iloc[0, 1]
    print(f"\nCorrelation between ImageNet Accuracy and Brain Score: {correlation:.3f}")
    
    # Calculate correlation for each model type
    print("\nCorrelation by Model Type:")
    for model_type, group in neural_data.groupby('model_type'):
        valid_group = group[['ImageNet-top1', 'MajajHong2015.IT-pls']].dropna()
        if len(valid_group) >= 5:  # Only calculate if enough data points
            type_corr = valid_group.corr().iloc[0, 1]
            print(f"{model_type}: {type_corr:.3f} (n={len(valid_group)})")
    
    print("\nAnalysis complete!") 