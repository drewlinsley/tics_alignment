import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from matplotlib.colors import LogNorm, to_rgb, to_hex
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
import colorsys


plot_data = True


# Create model type categories
def categorize_model_type(model_type):
    model_type = model_type.lower()
    if ('conv' in model_type or 'densenet' in model_type or 'cnn' in model_type or 'resnet' in model_type or 'focal' in model_type):
        if 'vit' in model_type or 'transformer' in model_type or 'former' in model_type:
            return 'ConvViT'
        return 'CNN'
    if 'cvt' in model_type:
        return 'ConvViT'
    elif ('vit' in model_type or 'transformer' in model_type or 'former' in model_type or 'unknown' in model_type):
        return 'ViT'
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Create dataset categories
def categorize_dataset(dataset):
    dataset = str(dataset).lower()
    if any(d in dataset for d in ['yfc-semisl', 'in1k-dist', 'in1k-in1k', 'in1k-ap', 'in1k', 'sw', 'sail', 'fb', 'msft', 'tv', 'mx', 'snap', 'ra', 'ms', 'ft', 'lamb', 'miil', 'rvgg', 'a3', 'c3', 'a1h', 'am', 'tf', 'gluon', 'nav', 'augreg', 'cvnets']):
        return 'ImageNet-1K'
    elif any(d in dataset for d in ['lvd142m', 'laion2b', 'mx_in1k', 'openai', 'mim', 'fcmae', 'in12k', 'in22k', 'ig1b-wsl', 'in21k', 'in21k-selfsl', 'ig1b-swsl', 'jft300m-ns', 'orig']):
        return 'Internet-scale'
    elif 'in1k-adv' in dataset:
        return 'Adversarial'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# Function for piecewise linear fit with two segments
def piecewise_linear(x, x0, y0, k1, k2):
    """
    Piecewise linear function with two segments.
    x0, y0: coordinates of the breakpoint
    k1: slope of the first segment
    k2: slope of the second segment
    """
    return np.where(x < x0, y0 + k1 * (x - x0), y0 + k2 * (x - x0))

def compute_piecewise_upper_bound(x, y, bins=8):
    """
    Computes piecewise linear upper bound based on convex hull (Pareto front).
    Returns the curve points and the breakpoint parameters.
    """
    # Sort data by x values
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    print(f"Computing piecewise upper bound with {len(x_sorted)} points")
    
    # First step: bin the data and find max y in each bin to reduce noise
    bins = np.linspace(x_sorted.min(), x_sorted.max(), num=bins)
    bin_indices = np.digitize(x_sorted, bins)
    
    max_points_x = []
    max_points_y = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.any(mask):
            max_points_x.append(x_sorted[mask][np.argmax(y_sorted[mask])])
            max_points_y.append(np.max(y_sorted[mask]))
    
    # Convert to numpy arrays for further processing
    points = np.array(list(zip(max_points_x, max_points_y)))
    
    if len(points) < 3:
        print("Not enough points after binning")
        # Return a simple default model
        x0 = (x_sorted.max() + x_sorted.min()) / 2
        y0 = np.median(y_sorted)
        k1 = 0
        k2 = 0
        return x_sorted, y_sorted, [x0, y0, k1, k2]
    
    # Sort points by x-coordinate
    points = points[points[:, 0].argsort()]
    
    # Compute upper convex hull (Pareto front)
    pareto_indices = []
    current_max_y = float('-inf')
    
    # Manual Pareto front computation
    for i in range(len(points)):
        if points[i, 1] > current_max_y:
            pareto_indices.append(i)
            current_max_y = points[i, 1]
    
    # Get the Pareto front points
    pareto_points = points[pareto_indices]
    
    print(f"Found {len(pareto_points)} points on Pareto front")
    
    # If we have sufficient points on the Pareto front
    if len(pareto_points) > 3:
        # Find the peak in the Pareto front
        max_y_idx = np.argmax(pareto_points[:, 1])
        x0 = pareto_points[max_y_idx, 0]
        y0 = pareto_points[max_y_idx, 1]
        
        print(f"Peak found at x={x0}, y={y0}")
        
        # Now compute separate Pareto fronts for pre and post changepoint
        # First, split all original points
        pre_points = points[points[:, 0] <= x0]
        post_points = points[points[:, 0] > x0]
        
        # Compute Pareto front for pre-changepoint region
        pre_pareto_indices = []
        pre_max_y = float('-inf')
        
        for i in range(len(pre_points)):
            if pre_points[i, 1] > pre_max_y:
                pre_pareto_indices.append(i)
                pre_max_y = pre_points[i, 1]
        
        pre_pareto = pre_points[pre_pareto_indices] if pre_pareto_indices else None
        
        # Compute Pareto front for post-changepoint region
        # For post region, we want decreasing values as x increases
        post_pareto_indices = []
        post_min_x = float('inf')
        
        # Sort by y-coordinate descending
        post_points = post_points[post_points[:, 1].argsort()[::-1]]
        
        for i in range(len(post_points)):
            if post_points[i, 0] < post_min_x:
                post_pareto_indices.append(i)
                post_min_x = post_points[i, 0]
        
        post_pareto = post_points[post_pareto_indices] if post_pareto_indices else None
        
        # Resort post_pareto by x-coordinate
        if post_pareto is not None and len(post_pareto) > 0:
            post_pareto = post_pareto[post_pareto[:, 0].argsort()]
        
        # Fit lines to the separate Pareto fronts
        k1, k2 = None, None
        
        if pre_pareto is not None and len(pre_pareto) > 2:
            try:
                X_pre = sm.add_constant(pre_pareto[:, 0])
                model_pre = sm.OLS(pre_pareto[:, 1], X_pre).fit()
                k1 = model_pre.params[1]
                print(f"Pre-changepoint slope: k1={k1}")
            except Exception as e:
                print(f"Error fitting pre-changepoint: {e}")
        
        if post_pareto is not None and len(post_pareto) > 2:
            try:
                X_post = sm.add_constant(post_pareto[:, 0])
                model_post = sm.OLS(post_pareto[:, 1], X_post).fit()
                k2 = model_post.params[1]
                print(f"Post-changepoint slope: k2={k2}")
            except Exception as e:
                print(f"Error fitting post-changepoint: {e}")
        
        # If either fit failed, fall back to fitting original points
        if k1 is None or k2 is None:
            print("Falling back to original pareto front for fitting")
            
            # Split the original pareto points
            mask_before = pareto_points[:, 0] <= x0
            mask_after = pareto_points[:, 0] > x0
            
            if sum(mask_before) > 2 and k1 is None:
                X_before = sm.add_constant(pareto_points[mask_before, 0])
                model_before = sm.OLS(pareto_points[mask_before, 1], X_before).fit()
                k1 = model_before.params[1]
            
            if sum(mask_after) > 2 and k2 is None:
                X_after = sm.add_constant(pareto_points[mask_after, 0])
                model_after = sm.OLS(pareto_points[mask_after, 1], X_after).fit()
                k2 = model_after.params[1]
        
        # If we still don't have slopes, use the raw data
        if k1 is None or k2 is None:
            print("Falling back to original data for fitting")
            
            # Split all data
            mask_before = x_sorted <= x0
            mask_after = x_sorted > x0
            
            if sum(mask_before) > 2 and k1 is None:
                X_before = sm.add_constant(x_sorted[mask_before])
                model_before = sm.OLS(y_sorted[mask_before], X_before).fit()
                k1 = model_before.params[1]
            
            if sum(mask_after) > 2 and k2 is None:
                X_after = sm.add_constant(x_sorted[mask_after])
                model_after = sm.OLS(y_sorted[mask_after], X_after).fit()
                k2 = model_after.params[1]
        
        # Final fallback for slopes
        if k1 is None:
            k1 = 0.1  # Default positive slope
        if k2 is None:
            k2 = -0.1  # Default negative slope
        
        params = [x0, y0, k1, k2]
        
        # Combine all pareto points for plotting
        all_pareto_x = []
        all_pareto_y = []
        
        if pre_pareto is not None and len(pre_pareto) > 0:
            all_pareto_x.extend(pre_pareto[:, 0])
            all_pareto_y.extend(pre_pareto[:, 1])
        
        if post_pareto is not None and len(post_pareto) > 0:
            all_pareto_x.extend(post_pareto[:, 0])
            all_pareto_y.extend(post_pareto[:, 1])
        
        if len(all_pareto_x) > 0:
            return np.array(all_pareto_x), np.array(all_pareto_y), params
        else:
            return pareto_points[:, 0], pareto_points[:, 1], params
    
    # Fallback to simpler approach
    print("Using fallback approach")
    
    # Find maximum y value in original data
    max_y_idx = np.argmax(y_sorted)
    x0 = x_sorted[max_y_idx]
    y0 = y_sorted[max_y_idx]
    
    # Simple fit to all data
    mask_before = x_sorted <= x0
    mask_after = x_sorted > x0
    
    try:
        if sum(mask_before) > 2:
            X_before = sm.add_constant(x_sorted[mask_before])
            model_before = sm.OLS(y_sorted[mask_before], X_before).fit()
            k1 = model_before.params[1]
        else:
            k1 = 0.1
        
        if sum(mask_after) > 2:
            X_after = sm.add_constant(x_sorted[mask_after])
            model_after = sm.OLS(y_sorted[mask_after], X_after).fit()
            k2 = model_after.params[1]
        else:
            k2 = -0.1
        
        params = [x0, y0, k1, k2]
        return x_sorted, y_sorted, params
    except Exception as e:
        print(f"Error in fallback: {e}")
        return x_sorted, y_sorted, [x0, y0, 0.1, -0.1]  # Hardcoded default

# Create a color scheme matching the image
def create_model_dataset_color_scheme():
    """Create a color mapping where each model type has a color family with distinct variations"""
    # Color families for each model type
    model_color_families = {
        'CNN': ['#E34234', '#EE9888'],  # Red family - darker and lighter
        'ConvViT': ['#2E8540', '#8FD096'],  # Green family - darker and lighter
        'ViT': ['#1E88E5', '#90CAF9'],  # Blue family - darker and lighter
        # 'SSL': ['#F9C232', '#FADA87', '#F6E8B1']  # Yellow family - three shades
    }
    
    # Add colors for Adversarial models - Purple family
    adversarial_colors = {
        'CNN - Adversarial': '#fbbc05',  # Deep yellow
        'ConvViT - Adversarial': '#f5cc56',  # Medium yellow
        'ViT - Adversarial': '#f5dc93',  # Light yellow
    }
    
    # Dataset mapping to color variations
    dataset_to_index = {
        'ImageNet-1K': 0,  # Darker shade
        'Internet-scale vision': 1,  # Lighter shade
        'Internet-scale vision & language': 2  # Third shade (for SSL only)
    }
    
    # Create the full color mapping
    color_mapping = {}
    
    # Get all unique model and dataset types
    model_types = ['CNN', 'ConvViT', 'ViT']
    dataset_types = ['ImageNet-1K', 'Internet-scale vision', 'Internet-scale vision & language']
    
    # Create mappings for all model-dataset combinations
    for model_type in model_types:
        color_family = model_color_families[model_type]
        
        for dataset in dataset_types:
            category = f"{model_type} - {dataset}"
            
            # Get the appropriate index for this dataset
            idx = dataset_to_index[dataset]
            
            # Only SSL has three colors, others have two
            if model_type != 'SSL' and idx >= len(color_family):
                continue
                
            # Only use valid indices
            if idx < len(color_family):
                color_mapping[category] = color_family[idx]
    
    # Add the adversarial colors to the mapping
    color_mapping.update(adversarial_colors)
    
    return color_mapping

# Create a dictionary to store changepoints for each plot
changepoints = {}

# Create a function to make the plots
def create_plot(x_column, y_column='normalized_brain_score', title=None, save_path=None, ylim=None, annotate_outliers=False):
    # Set publication-quality style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Raleway'],
        'font.size': 12,
        'axes.linewidth': 0.8,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 8,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.8,
        'savefig.dpi': 600,
        'savefig.format': 'pdf'
    })
    
    # Create figure with two subplots - main plot and legend
    fig = plt.figure(figsize=(4.5 * 2, 3.0 * 2))
    gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
    
    # Main plot subplot
    ax_main = plt.subplot(gs[0])
    
    # Add human accuracy patch for multi_label_acc plots
    human_accuracy_region = None
    if x_column == 'multi_label_acc':
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
        
        # Store the region where "Human accuracy" text will be
        text_x = (human_acc_min + human_acc_max) / 2
        text_y = y_max * 0.95
        human_accuracy_region = {
            'x': text_x,
            'y': text_y,
            'width': (human_acc_max - human_acc_min),
            'height': 0.05 * (y_max - y_min)  # Approximate text height
        }
        
        # Add text label
        ax_main.text(
            text_x,
            text_y,
            'Human accuracy',
            fontsize=8,
            color='#505050',
            ha='center',
            va='top',
            style='italic',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=None, pad=2),
            zorder=1
        )
    
    # Replace 'selfsup' with 'SSL' in the joined_df
    joined_df['model_type_category'] = joined_df['model_type_category'].replace('selfsup', 'SSL')
    # Update category column to reflect the change
    joined_df['category'] = joined_df['model_type_category'] + ' - ' + joined_df['dataset_category']
    
    # Create color mapping matching the image
    color_dict = create_model_dataset_color_scheme()
    
    # Define marker styles
    marker_styles = {
        'vision': 's',               # Square for vision-only models
        'VLM': 'd',                  # Diamond for vision & language models
        'unknown': 'o'               # Circle as fallback
    }
    
    # Sort categories by model type and dataset type
    model_type_order = ['CNN', 'ConvViT', 'ViT', 'SSL']
    dataset_type_order = ['ImageNet-1K', 'Internet-scale vision', 'Internet-scale vision & language', 'Adversarial']
    
    # Function to get the sort key for a category
    def get_sort_key(category):
        model_type, dataset_type = category.split(' - ', 1)
        return (model_type_order.index(model_type) if model_type in model_type_order else 999,
                dataset_type_order.index(dataset_type) if dataset_type in dataset_type_order else 999)
    
    # Get and sort categories
    categories = sorted(joined_df['category'].unique(), key=get_sort_key)
    
    # Create scatter plot with publication-quality styling
    for cat in categories:
        model_type, dataset_type = cat.split(' - ', 1)
        cat_data = joined_df[joined_df['category'] == cat]
        
        if len(cat_data) > 0:
            ax_main.scatter(
                cat_data[x_column],
                cat_data[y_column],
                color=color_dict.get(cat, '#999999'),
                marker=marker_styles.get(cat_data['vision_or_VLM'].iloc[0], 'o'),
                alpha=0.8,  
                s=60,
                label=cat,
                edgecolor='black',  
                linewidth=0.5,
                zorder=5
            )
    
    # Calculate piecewise linear upper bound
    x_data = joined_df[x_column].values
    y_data = joined_df[y_column].values
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    
    # Store the changepoint for later plotting
    changepoint = None
    
    if sum(valid_indices) > 4:
        _, _, params = compute_piecewise_upper_bound(
            x_data[valid_indices], 
            y_data[valid_indices],
            bins=8
        )
        
        # If we have valid parameters, add the extended lines and change marker
        if params is not None:
            x0, y0, k1, k2 = params
            x_min, x_max = ax_main.get_xlim()
            y_min, y_max = ax_main.get_ylim()
            
            # Calculate intercepts for the equation display
            b1 = y0 - k1 * x0  # y = k1*x + b1
            b2 = y0 - k2 * x0  # y = k2*x + b2
            
            # Extend first segment across x-axis
            x_pre = np.array([x_min, x_max])
            y_pre = y0 + k1 * (x_pre - x0)
            ax_main.plot(x_pre, y_pre, color='black', alpha=0.7, linestyle='--', linewidth=2.0)
            
            # Extend second segment across x-axis
            x_post = np.array([x_min, x_max])
            y_post = y0 + k2 * (x_post - x0)
            ax_main.plot(x_post, y_post, color='black', alpha=0.7, linestyle='--', linewidth=2.0)
            
            # Place white dot with black border at the change point
            ax_main.plot(x0, y0, marker='o', markersize=10, 
                    markerfacecolor='white', markeredgecolor='black', 
                    markeredgewidth=1.5, alpha=1.0, zorder=20)
            
            # Add equations in the top-right corner
            # Format the equations with 2 decimal places
            eq1 = f"y = {k1:.2f}x + {b1:.2f}"
            eq2 = f"y = {k2:.2f}x + {b2:.2f}"
            
            # Add a text box with both equations
            equation_text = f"Pre-threshold: {eq1}\nPost-threshold: {eq2}"
            
            # Place in top-right corner with a white background
            ax_main.text(
                0.98, 0.98, 
                equation_text,
                transform=ax_main.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#CCCCCC')
            )
            
            # Also add equations near the lines themselves
            # For the pre-threshold line - place near the middle of the line
            pre_x = x_min + (x0 - x_min) / 2
            pre_y = b1 + k1 * pre_x
            
            # Adjust position based on slope to avoid overlapping
            y_offset = 0.05 * (y_max - y_min)
            pre_y += y_offset if k1 < 0 else -y_offset
            
            ax_main.text(
                pre_x, pre_y,
                eq1,
                fontsize=8,
                verticalalignment='bottom' if k1 < 0 else 'top',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=None, pad=0.2)
            )
            
            # For the post-threshold line - place near the middle of the visible part
            post_x = x0 + (x_max - x0) / 2
            post_y = b2 + k2 * post_x
            
            # Adjust position based on slope to avoid overlapping
            post_y += y_offset if k2 < 0 else -y_offset
            
            ax_main.text(
                post_x, post_y,
                eq2,
                fontsize=8,
                verticalalignment='bottom' if k2 < 0 else 'top',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=None, pad=0.2)
            )
    
    # Set labels and style for main plot
    if title and ' vs ' in title:
        y_label, x_label = title.split(' vs ')
    else:
        x_label = x_column
        y_label = 'IT cortex Brain Score' if y_column == 'normalized_brain_score' else y_column
    
    ax_main.set_xlabel(x_label, fontsize=10, labelpad=5)
    ax_main.set_ylabel(y_label, fontsize=10, labelpad=5)
    if title:
        ax_main.set_title(title, fontsize=11, pad=8)
    
    # Clean up main plot
    sns.despine(ax=ax_main)
    ax_main.grid(False)
    
    if ylim is not None:
        ax_main.set_ylim(ylim)
    
    # Plot the changepoint LAST to ensure it's on top
    if changepoint:
        x0, y0 = changepoint
        # # Add a highlight circle behind the marker
        # ax_main.plot(x0, y0, marker='o', markersize=16, 
        #             markerfacecolor='none', markeredgecolor='#FF4500', 
        #             markeredgewidth=2.0, alpha=0.7, zorder=19)
        # Add the white dot with black border
        ax_main.plot(x0, y0, marker='o', markersize=10, 
                    markerfacecolor='white', markeredgecolor='black', 
                    markeredgewidth=1.5, alpha=1.0, zorder=20)
    
    if annotate_outliers:
        # Get all valid data points
        valid_data = joined_df[~joined_df[x_column].isna() & ~joined_df[y_column].isna()].copy()
        
        if len(valid_data) > 0:
            # Get current axis limits
            x_min, x_max = ax_main.get_xlim()
            y_min, y_max = ax_main.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Find models at the extremes and strategic positions
            max_x_model = valid_data.loc[valid_data[x_column].idxmax()]
            max_y_model = valid_data.loc[valid_data[y_column].idxmax()]
            
            # Find mid-range models
            x_25 = valid_data[x_column].quantile(0.25)
            x_75 = valid_data[x_column].quantile(0.75)
            mid_x_high_y = valid_data[
                (valid_data[x_column] >= x_25) & 
                (valid_data[x_column] <= x_75)
            ].sort_values(y_column, ascending=False).head(1)
            
            y_25 = valid_data[y_column].quantile(0.25)
            y_75 = valid_data[y_column].quantile(0.75)
            mid_y_high_x = valid_data[
                (valid_data[y_column] >= y_25) & 
                (valid_data[y_column] <= y_75)
            ].sort_values(x_column, ascending=False).head(1)
            
            # Combine selected models
            selected_models = pd.concat([
                pd.DataFrame([max_x_model]),
                pd.DataFrame([max_y_model]),
                mid_x_high_y,
                mid_y_high_x
            ]).drop_duplicates()
            
            # Convert all data points to numpy array for distance calculations
            all_points = np.array(list(zip(valid_data[x_column], valid_data[y_column])))
            
            # Track used positions
            used_positions = []
            
            # Function to find nearest white space
            def find_nearest_white_space(x, y, all_points, used_positions):
                # First try: Look for ideal white space
                min_dist_to_points = 0.1 * np.sqrt(x_range**2 + y_range**2)
                min_dist_to_labels = 0.15 * np.sqrt(x_range**2 + y_range**2)
                
                best_pos = None
                min_dist = float('inf')
                
                # First attempt with strict spacing requirements
                for distance in np.linspace(0.1, 0.3, 5) * np.sqrt(x_range**2 + y_range**2):
                    for angle in np.linspace(0, 2*np.pi, 16):
                        dx = distance * np.cos(angle)
                        dy = distance * np.sin(angle)
                        new_x = x + dx
                        new_y = y + dy
                        
                        # Check if position is within plot bounds with padding
                        padding = 0.05 * min(x_range, y_range)
                        if (new_x < x_min + padding or new_x > x_max - padding or 
                            new_y < y_min + padding or new_y > y_max - padding):
                            continue
                        
                        # Check distance to all points
                        point_dists = np.sqrt(np.sum((all_points - [new_x, new_y])**2, axis=1))
                        if np.min(point_dists) < min_dist_to_points:
                            continue
                        
                        # Check distance to used label positions
                        if any(np.sqrt((new_x - ux)**2 + (new_y - uy)**2) < min_dist_to_labels 
                              for ux, uy in used_positions):
                            continue
                        
                        # Special check for Human Accuracy label region
                        if human_accuracy_region is not None:
                            # Check if potential position overlaps with human accuracy region
                            if (abs(new_x - human_accuracy_region['x']) < human_accuracy_region['width']/2 and
                                abs(new_y - human_accuracy_region['y']) < human_accuracy_region['height']):
                                continue
                        
                        total_dist = np.sqrt(dx**2 + dy**2)
                        if total_dist < min_dist:
                            min_dist = total_dist
                            best_pos = (new_x, new_y)
                
                # If first attempt failed, try again with relaxed constraints
                if best_pos is None:
                    print("First attempt failed, trying with relaxed constraints...")
                    min_dist_to_points *= 0.5  # Reduce minimum distance requirements
                    min_dist_to_labels *= 0.5
                    
                    for distance in np.linspace(0.1, 0.4, 8) * np.sqrt(x_range**2 + y_range**2):
                        for angle in np.linspace(0, 2*np.pi, 24):  # More angles to try
                            dx = distance * np.cos(angle)
                            dy = distance * np.sin(angle)
                            new_x = x + dx
                            new_y = y + dy
                            
                            # Check if position is within plot bounds with minimal padding
                            padding = 0.02 * min(x_range, y_range)  # Reduced padding
                            if (new_x < x_min + padding or new_x > x_max - padding or 
                                new_y < y_min + padding or new_y > y_max - padding):
                                continue
                            
                            # Relaxed checks for point distances
                            point_dists = np.sqrt(np.sum((all_points - [new_x, new_y])**2, axis=1))
                            if np.min(point_dists) < min_dist_to_points:
                                continue
                            
                            # Relaxed checks for label distances
                            if any(np.sqrt((new_x - ux)**2 + (new_y - uy)**2) < min_dist_to_labels 
                                for ux, uy in used_positions):
                                continue
                            
                            # Still maintain strict check for Human Accuracy region
                            if human_accuracy_region is not None:
                                if (abs(new_x - human_accuracy_region['x']) < human_accuracy_region['width']/2 and
                                    abs(new_y - human_accuracy_region['y']) < human_accuracy_region['height']):
                                    continue
                            
                            total_dist = np.sqrt(dx**2 + dy**2)
                            if total_dist < min_dist:
                                min_dist = total_dist
                                best_pos = (new_x, new_y)
                
                # If still no solution, use absolute fallback
                if best_pos is None:
                    print("Using fallback position...")
                    # Try corners in order: top-right, top-left, bottom-right, bottom-left
                    corners = [
                        (x_max - 0.05 * x_range, y_max - 0.05 * y_range),
                        (x_min + 0.05 * x_range, y_max - 0.05 * y_range),
                        (x_max - 0.05 * x_range, y_min + 0.05 * y_range),
                        (x_min + 0.05 * x_range, y_min + 0.05 * y_range)
                    ]
                    
                    for corner in corners:
                        if not any(np.sqrt((corner[0] - ux)**2 + (corner[1] - uy)**2) < 0.1 * min(x_range, y_range)
                                for ux, uy in used_positions):
                            best_pos = corner
                            break
                    
                    # If even corners fail, use absolute position
                    if best_pos is None:
                        best_pos = (x_max - 0.02 * x_range, y_max - 0.02 * y_range)
                
                return best_pos
            
            # Add annotations for each selected model
            for idx, row in selected_models.iterrows():
                model_name = row['model']
                x_pos = row[x_column]
                y_pos = row[y_column]
                
                # Find nearest white space
                label_pos = find_nearest_white_space(x_pos, y_pos, all_points, used_positions)
                
                if label_pos is None:
                    continue  # Skip if no suitable position found
                
                # Calculate arrow curvature based on distance
                dx = label_pos[0] - x_pos
                dy = label_pos[1] - y_pos
                distance = np.sqrt(dx**2 + dy**2)
                rad = min(0.2, distance / (x_range + y_range))  # Scale curvature with distance
                
                # Determine text alignment based on label position relative to point
                ha = 'right' if label_pos[0] < x_pos else 'left'
                va = 'top' if label_pos[1] < y_pos else 'bottom'
                
                # Create annotation with curved arrow
                ax_main.annotate(
                    model_name,
                    xy=(x_pos, y_pos),
                    xytext=label_pos,
                    xycoords='data',
                    textcoords='data',
                    ha=ha,
                    va=va,
                    fontsize=8,
                    color='#333333',
                    bbox=dict(
                        boxstyle='round,pad=0.5',  # Increased padding
                        fc='white',
                        alpha=1.0,    # Fully opaque background
                        ec='#CCCCCC',
                        zorder=100    # Ensure box is on top
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        connectionstyle=f'arc3,rad={rad}',
                        color='#333333',
                        alpha=0.7,
                        shrinkA=0,
                        shrinkB=5,
                        mutation_scale=15,
                        linewidth=2,
                        zorder=100
                    ),
                    zorder=100
                )
                
                used_positions.append(label_pos)
    
    # Legend subplot
    ax_legend = plt.subplot(gs[1])
    ax_legend.axis('off')  # Hide axes
    
    # Create legend handles
    handles = []
    for cat in categories:
        model_type, dataset_type = cat.split(' - ', 1)
        cat_data = joined_df[joined_df['category'] == cat]
        
        if len(cat_data) > 0:
            vision_or_vlm = cat_data['vision_or_VLM'].iloc[0]
            handle = plt.Line2D(
                [0], [0],
                marker=marker_styles.get(vision_or_vlm, 'o'),
                color='w',
                markerfacecolor=color_dict.get(cat, '#999999'),
                markeredgecolor='black',
                markersize=8,
                markeredgewidth=0.5,
                linestyle='None',
            )
            handles.append((cat, handle))
    
    # Create larger legend in the second subplot
    sorted_handles = sorted(handles, key=lambda x: get_sort_key(x[0]))
    legend = ax_legend.legend(
        [h for _, h in sorted_handles], 
        [cat.replace(' - ', '\n') for cat, _ in sorted_handles],
        title='Models',
        loc='center left',
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        framealpha=0.9,
        edgecolor='#DDDDDD',
        borderpad=0.5,
        handletextpad=0.5,
        markerscale=1.2
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', format='png')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    
    plt.show()


def categorize_training_data(model_name):
    """
    Categorize models based on their training dataset.
    
    Args:
        model_name (str): Model name in format <model name>.<training set>
    
    Returns:
        str: Training data category ("ImageNet-1K", "Internet-scale vision", or "Adversarial")
    """
    model_name = str(model_name).lower()
    
    # Check for Adversarial training
    if 'eps' in model_name:
        return "Adversarial"
    
    # Check for Internet-scale indicators
    internet_scale_indicators = [
        'in22k', 'openai', 'laion2b', 'in21k', 'in12k', 'm38m',
        'jft', 'mim', 'fcmae', 'dino', 'clip', 'ig1b', 'wsl', 'swsl',
        'lvd142m', 'laion', 'in21', 'in22', 'in12'
    ]
    
    for indicator in internet_scale_indicators:
        if indicator in model_name:
            return "Internet-scale vision"
    
    # If no other indicators, it's likely ImageNet-1K
    return "ImageNet-1K"


# Set the style
plt.style.use('default')
sns.set_style("ticks")

# Load the data
clickme_alignment = pd.read_csv('csvs/large_scale_results_hmn.csv')
neural_alignment = pd.read_csv('csvs/neural_alignment_red_central_it_211_250_shifted.csv')
model_analysis = pd.read_csv('csvs/model_analysis_gemini.csv')
model_metadata = pd.read_csv('csvs/model_metadata-in1k.csv')

# Change the data labels - Fix the SettingWithCopyWarning by using .loc
# Create a copy of the DataFrame first
clickme_alignment = clickme_alignment.copy()

# Use .loc for safer assignment
clickme_alignment.loc[clickme_alignment.type1 == "hybrid", "type1"] = "ConvViT"
clickme_alignment.loc[clickme_alignment.type1 == "vit", "type1"] = "ViT"
clickme_alignment.loc[clickme_alignment.type1 == "cnn", "type1"] = "CNN"

# Handle the adv mask
mask = clickme_alignment.type1 == "adv"
clickme_alignment.loc[mask, "type1"] = "CNN"
clickme_alignment = clickme_alignment.rename(columns={'type1': 'model_type_category'})
clickme_alignment.loc[mask, "type2"] = "Adversarial"

# Change the rest of type2 values
clickme_alignment.loc[clickme_alignment.type2 == "none", "type2"] = "ImageNet-1K"
clickme_alignment.loc[clickme_alignment.type2 == "l2", "type2"] = "ImageNet-1K"
clickme_alignment.loc[clickme_alignment.type2 == "linf", "type2"] = "ImageNet-1K"
clickme_alignment.loc[clickme_alignment.type2 == "dino", "type2"] = "Internet-scale vision"
clickme_alignment.loc[clickme_alignment.type2 == "big_data", "type2"] = "Internet-scale vision"
clickme_alignment.loc[clickme_alignment.type2 == "clip", "type2"] = "Internet-scale vision & language"
clickme_alignment = clickme_alignment.rename(columns={'type2': 'dataset_category'})

# Change selfsup to SSL
# clickme_alignment.loc[clickme_alignment.model_type_category == 'selfsup', 'model_type_category'] = 'SSL'
clickme_alignment.loc[clickme_alignment.model_type_category == 'selfsup', 'model_type_category'] = 'ViT'
# Join the dataframes and track models lost at each step
print(f"Initial models in clickme_alignment: {len(clickme_alignment)}")

# First merge
joined_df = clickme_alignment.merge(
    model_analysis,
    left_on='model',
    right_on='model_name',
    how='left'
)
print(f"Models after merging with model_analysis: {len(joined_df)}")
print(f"Lost {len(clickme_alignment) - len(joined_df)} models")
lost_models = set(clickme_alignment['model']) - set(joined_df['model'])
print(f"Models lost in first merge: {lost_models}")

# Second merge
pre_merge_count = len(joined_df)
joined_df = joined_df.merge(
    neural_alignment,
    left_on='model',
    right_on='model_name',
    how='inner'
)
print(f"Models after merging with neural_alignment: {len(joined_df)}")
print(f"Lost {pre_merge_count - len(joined_df)} models")
lost_models = set(clickme_alignment['model'].iloc[clickme_alignment.index.isin(range(pre_merge_count))]) - set(joined_df['model'])
print(f"Models lost in second merge: {lost_models}")

# Prepare for final merge
joined_df["full_model"] = joined_df.model
joined_df["model"] = joined_df.model.str.split('.').str[0]

# Then in your data processing section, add:
joined_df['dataset_category'] = joined_df['full_model'].apply(categorize_training_data)

# For models that use language as well (like CLIP), you can further refine:
language_model_indicators = ['clip', 'openai', 'laion2b']
language_mask = joined_df['full_model'].str.lower().apply(
    lambda x: any(indicator in x for indicator in language_model_indicators)
)
joined_df["vision_or_VLM"] = joined_df["dataset_category"].apply(lambda x: "vision" if "language" not in x else "VLM")
joined_df.loc[language_mask, 'dataset_category'] = "Internet-scale vision & language"

lost_models = set(joined_df["model"].iloc[joined_df.index.isin(range(pre_merge_count))]) - set(joined_df['model'])
if len(lost_models):
    print(f"Models lost in final merge: {lost_models}")
# Apply the categorization
# joined_df['model_type_category'] = joined_df['architecture'].apply(categorize_model_type)
# joined_df['dataset_category'] = joined_df['dataset'].apply(categorize_dataset)

# Create combined category for coloring
joined_df['category'] = joined_df['model_type_category'] + ' - ' + joined_df['dataset_category']

# Calculate normalized brain score
joined_df['normalized_brain_score'] = joined_df['brain_score'] / joined_df['ceiling_score']

# Rename columns with hyphens to use underscores
for col in joined_df.columns:
    if '-' in col:
        new_col = col.replace('-', '_')
        joined_df.rename(columns={col: new_col}, inplace=True)

# Create output directory if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Create the five scatter plots with annotations
if plot_data:
    # # First create the standard plots
    # create_plot('multi_label_acc', 
    #         title='Brain Score vs Multi-label Accuracy', 
    #         save_path=os.path.join(output_dir, 'multi_label_acc_vs_brain.png'),
    #         ylim=(0, 1))
    
    # # Then create annotated versions
    # create_plot('multi_label_acc', 
    #         title='Brain Score vs Multi-label Accuracy', 
    #         save_path=os.path.join(output_dir, 'multi_label_acc_vs_brain_annotated.png'),
    #         ylim=(0, 1),
    #         annotate_outliers=True)
    
    # # Repeat for other plots
    # create_plot('multi_label_acc', 
    #         y_column='spearman',
    #         title='ClickMe vs Multi-label Accuracy', 
    #         save_path=os.path.join(output_dir, 'multi_label_acc_vs_clickme.png'),
    #         ylim=(-0.3, 1))
    
    # create_plot('multi_label_acc', 
    #         y_column='spearman',
    #         title='ClickMe vs Multi-label Accuracy', 
    #         save_path=os.path.join(output_dir, 'multi_label_acc_vs_clickme_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)

    # create_plot('results_imagenet', 
    #         title='Brain Score vs ImageNet Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman',
    #         annotate_outliers=True)
    # create_plot('results_imagenet',
    #         y_column='spearman',
    #         title='Brain Score vs ImageNet Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)

    # create_plot('results_sketch', 
    #         title='Brain Score vs Sketch Accuracy', 
    #         save_path=os.path.join(output_dir, 'sketch_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman',
    #         annotate_outliers=True)
    # create_plot('results_sketch', 
    #         y_column='spearman',
    #         title='Brain Score vs Sketch Accuracy', 
    #         save_path=os.path.join(output_dir, 'sketch_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)

    # create_plot('results_imagenetv2_matched_frequency', 
    #         title='Brain Score vs ImageNetV2 Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenetv2_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman',
    #         annotate_outliers=True)
    # create_plot('results_imagenetv2_matched_frequency', 
    #         y_column='spearman',
    #         title='Brain Score vs ImageNetV2 Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenetv2_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)

    # create_plot('results_imagenet_r', 
    #         title='Brain Score vs ImageNet-R Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_r_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman',
    #         annotate_outliers=True)
    # create_plot('results_imagenet_r', 
    #         y_column='spearman',
    #         title='Brain Score vs ImageNet-R Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_r_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)

    create_plot('results_imagenet_a', 
            title='Brain Score vs ImageNet-A Accuracy', 
            save_path=os.path.join(output_dir, 'imagenet_a_vs_brain.png'),
            ylim=(-0.3, 1),
            y_column='spearman',
            annotate_outliers=True)
    create_plot('results_imagenet_a', 
            y_column='spearman',
            title='Brain Score vs ImageNet-A Accuracy', 
            save_path=os.path.join(output_dir, 'imagenet_a_vs_brain_annotated.png'),
            ylim=(-0.3, 1),
            annotate_outliers=True)

# Define the variables to analyze
dependent_variables = [
    'normalized_brain_score',
    'spearman'
    # 'results_imagenet',
    # 'results_sketch',
    # 'results_imagenetv2_matched_frequency',
    # 'results_imagenet_r',
    # 'results_imagenet_a'
]

# Define the categorical and continuous predictors
categorical_predictors = [
    'model_type_category',
    'dataset_category',
    'category'  # combined predictor
]

continuous_predictors = [
    'input_size',
    'multi_label_acc',
    'total_params',
    'num_transformer_layers',
    'num_normalizations',
    'num_skip_connections',
    'num_convolutional_layers',
    'num_dense_layers',
    'final_receptive_field_size',
    'num_strides',
    'num_max_pools'
]

# Create a directory for ANOVA results
anova_dir = 'anova_results'
os.makedirs(anova_dir, exist_ok=True)

# Initialize a DataFrame to store ANOVA results for reporting
all_anova_results = []  # pd.DataFrame(columns=['Dependent_Variable', 'Predictor', 'F_Value', 'p_Value', 'R_Squared'])

# Run ANOVA analyses for each dependent variable
for dv in dependent_variables:
    print(f"\nAnalyzing {dv}...")
    
    # Filter out rows with missing values for this dependent variable
    df_filtered = joined_df[~joined_df[dv].isna()].copy()
    
    if len(df_filtered) < 5:
        print(f"Not enough data for {dv}, skipping...")
        continue
    
    # Use the stored changepoint if available
    y_column = 'normalized_brain_score' if dv != 'normalized_brain_score' else 'spearman'
    key = f"{dv}_{y_column}"
    
    changepoint_x = None
    if key in changepoints:
        changepoint_x = changepoints[key]
        print(f"Using stored changepoint for {dv}: {changepoint_x}")
    else:
        # Compute the changepoint if not already stored
        x_data = df_filtered[dv].values
        y_data = df_filtered[y_column].values
        valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
        
        if sum(valid_indices) > 4:
            _, _, params = compute_piecewise_upper_bound(
                x_data[valid_indices], 
                y_data[valid_indices],
                bins=8
            )
            if params is not None:
                changepoint_x = params[0]  # x0 from the params
                print(f"Computed changepoint for {dv}: {changepoint_x}")
    
    # Create data subsets based on changepoint
    if changepoint_x is not None:
        df_pre = df_filtered[df_filtered[dv] <= changepoint_x].copy()
        df_post = df_filtered[df_filtered[dv] > changepoint_x].copy()
        print(f"Pre-changepoint: {len(df_pre)} samples, Post-changepoint: {len(df_post)} samples")
        
        # Skip if either subset is too small
        if len(df_pre) < 5 or len(df_post) < 5:
            print(f"Not enough data in one of the subsets for {dv}, skipping split analysis...")
            data_sets = [("all", df_filtered)]
        else:
            data_sets = [("all", df_filtered), ("pre", df_pre), ("post", df_post)]
    else:
        print(f"No changepoint found for {dv}, analyzing all data only")
        data_sets = [("all", df_filtered)]
    
    # Run analyses for each data subset
    for subset_name, subset_df in data_sets:
        print(f"\nAnalyzing {subset_name} data for {dv}...")
        
        # Create a working copy of the data
        analysis_df = subset_df.copy()
        
        # Z-score the continuous predictors
        for predictor in continuous_predictors:
            if predictor in analysis_df.columns and analysis_df[predictor].std() > 0:
                analysis_df[f'{predictor}_z'] = (analysis_df[predictor] - analysis_df[predictor].mean()) / analysis_df[predictor].std()
        
        # Effect-code the categorical predictors
        for predictor in categorical_predictors:
            if predictor in analysis_df.columns and len(analysis_df[predictor].unique()) >= 2:
                # Get dummies with drop_first=True to create effect coding
                dummies = pd.get_dummies(analysis_df[predictor], prefix=predictor, drop_first=True)
                
                # Convert from dummy (0/1) to effect coding (-1/1)
                for col in dummies.columns:
                    dummies[col] = dummies[col] * 2 - 1
                
                # Add to dataframe
                analysis_df = pd.concat([analysis_df, dummies], axis=1)
        
        # Run regression for all predictors using the same approach
        all_predictors = []
        
        # Add z-scored continuous predictors
        for predictor in continuous_predictors:
            z_pred = f'{predictor}_z'
            if z_pred in analysis_df.columns and not analysis_df[z_pred].isna().all():
                all_predictors.append(z_pred)
        
        # Add effect-coded categorical predictors
        for predictor in categorical_predictors:
            if predictor in analysis_df.columns:
                effect_cols = [col for col in analysis_df.columns if col.startswith(f'{predictor}_')]
                all_predictors.extend(effect_cols)
        
        # Run individual regressions for each predictor for detailed analysis
        for predictor in all_predictors:
            df_pred = analysis_df[~analysis_df[predictor].isna()].copy()
            
            if len(df_pred) < 5 or df_pred[predictor].std() == 0:
                continue
                
            try:
                # Run linear regression
                X = sm.add_constant(df_pred[predictor])
                model = sm.OLS(df_pred[dv], X).fit()
                
                # Extract results
                f_value = model.fvalue
                p_value = model.f_pvalue
                r_squared = model.rsquared
                beta = model.params[1]  # Standardized coefficient
                
                # Determine predictor type
                pred_type = 'continuous' if any(p in predictor for p in continuous_predictors) else 'categorical'
                orig_predictor = predictor.split('_')[0] if pred_type == 'categorical' else predictor.replace('_z', '')
                
                all_anova_results.append({
                    'Dependent_Variable': f"{dv}_{subset_name}",
                    'Predictor': orig_predictor,
                    'Predictor_Type': pred_type,
                    'F_Value': f_value,
                    'p_Value': p_value,
                    'R_Squared': r_squared,
                    'Beta': beta,
                    'Data_Subset': subset_name,
                    'Sample_Size': len(df_pred)
                })
                
                print(f"Regression for {predictor}: F={f_value:.2f}, p={p_value:.4f}, R²={r_squared:.4f}, β={beta:.4f}")
                
                # Create and save plots for significant relationships
                if p_value < 0.05:
                    plt.figure(figsize=(10, 6))
                    
                    if pred_type == 'continuous':
                        sns.regplot(x=predictor, y=dv, data=df_pred, scatter_kws={'alpha':0.7})
                    else:
                        # For categorical, create a special plot showing the effect
                        sns.boxplot(x=df_pred[predictor], y=df_pred[dv])
                        
                    plt.title(f'Effect of {orig_predictor} on {dv} ({subset_name})\nF={f_value:.2f}, p={p_value:.4f}, R²={r_squared:.4f}, β={beta:.4f}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(anova_dir, f'regplot_{dv}_{orig_predictor}_{subset_name}.png'), dpi=300)
                    plt.close()
            
            except Exception as e:
                print(f"Error analyzing {predictor} for {dv} ({subset_name}): {e}")

# Create separate heatmaps for each data subset
for subset_name in ["all", "pre", "post"]:
    subset_results = pd.DataFrame([r for r in all_anova_results if r['Data_Subset'] == subset_name])
    
    if len(subset_results) > 0:
        significant_results = subset_results[subset_results['p_Value'] < 0.05].copy()
        
        if len(significant_results) > 0:
            # Save the significant results to a CSV
            significant_results.to_csv(os.path.join(anova_dir, f'significant_relationships_{subset_name}.csv'), index=False)
            
            # Create a pivot table for the heatmap
            heatmap_data = significant_results.pivot_table(
                index='Predictor', 
                columns='Dependent_Variable',
                values='R_Squared',
                fill_value=0
            )
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f')
            plt.title(f'R-squared Values for Significant Predictors ({subset_name} data, p < 0.05)')
            plt.tight_layout()
            plt.savefig(os.path.join(anova_dir, f'r_squared_heatmap_{subset_name}.png'), dpi=300)
            plt.close()
            
            # Create a heatmap of F-values
            f_heatmap_data = significant_results.pivot_table(
                index='Predictor', 
                columns='Dependent_Variable',
                values='F_Value',
                fill_value=0
            )
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(f_heatmap_data, annot=True, cmap='magma', fmt='.1f', norm=LogNorm())
            plt.title(f'F-Values for Significant Predictors ({subset_name} data, p < 0.05)')
            plt.tight_layout()
            plt.savefig(os.path.join(anova_dir, f'f_value_heatmap_{subset_name}.png'), dpi=300)
            plt.close()

print(f"\nAll ANOVA analyses complete. Results saved to {anova_dir}/")
