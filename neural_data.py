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
from sklearn.linear_model import RANSACRegressor
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap


plot_data = True

# At the top of the file, add a dictionary to store changepoints
# This will be a global variable to store changepoints computed during plotting
changepoint_cache = {}

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

def compute_piecewise_upper_bound(x, y):
    """
    Compute a piecewise linear fit with a single changepoint.
    
    Args:
        x: Array of x-coordinates
        y: Array of y-coordinates
        
    Returns:
        Tuple containing:
        - pre_line: (slope, intercept) for pre-changepoint line
        - post_line: (slope, intercept) for post-changepoint line
        - params: Dictionary with additional parameters
    """
    # Detect the changepoint
    change_x, change_y = detect_changepoint(x, y)
    
    # Split data into pre and post changepoint segments
    pre_mask = x <= change_x
    post_mask = x > change_x
    
    pre_x, pre_y = x[pre_mask], y[pre_mask]
    post_x, post_y = x[post_mask], y[post_mask]
    
    # Compute hulls for each segment
    pre_hull_x, pre_hull_y = compute_pareto_front(pre_x, pre_y)
    post_hull_x, post_hull_y = compute_pareto_front(post_x, post_y)
    
    # Get first point on pre-hull
    if len(pre_hull_x) > 0:
        first_x, first_y = pre_hull_x[0], pre_hull_y[0]
    else:
        first_x, first_y = change_x, change_y
    
    # Fit pre-line: passes through first hull point and changepoint
    if len(pre_hull_x) >= 2:
        pre_slope, pre_intercept = fit_hull_segment(pre_hull_x, pre_hull_y, (change_x, change_y))
    else:
        # Direct line between first point and changepoint
        if first_x != change_x:
            pre_slope = (change_y - first_y) / (change_x - first_x)
        else:
            pre_slope = 0
        pre_intercept = change_y - pre_slope * change_x
    
    # Fit post-line: use all points on the post-hull and constrain to pass through changepoint
    if len(post_hull_x) >= 2:
        # Shift coordinates so changepoint is at origin
        shifted_x = post_hull_x - change_x
        shifted_y = post_hull_y - change_y
        
        # Fit through origin (constrained to pass through changepoint)
        if np.any(shifted_x != 0):  # Avoid division by zero
            # Use weighted fit to give more importance to points with higher y values
            weights = post_hull_y / np.max(post_hull_y) if np.max(post_hull_y) > 0 else np.ones_like(post_hull_y)
            model = sm.WLS(shifted_y, shifted_x[:, np.newaxis], weights=weights).fit()
            post_slope = model.params[0]
        else:
            post_slope = 0
        post_intercept = change_y - post_slope * change_x
        
        # Validate the fit - if it's not reasonably tracking the convex hull, 
        # default to using the last hull point
        if len(post_hull_x) > 2:
            # Check fit quality by ensuring most points are above the line (with some tolerance)
            y_pred = post_slope * post_hull_x + post_intercept
            points_above = np.sum(post_hull_y > (y_pred - 0.01))
            
            if points_above < len(post_hull_x) * 0.5:  # If fewer than half the points are above line
                # Use last hull point instead
                last_x, last_y = post_hull_x[-1], post_hull_y[-1]
                if last_x != change_x:
                    post_slope = (last_y - change_y) / (last_x - change_x)
                else:
                    post_slope = 0
                post_intercept = change_y - post_slope * change_x
    else:
        # Fallback if not enough points
        post_slope = 0
        post_intercept = change_y
    
    # Build parameters dictionary
    params = {
        'changepoint': (change_x, change_y),
        'first_point': (first_x, first_y),
        'last_point': (last_x, last_y)
    }
    
    return (pre_slope, pre_intercept), (post_slope, post_intercept), params

# Helper function to compute Pareto front - ensures full range coverage
def compute_pareto_front(x, y):
    """
    Compute the Pareto front (convex hull of max y values) for a set of points.
    
    Args:
        x: Array of x-coordinates
        y: Array of y-coordinates
        
    Returns:
        Tuple of arrays (hull_x, hull_y) representing the Pareto front points
    """
    # Sort points by x-coordinate
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    # Compute the Pareto front (max y value for each x)
    hull_indices = []
    current_max_y = float('-inf')
    
    for i in range(len(x_sorted)):
        if y_sorted[i] > current_max_y:
            hull_indices.append(i)
            current_max_y = y_sorted[i]
    
    # Return the hull points
    hull_x = x_sorted[hull_indices]
    hull_y = y_sorted[hull_indices]
    
    return hull_x, hull_y

def detect_changepoint(x, y):
    """
    Detect the changepoint where y starts to decrease after increasing.
    
    Args:
        x: Array of x-coordinates
        y: Array of y-coordinates
        
    Returns:
        Tuple (change_x, change_y) representing the changepoint coordinates
    """
    # Compute the Pareto front
    hull_x, hull_y = compute_pareto_front(x, y)
    
    if len(hull_x) < 3:
        # Not enough points for changepoint detection
        # Return the point with maximum y
        max_idx = np.argmax(hull_y)
        return hull_x[max_idx], hull_y[max_idx]
    
    # Calculate slopes between consecutive hull points
    slopes = np.diff(hull_y) / np.diff(hull_x)
    
    # Find where slope changes from positive to negative
    change_indices = []
    for i in range(1, len(slopes)):
        if slopes[i-1] > 0 and slopes[i] < 0:
            change_indices.append(i)
    
    if change_indices:
        # Find the largest drop in slope
        max_drop_idx = change_indices[0]
        max_drop = slopes[change_indices[0]-1] - slopes[change_indices[0]]
        
        for idx in change_indices[1:]:
            drop = slopes[idx-1] - slopes[idx]
            if drop > max_drop:
                max_drop = drop
                max_drop_idx = idx
        
        # The changepoint is at the end of the positive slope
        change_idx = max_drop_idx
        return hull_x[change_idx], hull_y[change_idx]
    else:
        # If no clear positive-to-negative transition, use the peak
        max_idx = np.argmax(hull_y)
        return hull_x[max_idx], hull_y[max_idx]

def fit_hull_segment(hull_x, hull_y, fixed_point=None):
    """
    Fit a line to hull points, ensuring it passes through a fixed point if specified.
    
    Args:
        hull_x: x-coordinates of hull points
        hull_y: y-coordinates of hull points
        fixed_point: Optional tuple (x, y) that the line must pass through
        
    Returns:
        Tuple (slope, intercept) for the fitted line
    """
    if len(hull_x) < 2:
        return 0, 0
    
    if fixed_point is None:
        # Standard linear regression
        X = sm.add_constant(hull_x)
        model = sm.OLS(hull_y, X).fit()
        slope = model.params[1]
        intercept = model.params[0]
    else:
        # Constrained fit passing through the fixed point
        if len(hull_x) == 2:
            # With only two points, just connect them
            fx, fy = fixed_point
            other_x = hull_x[0] if hull_x[1] == fx else hull_x[1]
            other_y = hull_y[0] if hull_y[1] == fy else hull_y[1]
            
            if other_x == fx:  # Avoid division by zero
                slope = 0
            else:
                slope = (other_y - fy) / (other_x - fx)
                
            intercept = fy - slope * fx
        else:
            # With more points, find the best fit that passes through fixed_point
            fx, fy = fixed_point
            
            # Shift coordinates so fixed point is at origin
            shifted_x = hull_x - fx
            shifted_y = hull_y - fy
            
            # Fit through origin (no intercept)
            model = sm.OLS(shifted_y, shifted_x[:, np.newaxis]).fit()
            slope = model.params[0]
            
            # Calculate intercept to pass through fixed point
            intercept = fy - slope * fx
    
    return slope, intercept

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
    # joined_df['model_type_category'] = joined_df['model_type_category'].replace('selfsup', 'SSL')
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
    model_type_order = ['CNN', 'ConvViT', 'ViT']
    dataset_type_order = ['ImageNet-1K', 'Internet-scale vision', 'Internet-scale vision & language', 'Adversarially trained']
    
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
                s=25,
                label=cat,
                edgecolor='black',  
                linewidth=0.3,
                zorder=5  # Set zorder for scatter points
            )
    
    # Calculate piecewise linear upper bound
    x_data = joined_df[x_column].values
    y_data = joined_df[y_column].values
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    
    # Get current axis limits before any additional plotting
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    
    if sum(valid_indices) > 4:  # Need enough points for a meaningful fit
        # Compute piecewise fit
        (pre_slope, pre_intercept), (post_slope, post_intercept), fit_params = compute_piecewise_upper_bound(
            x_data[valid_indices], 
            y_data[valid_indices]
        )
        
        # Extract parameters
        change_x, change_y = fit_params['changepoint']
        
        # Store the changepoint in the global cache with a key based on the columns
        # This ensures we can retrieve it later for the ANOVA
        global changepoint_cache
        cache_key = f"{x_column}_{y_column}"
        changepoint_cache[cache_key] = change_x
        
        # Plot pre-changepoint line - extend all the way across the plot
        x_pre = np.array([x_min, x_max])
        y_pre = pre_slope * x_pre + pre_intercept
        plt.plot(x_pre, y_pre, color='black', alpha=0.4, linestyle='--', linewidth=1.0)
        
        # Plot post-changepoint line - extend all the way across the plot
        x_post = np.array([x_min, x_max])
        y_post = post_slope * x_post + post_intercept
        plt.plot(x_post, y_post, color='black', alpha=0.4, linestyle='--', linewidth=1.0)
        
        # Format equations
        eq1 = f"y = {pre_slope:.3f}x + {pre_intercept:.3f}"
        eq2 = f"y = {post_slope:.3f}x + {post_intercept:.3f}"
        
        # Place equations in a text box - using absolute y=0.85
        plt.text(x_min + 0.05*(x_max - x_min),  # Fixed left position
                0.85,                            # Fixed absolute y position
                f"Pre-threshold: {eq1}\nPost-threshold: {eq2}", 
                fontsize=8,
                ha='left',
                va='top',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='#CCCCCC', boxstyle='round,pad=0.5'),
                zorder=10)
        
        # Highlight the model at the changepoint and the model with the highest x-value post-changepoint
        from matplotlib.patches import ConnectionPatch
        
        valid_x = x_data[valid_indices]
        valid_y = y_data[valid_indices]
        valid_models = joined_df.iloc[valid_indices].reset_index(drop=True)
        
        # Find model closest to changepoint
        distances_to_changepoint = np.sqrt((valid_x - change_x)**2 + (valid_y - change_y)**2)
        changepoint_model_idx = np.argmin(distances_to_changepoint)
        
        # Find model with the highest x-value post-changepoint
        post_mask = valid_x > change_x
        post_x, post_y = valid_x[post_mask], valid_y[post_mask]
        
        if len(post_x) > 0:
            far_right_idx = np.argmax(post_x)
            far_right_x, far_right_y = post_x[far_right_idx], post_y[far_right_idx]
            far_right_model_idx = post_mask.nonzero()[0][far_right_idx]
        else:
            far_right_x, far_right_y = change_x, change_y
            far_right_model_idx = changepoint_model_idx
        
        # Get model names
        possible_name_cols = ['model_name', 'name', 'meta_name', 'model', 'model_id']
        model_name_col = None
        
        for col in possible_name_cols:
            if col in valid_models.columns:
                model_name_col = col
                break
                
        if model_name_col is None:
            changepoint_model_name = "Changepoint Model"
            far_right_model_name = "Far Right Model"
        else:
            def get_short_name(full_name):
                if '/' in str(full_name):
                    parts = str(full_name).split('/')
                    return parts[-1]
                else:
                    name = str(full_name)
                    if len(name) > 20:
                        return name[:17] + "..."
                    return name
            
            changepoint_model_name = get_short_name(valid_models.iloc[changepoint_model_idx][model_name_col])
            far_right_model_name = get_short_name(valid_models.iloc[far_right_model_idx][model_name_col])
        
        # Helper function to adjust arrow endpoint to be very close to the dot's center
        def adjust_endpoint(start_x, start_y, end_x, end_y, marker_radius=0.005):
            dx, dy = end_x - start_x, end_y - start_y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < marker_radius:
                return end_x, end_y
                
            dx, dy = dx/distance, dy/distance
            adjusted_end_x = end_x - dx * marker_radius
            adjusted_end_y = end_y - dy * marker_radius
            
            return adjusted_end_x, adjusted_end_y
        
        # Calculate appropriate marker distance
        x_range = x_max - x_min
        y_range = y_max - y_min
        plot_scale = np.sqrt(x_range**2 + y_range**2)
        marker_distance = plot_scale * 0.003
        
        # Add annotation for changepoint model
        cp_x, cp_y = valid_x[changepoint_model_idx], valid_y[changepoint_model_idx]
        cp_text_x, cp_text_y = cp_x - 0.05 * (x_max - x_min), cp_y + 0.05 * (y_max - y_min)
        adjusted_cp_x, adjusted_cp_y = adjust_endpoint(cp_text_x, cp_text_y, cp_x, cp_y, marker_distance)
        
        con = ConnectionPatch(
            xyA=(cp_text_x, cp_text_y),
            xyB=(adjusted_cp_x, adjusted_cp_y),
            coordsA="data", coordsB="data",
            arrowstyle="-|>", 
            color="black",
            connectionstyle=f"arc3,rad={0.2}",
            linewidth=1,
            zorder=11
        )
        plt.gca().add_artist(con)
        
        plt.text(cp_text_x, cp_text_y, changepoint_model_name, fontsize=8,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'),
                zorder=12)
        
        # Add annotation for far right model
        far_right_text_x, far_right_text_y = far_right_x - 0.05 * (x_max - x_min), far_right_y + 0.05 * (y_max - y_min)
        adjusted_far_right_x, adjusted_far_right_y = adjust_endpoint(far_right_text_x, far_right_text_y, far_right_x, far_right_y, marker_distance)
        
        con = ConnectionPatch(
            xyA=(far_right_text_x, far_right_text_y),
            xyB=(adjusted_far_right_x, adjusted_far_right_y),
            coordsA="data", coordsB="data",
            arrowstyle="-|>", 
            color="black",
            connectionstyle=f"arc3,rad={0.2}",
            linewidth=1,
            zorder=11
        )
        plt.gca().add_artist(con)
        
        plt.text(far_right_text_x, far_right_text_y, far_right_model_name, fontsize=8,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'),
                zorder=12)
    
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
joined_df["vision_or_VLM"] = joined_df["dataset_category"].apply(lambda x: "vision" if "language" not in x else "VLM")
)
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

# # Prepare for statistics
# Create a number_of_layers column
joined_df['num_layers'] = joined_df['num_transformer_layers'] + joined_df['num_convolutional_layers'] + joined_df['num_dense_layers']

# Create binary indicators for transformer and CNN
joined_df['is_transformer'] = joined_df['model_type_category'].apply(lambda x: 1 if x in ["ViT", "ConvViT"] else 0)
joined_df['is_cnn'] = joined_df['model_type_category'].apply(lambda x: 1 if x in ["CNN", "ConvViT"] else 0)

# Create binary indicators for datasets
joined_df['is_imagenet'] = joined_df['dataset_category'].apply(lambda x: 1 if x == 'ImageNet-1K' else 0)
joined_df['is_internet_scale'] = joined_df['dataset_category'].apply(lambda x: 1 if x == 'Internet-scale vision' else 0)
joined_df['is_adversarial'] = joined_df['dataset_category'].apply(lambda x: 1 if x == 'Adversarial' else 0)
joined_df['is_language'] = joined_df['dataset_category'].apply(lambda x: 1 if x == 'Internet-scale vision & language' else 0)

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
    
    # Then create annotated versions
    create_plot('multi_label_acc', 
            title='Brain Score vs Multi-label Accuracy', 
            save_path=os.path.join(output_dir, 'multi_label_acc_vs_brain_annotated.png'),
            ylim=(0, 1),
            annotate_outliers=True)
    
    # # Repeat for other plots
    # create_plot('multi_label_acc', 
    #         y_column='spearman',
    #         title='ClickMe vs Multi-label Accuracy', 
    #         save_path=os.path.join(output_dir, 'multi_label_acc_vs_clickme.png'),
    #         ylim=(-0.3, 1))
    
    create_plot('multi_label_acc', 
            y_column='spearman',
            title='ClickMe vs Multi-label Accuracy', 
            save_path=os.path.join(output_dir, 'multi_label_acc_vs_clickme_annotated.png'),
            ylim=(-0.3, 1),
            annotate_outliers=True)

    # create_plot('results_imagenet', 
    #         title='Brain Score vs ImageNet Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman')
    create_plot('results_imagenet',
            y_column='spearman',
            title='Brain Score vs ImageNet Accuracy', 
            save_path=os.path.join(output_dir, 'imagenet_vs_brain_annotated.png'),
            ylim=(-0.3, 1),
            annotate_outliers=True)

    # create_plot('results_sketch', 
    #         title='Brain Score vs Sketch Accuracy', 
    #         save_path=os.path.join(output_dir, 'sketch_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman')
    create_plot('results_sketch', 
            y_column='spearman',
            title='Brain Score vs Sketch Accuracy', 
            save_path=os.path.join(output_dir, 'sketch_vs_brain_annotated.png'),
            ylim=(-0.3, 1),
            annotate_outliers=True)

    # create_plot('results_imagenetv2_matched_frequency', 
    #         title='Brain Score vs ImageNetV2 Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenetv2_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman')
    create_plot('results_imagenetv2_matched_frequency', 
            y_column='spearman',
            title='Brain Score vs ImageNetV2 Accuracy', 
            save_path=os.path.join(output_dir, 'imagenetv2_vs_brain_annotated.png'),
            ylim=(-0.3, 1),
            annotate_outliers=True)

    # create_plot('results_imagenet_r', 
    #         title='Brain Score vs ImageNet-R Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_r_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman')
    create_plot('results_imagenet_r', 
            y_column='spearman',
            title='Brain Score vs ImageNet-R Accuracy', 
            save_path=os.path.join(output_dir, 'imagenet_r_vs_brain_annotated.png'),
            ylim=(-0.3, 1),
            annotate_outliers=True)

    # create_plot('results_imagenet_a', 
    #         title='Brain Score vs ImageNet-A Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_a_vs_brain.png'),
    #         ylim=(-0.3, 1),
    #         y_column='spearman')
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
    # 'model_type_category',
    # 'dataset_category',
    # 'category'  # combined predictor — interaction term between model_type_category and dataset_category
]

continuous_predictors = [
    'is_transformer',
    'is_cnn',
    'is_imagenet',
    'is_internet_scale',
    'is_adversarial',
    'is_language',
    'input_size',
    'multi_label_acc',
    'total_params',
    # 'num_normalizations',
    # 'num_skip_connections',
    'num_layers',
    # 'final_receptive_field_size',
    # 'num_strides',
    # 'num_max_pools'
]

# Create a directory for ANOVA results
anova_dir = 'anova_results'
os.makedirs(anova_dir, exist_ok=True)

# Initialize a DataFrame to store ANOVA results for reporting
all_anova_results = []  # pd.DataFrame(columns=['Dependent_Variable', 'Predictor', 'F_Value', 'p_Value', 'R_Squared'])

# Create a function to run the ANOVAs and generate the heatmap
def run_anova_with_heatmap(joined_df, independent_variables, dependent_variables, anova_dir):
    # Dictionary to store results for heatmap
    heatmap_results = {}
    
    # For each dependent variable
    for y_column in dependent_variables:
        # Create a nested dictionary for this dependent variable
        heatmap_results[y_column] = {}
        
        # For each independent variable
        for x_column in independent_variables:
            print(f"\nAnalyzing {y_column} vs {x_column}")
            
            # Create a filtered dataframe with non-null values for this pair
            analysis_df = joined_df.dropna(subset=[x_column, y_column])
            
            # Get cached changepoint or compute it
            cache_key = f"{x_column}_{y_column}"
            
            if cache_key in changepoint_cache:
                changepoint_x = changepoint_cache[cache_key]
                print(f"Using cached changepoint at x = {changepoint_x:.4f}")
            else:
                # Compute changepoint if not cached
                valid_x = analysis_df[x_column].values
                valid_y = analysis_df[y_column].values
                
                if len(valid_x) > 4:
                    (_, _), (_, _), fit_params = compute_piecewise_upper_bound(valid_x, valid_y)
                    changepoint_x = fit_params['changepoint'][0]
                    print(f"Computed changepoint at x = {changepoint_x:.4f}")
                else:
                    print("Not enough data for changepoint detection. Using median x-value.")
                    changepoint_x = np.median(valid_x)
            
            # Split data into pre and post changepoint
            pre_df = analysis_df[analysis_df[x_column] <= changepoint_x]
            post_df = analysis_df[analysis_df[x_column] > changepoint_x]
            
            print(f"Pre-changepoint: {len(pre_df)} samples")
            print(f"Post-changepoint: {len(post_df)} samples")
            
            # Run ANOVA for pre-changepoint
            pre_results = {}
            if len(pre_df) > 10:
                # Run your existing ANOVA analysis on pre_df
                # Example using statsmodels:
                import statsmodels.formula.api as smf
                
                # Adjust the formula based on your actual predictor variables
                formula = f"{y_column} ~ {x_column}"
                pre_model = smf.ols(formula, data=pre_df).fit()
                
                # Extract coefficients and p-values
                pre_coef = pre_model.params[x_column]
                pre_pval = pre_model.pvalues[x_column]
                
                pre_results = {
                    'coef': pre_coef,
                    'pval': pre_pval,
                    'significant': pre_pval < 0.05
                }
            else:
                pre_results = {
                    'coef': np.nan,
                    'pval': np.nan,
                    'significant': False
                }
            
            # Run ANOVA for post-changepoint
            post_results = {}
            if len(post_df) > 10:
                # Run your existing ANOVA analysis on post_df
                # Example using statsmodels:
                formula = f"{y_column} ~ {x_column}"
                post_model = smf.ols(formula, data=post_df).fit()
                
                # Extract coefficients and p-values
                post_coef = post_model.params[x_column]
                post_pval = post_model.pvalues[x_column]
                
                post_results = {
                    'coef': post_coef,
                    'pval': post_pval,
                    'significant': post_pval < 0.05
                }
            else:
                post_results = {
                    'coef': np.nan,
                    'pval': np.nan,
                    'significant': False
                }
            
            # Store results for heatmap
            heatmap_results[y_column][x_column] = {
                'pre': pre_results,
                'post': post_results
            }
    
    # Create heatmaps for each dependent variable
    for y_column in dependent_variables:
        plot_coefficient_heatmap(heatmap_results[y_column], y_column, anova_dir)

def plot_coefficient_heatmap(results_dict, dependent_var, save_dir):
    """
    Create a heatmap showing pre and post coefficients side by side.
    
    Args:
        results_dict: Dictionary containing pre/post results for each predictor
        dependent_var: Name of the dependent variable
        save_dir: Directory to save the plots
    """
    # Create DataFrames for coefficients and p-values
    predictors = list(results_dict.keys())
    
    # Initialize empty arrays
    pre_coefs = []
    pre_pvals = []
    post_coefs = []
    post_pvals = []
    
    # Fill arrays with data
    for predictor in predictors:
        pre_coefs.append(results_dict[predictor]['pre']['coef'])
        pre_pvals.append(results_dict[predictor]['pre']['pval'])
        post_coefs.append(results_dict[predictor]['post']['coef'])
        post_pvals.append(results_dict[predictor]['post']['pval'])
    
    # Create dataframes
    coef_df = pd.DataFrame({
        'Pre-Threshold': pre_coefs,
        'Post-Threshold': post_coefs
    }, index=predictors)
    
    pval_df = pd.DataFrame({
        'Pre-Threshold': pre_pvals,
        'Post-Threshold': post_pvals
    }, index=predictors)
    
    # Create a custom color map - blue for negative, white for zero, red for positive
    colors = ["blue", "white", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
    
    # Create the heatmap
    plt.figure(figsize=(10, max(6, len(predictors) * 0.4)))
    
    # Calculate the absolute max for symmetric color scaling
    abs_max = max(abs(coef_df.min().min()), abs(coef_df.max().max()))
    if abs_max == 0 or np.isnan(abs_max):
        abs_max = 1.0  # Default if all values are 0 or NaN
    
    # Plot the heatmap
    ax = sns.heatmap(coef_df, annot=True, cmap=custom_cmap, center=0, 
                    vmin=-abs_max, vmax=abs_max, 
                    linewidths=0.5, cbar_kws={"label": "Coefficient Value"})
    
    # Add significance stars
    for i, predictor in enumerate(predictors):
        # Pre-threshold stars
        if not np.isnan(pre_pvals[i]):
            if pre_pvals[i] < 0.001:
                ax.text(0.5, i + 0.5, "***", ha='center', va='center', color='black', fontweight='bold')
            elif pre_pvals[i] < 0.01:
                ax.text(0.5, i + 0.5, "**", ha='center', va='center', color='black', fontweight='bold')
            elif pre_pvals[i] < 0.05:
                ax.text(0.5, i + 0.5, "*", ha='center', va='center', color='black', fontweight='bold')
        
        # Post-threshold stars
        if not np.isnan(post_pvals[i]):
            if post_pvals[i] < 0.001:
                ax.text(1.5, i + 0.5, "***", ha='center', va='center', color='black', fontweight='bold')
            elif post_pvals[i] < 0.01:
                ax.text(1.5, i + 0.5, "**", ha='center', va='center', color='black', fontweight='bold')
            elif post_pvals[i] < 0.05:
                ax.text(1.5, i + 0.5, "*", ha='center', va='center', color='black', fontweight='bold')
    
    # Set title and labels
    plt.title(f"Pre vs Post Threshold Coefficients for {dependent_var}")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    
    # Add a legend for stars
    plt.figtext(0.01, 0.01, "*p<0.05, **p<0.01, ***p<0.001", ha="left", fontsize=9)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'heatmap_{dependent_var}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Replace the existing ANOVA section with a call to this new function
# ...
# Where your existing ANOVA analysis is, replace with:
run_anova_with_heatmap(joined_df, independent_variables, dependent_variables, anova_dir)
