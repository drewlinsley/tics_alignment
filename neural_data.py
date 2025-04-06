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
from matplotlib_venn import venn3, venn3_circles
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


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
def create_plot(x_column, y_column='normalized_brain_score', title=None, save_path=None, ylim=None, xlim=None, annotate_outliers=False, plot_lines=True):
    # Set publication-quality style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Raleway'],
        'font.size': 14,  # Increased from 12
        'axes.linewidth': 1.0,  # Increased from 0.8
        'axes.labelsize': 16,  # Increased from 12
        'axes.titlesize': 18,  # Increased from 12
        'xtick.labelsize': 14,  # Increased from 12
        'ytick.labelsize': 14,  # Increased from 12
        'legend.fontsize': 12,  # Increased from 8
        'lines.linewidth': 2.0,  # Increased from 1.5
        'patch.linewidth': 1.0,  # Increased from 0.8
        'savefig.dpi': 600,
        'savefig.format': 'pdf'
    })
    
    # Create figure with two subplots - main plot and legend
    fig = plt.figure(figsize=(5.5 * 2, 3.0 * 2))
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
            fontsize=12,  # Increased from 8
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
                s=80,  # Increased from 25
                label=cat,
                edgecolor='black',  
                linewidth=0.5,
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
        if plot_lines:
            plt.plot(x_pre, y_pre, color='black', alpha=0.6, linestyle='--', linewidth=2.0)  # Increased from 1.5
        
        # Plot post-changepoint line - extend all the way across the plot
        x_post = np.array([x_min, x_max])
        y_post = post_slope * x_post + post_intercept
        if plot_lines:
            plt.plot(x_post, y_post, color='black', alpha=0.6, linestyle='--', linewidth=2.0)  # Increased from 1.5
        
        # Format equations
        eq1 = f"y = {pre_slope:.3f}x + {pre_intercept:.3f}"
        eq2 = f"y = {post_slope:.3f}x + {post_intercept:.3f}"
        
        if plot_lines:
            # Place equations in a text box - using absolute y=0.85
            plt.text(x_min + 0.05*(x_max - x_min),  # Fixed left position
                    0.85,                            # Fixed absolute y position
                    f"Pre-changepoint: {eq1}\nPost-changepoint: {eq2}", 
                    fontsize=12,  # Increased from 8
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
        def adjust_endpoint(start_x, start_y, end_x, end_y, marker_radius=0.008):  # Increased from 0.005
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
        marker_distance = plot_scale * 0.005  # Increased from 0.003
        
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
            linewidth=1.5,  # Increased from 1.0
            zorder=11
        )
        plt.gca().add_artist(con)
        
        plt.text(cp_text_x, cp_text_y, changepoint_model_name, fontsize=12,  # Increased from 8
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
            linewidth=1.5,  # Increased from 1.0
            zorder=11
        )
        plt.gca().add_artist(con)
        
        plt.text(far_right_text_x, far_right_text_y, far_right_model_name, fontsize=12,  # Increased from 8
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'),
                zorder=12)
    
    # Set labels and style for main plot
    if title and ' vs ' in title:
        y_label, x_label = title.split(' vs ')
    else:
        x_label = x_column
        # Updated y-axis labels as requested
        if y_column == 'normalized_brain_score':
            y_label = 'Alignment with IT activity'
        elif y_column == 'spearman':
            y_label = 'Alignment with human feature importance'
        else:
            y_label = y_column
    
    ax_main.set_xlabel(x_label, fontsize=16, labelpad=8)  # Increased from 10, 5
    ax_main.set_ylabel(y_label, fontsize=16, labelpad=8)  # Increased from 10, 5
    
    # Remove title as requested
    # if title:
    #     ax_main.set_title(title, fontsize=18, pad=10)  # Increased from 11, 8
    
    # Clean up main plot
    sns.despine(ax=ax_main)
    ax_main.grid(False)
    
    if ylim is not None:
        ax_main.set_ylim(ylim)
    if xlim is not None:
        ax_main.set_xlim(xlim)
    
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
                markersize=10,  # Increased from 8
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
        fontsize=12,  # Increased from 10
        title_fontsize=14,  # Increased from 12
        frameon=True,
        framealpha=0.9,
        edgecolor='#DDDDDD',
        borderpad=0.5,
        handletextpad=0.5,
        markerscale=1.5  # Increased from 1.2
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
neural_alignment = pd.read_csv('csvs/neural_alignment_red_central_it_shifted_211_250.csv')
# neural_alignment = pd.read_csv('csvs/brainscore_MajajHong2015public_IT.csv')
model_analysis = pd.read_csv('csvs/model_analysis_gemini.csv')
model_metadata = pd.read_csv('csvs/model_metadata-in1k.csv')

# Change the data labels - Fix the SettingWithCopyWarning by using .loc
if "new_brain_score" not in neural_alignment.columns:
    try:
        neural_alignment['new_brain_score'] = neural_alignment['brain_score']
        neural_alignment['new_brain_score_std'] = neural_alignment['brain_score_std']    
    except:
        neural_alignment['new_brain_score'] = neural_alignment['brainscore']
        neural_alignment['new_brain_score_std'] = neural_alignment['brain_score_std']    

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
joined_df['normalized_brain_score'] = joined_df['new_brain_score']  #  / joined_df['ceiling_score']

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
    create_plot('multi_label_acc', 
            y_column='spearman',
            title='ClickMe vs Multi-label Accuracy', 
            save_path=os.path.join(output_dir, 'multi_label_acc_vs_clickme_annotated.png'),
            ylim=(-0.3, 1),
            annotate_outliers=True)
    create_plot('spearman', 
            y_column='normalized_brain_score',
            title='Spearman vs Normalized Brain Score', 
            save_path=os.path.join(output_dir, 'spearman_vs_normalized_brain_score_annotated.png'),
            ylim=(0.3, 1.05),
            xlim=(-0.3, 1),
            plot_lines=False,
            annotate_outliers=True)
    # # # 
    # create_plot('results_imagenet',
    #         y_column='spearman',
    #         title='Brain Score vs ImageNet Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)
    # create_plot('results_sketch', 
    #         y_column='spearman',
    #         title='Brain Score vs Sketch Accuracy', 
    #         save_path=os.path.join(output_dir, 'sketch_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)
    # create_plot('results_imagenetv2_matched_frequency', 
    #         y_column='spearman',
    #         title='Brain Score vs ImageNetV2 Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenetv2_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)
    # create_plot('results_imagenet_r', 
    #         y_column='spearman',
    #         title='Brain Score vs ImageNet-R Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_r_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)
    # create_plot('results_imagenet_a', 
    #         y_column='spearman',
    #         title='Brain Score vs ImageNet-A Accuracy', 
    #         save_path=os.path.join(output_dir, 'imagenet_a_vs_brain_annotated.png'),
    #         ylim=(-0.3, 1),
    #         annotate_outliers=True)

# Define the variables to analyze
from itertools import product

dependent_variables = [
    'normalized_brain_score',
    'spearman'
    # 'results_imagenet',
    # 'results_sketch',
    # 'results_imagenetv2_matched_frequency',
    # 'results_imagenet_r',
    # 'results_imagenet_a'
]
independent_variables = [
    'multi_label_acc',
]

# Create all combinations of dependent and independent variables
variable_combinations = list(product(dependent_variables, independent_variables))

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
    # 'multi_label_acc',
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

# Run ANOVA analyses for each dependent variable
for dv, iv in variable_combinations:
    print(f"\nAnalyzing {dv}...")
    
    # Filter out rows with missing values for this dependent variable
    df_filtered = joined_df[~joined_df[dv].isna()].copy()
    
    if len(df_filtered) < 5:
        print(f"Not enough data for {dv}, skipping...")
        continue
    
    # Use the stored changepoint if available
    y_column = 'normalized_brain_score' if dv != 'normalized_brain_score' else 'spearman'
    key = f"{iv}_{y_column}"
    
    changepoint_x = None
    if key in changepoint_cache:
        changepoint_x = changepoint_cache[key]
        print(f"Using cached changepoint for {dv}: {changepoint_x}")
    else:
        # Compute the changepoint if not already stored
        x_data = df_filtered[dv].values
        y_data = df_filtered[y_column].values
        valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
        
        if sum(valid_indices) > 4:
            _, _, params = compute_piecewise_upper_bound(
                x_data[valid_indices], 
                y_data[valid_indices]
            )
            if params is not None:
                changepoint_x = params['changepoint'][0]  # x0 from the params
                print(f"Computed changepoint for {dv}: {changepoint_x}")
    
    # Create data subsets based on changepoint
    if changepoint_x is not None:
        df_pre = df_filtered[df_filtered[iv] <= changepoint_x].copy()
        df_post = df_filtered[df_filtered[iv] > changepoint_x].copy()
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
        
        # Run a single regression with all predictors
        if len(all_predictors) > 0:
            try:
                # Remove rows with any NaN values in predictors or DV
                mask = ~analysis_df[all_predictors + [dv]].isna().any(axis=1)
                df_pred = analysis_df[mask].copy()
                
                if len(df_pred) < 5:
                    print(f"Not enough data points for {dv}_{subset_name}")
                    continue
                    
                # Check for zero variance predictors
                valid_predictors = []
                for pred in all_predictors:
                    if df_pred[pred].std() > 0:
                        valid_predictors.append(pred)
                    else:
                        print(f"Dropping {pred} due to zero variance")
                
                if len(valid_predictors) == 0:
                    print(f"No valid predictors for {dv}_{subset_name}")
                    continue
                
                # Run multiple regression
                X = sm.add_constant(df_pred[valid_predictors])
                model = sm.OLS(df_pred[dv], X).fit()
                
                # Extract overall model results
                f_value = model.fvalue
                p_value = model.f_pvalue
                r_squared = model.rsquared
                
                print(f"\nOverall model for {dv}_{subset_name}:")
                print(f"F={f_value:.2f}, p={p_value:.4f}, R²={r_squared:.4f}")
                print("\nCoefficients:")
                print(model.summary().tables[1])
                
                # Store results for each predictor
                for predictor in valid_predictors:
                    # Determine predictor type
                    pred_type = 'continuous' if any(p in predictor for p in continuous_predictors) else 'categorical'
                    orig_predictor = predictor.split('_')[0] if pred_type == 'categorical' else predictor.replace('_z', '')
                    
                    beta = model.params[predictor]
                    p_val = model.pvalues[predictor]
                    std_err = model.bse[predictor]  # Standard error for confidence intervals
                    
                    all_anova_results.append({
                        'Dependent_Variable': f"{dv}_{subset_name}",
                        'Predictor': orig_predictor,
                        'Predictor_Type': pred_type,
                        'F_Value': f_value,  # Overall model F
                        'p_Value': p_val,    # Individual predictor p-value
                        'R_Squared': r_squared,  # Overall model R²
                        'Beta': beta,
                        'Std_Error': std_err,  # Add standard error for confidence intervals
                        'Data_Subset': subset_name,
                        'Sample_Size': len(df_pred)
                    })
            except Exception as e:
                print(f"Error analyzing {dv} ({subset_name}): {e}")


def plot_pre_post_comparison(variable_combinations, all_anova_results, save_dir='anova_results'):
    """
    Create publication-ready plots comparing parameter estimates between pre and post changepoint data.
    
    Args:
        variable_combinations: List of (dependent_variable, independent_variable) tuples
        all_anova_results: List of dictionaries with ANOVA results
        save_dir: Directory to save the plots
    """
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Raleway'],
        'font.size': 12,
        'axes.linewidth': 0.8,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.8,
        'savefig.dpi': 600,
        'savefig.format': 'pdf'
    })
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier manipulation
    results_df = pd.DataFrame(all_anova_results)
    
    # First pass: Calculate global min and max values across all plots
    global_min = float('inf')
    global_max = float('-inf')
    
    # Collect all beta values and their confidence intervals
    all_values = []
    
    for dv, iv in variable_combinations:
        # Filter results for this DV
        dv_results = results_df[results_df['Dependent_Variable'].str.startswith(f"{dv}_")]
        
        if len(dv_results) == 0:
            continue
        
        # Get unique predictors for this DV
        predictors = dv_results['Predictor'].unique()
        
        # Calculate min and max values including confidence intervals
        for predictor in predictors:
            pre_data = dv_results[(dv_results['Predictor'] == predictor) & 
                                 (dv_results['Data_Subset'] == 'pre')]
            post_data = dv_results[(dv_results['Predictor'] == predictor) & 
                                  (dv_results['Data_Subset'] == 'post')]
            
            if len(pre_data) > 0:
                pre_beta = pre_data['Beta'].values[0]
                
                # Use standard errors directly if available
                if 'Std_Error' in pre_data.columns and not np.isnan(pre_data['Std_Error'].values[0]):
                    pre_se = pre_data['Std_Error'].values[0]
                else:
                    # Calculate standard errors from p-values
                    pre_p = pre_data['p_Value'].values[0]
                    pre_t = max(abs(stats.norm.ppf(pre_p / 2)), 0.001)
                    pre_se = abs(pre_beta) / pre_t
                
                pre_ci = 1.96 * pre_se
                all_values.append(pre_beta - pre_ci)
                all_values.append(pre_beta + pre_ci)
            
            if len(post_data) > 0:
                post_beta = post_data['Beta'].values[0]
                
                # Use standard errors directly if available
                if 'Std_Error' in post_data.columns and not np.isnan(post_data['Std_Error'].values[0]):
                    post_se = post_data['Std_Error'].values[0]
                else:
                    # Calculate standard errors from p-values
                    post_p = post_data['p_Value'].values[0]
                    post_t = max(abs(stats.norm.ppf(post_p / 2)), 0.001)
                    post_se = abs(post_beta) / post_t
                
                post_ci = 1.96 * post_se
                all_values.append(post_beta - post_ci)
                all_values.append(post_beta + post_ci)
    
    # Calculate global min and max from all collected values
    if all_values:
        # Instead of using the actual minimum, set a fixed minimum around -0.1
        global_min = -0.2
        global_max = max(all_values)
        
        # Ensure zero is included in the range
        if global_max < 0:
            global_max = 0.1  # If all values are negative, set a small positive max
    else:
        # Default values if no data
        global_min = -0.2
        global_max = 0.5
    
    # Add padding to the global limits for the maximum only
    y_padding = (global_max - global_min) * 0.15  # 15% padding
    global_max += y_padding
    
    # Create a consistent predictor order across all plots
    # First, collect all predictors from all DVs
    all_predictors = set()
    predictor_effects_by_dv = {}
    
    for dv, iv in variable_combinations:
        dv_results = results_df[results_df['Dependent_Variable'].str.startswith(f"{dv}_")]
        if len(dv_results) == 0:
            continue
            
        predictors = dv_results['Predictor'].unique()
        predictor_effects = []
        
        for predictor in predictors:
            pre_data = dv_results[(dv_results['Predictor'] == predictor) & 
                         (dv_results['Data_Subset'] == 'pre')]
            post_data = dv_results[(dv_results['Predictor'] == predictor) & 
                                  (dv_results['Data_Subset'] == 'post')]
            
            if len(pre_data) > 0 and len(post_data) > 0:
                pre_beta = pre_data['Beta'].values[0]
                post_beta = post_data['Beta'].values[0]
                avg_effect = (abs(pre_beta) + abs(post_beta)) / 2
                predictor_effects.append((predictor, avg_effect))
                all_predictors.add(predictor)
        
        predictor_effects_by_dv[dv] = predictor_effects
    
    # Calculate average effect size across all DVs for each predictor
    predictor_avg_effects = {}
    for predictor in all_predictors:
        effects = []
        for dv, effects_list in predictor_effects_by_dv.items():
            for p, effect in effects_list:
                if p == predictor:
                    effects.append(effect)
        if effects:
            predictor_avg_effects[predictor] = sum(effects) / len(effects)
        else:
            predictor_avg_effects[predictor] = 0
    
    # Sort predictors by average effect size across all DVs
    global_sorted_predictors = sorted(predictor_avg_effects.keys(), 
                                     key=lambda p: predictor_avg_effects[p], 
                                     reverse=True)
    
    # Second pass: Create plots with consistent y-axis limits and predictor order
    for dv, iv in variable_combinations:
        print(f"\nCreating comparison plot for {dv} vs {iv}")
        
        # Filter results for this DV
        dv_results = results_df[results_df['Dependent_Variable'].str.startswith(f"{dv}_")]
        
        if len(dv_results) == 0:
            print(f"No results found for {dv}")
            continue
        
        # Get unique predictors for this DV
        dv_predictors = dv_results['Predictor'].unique()
        
        # Use the global predictor order, but only include predictors present for this DV
        sorted_predictors = [p for p in global_sorted_predictors if p in dv_predictors]
        
        # Create figure with fixed dimensions for all plots
        fig_width = 5.0  # Fixed width for all plots
        fig_height = 6.0  # Fixed height for all plots
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Set width of bars - narrower bars
        bar_width = 0.15  # Already reduced from 0.25 or 0.35
        
        # Reduce the spacing between pairs of bars by using a denser array for indices
        # Original:
        # indices = np.arange(len(sorted_predictors))
        
        # Updated - make the spacing between bar groups more compact:
        indices = np.arange(len(sorted_predictors)) * 0.5  # Multiply by 0.7 to compress spacing
        
        # Then keep the rest of the bar positioning the same
        pre_betas = []
        post_betas = []
        pre_cis = []
        post_cis = []
        significant_pairs = []
        P_Values = []
        # Process each predictor in sorted order
        for i, predictor in enumerate(sorted_predictors):
            # Get pre and post results for this predictor
            pre_data = dv_results[(dv_results['Predictor'] == predictor) & 
                          (dv_results['Data_Subset'] == 'pre')]
            post_data = dv_results[(dv_results['Predictor'] == predictor) & 
                                   (dv_results['Data_Subset'] == 'post')]
            
            # Skip if we don't have both pre and post data
            if len(pre_data) == 0 or len(post_data) == 0:
                pre_betas.append(0)
                post_betas.append(0)
                pre_cis.append(0)
                post_cis.append(0)
                significant_pairs.append(False)
                continue
            
            # Extract parameter estimates
            pre_beta = pre_data['Beta'].values[0]
            post_beta = post_data['Beta'].values[0]
            pre_betas.append(pre_beta)
            post_betas.append(post_beta)
            
            # Use standard errors directly if available, otherwise calculate from p-values
            if 'Std_Error' in pre_data.columns and not np.isnan(pre_data['Std_Error'].values[0]):
                pre_se = pre_data['Std_Error'].values[0]
                post_se = post_data['Std_Error'].values[0]
            else:
                pre_p = pre_data['p_Value'].values[0]
                post_p = post_data['p_Value'].values[0]
                pre_t = max(abs(stats.norm.ppf(pre_p / 2)), 0.001)
                post_t = max(abs(stats.norm.ppf(post_p / 2)), 0.001)
                pre_se = abs(pre_beta) / pre_t
                post_se = abs(post_beta) / post_t
            
            # 95% confidence intervals
            pre_ci = 1.96 * pre_se
            post_ci = 1.96 * post_se
            pre_cis.append(pre_ci)
            post_cis.append(post_ci)
            
            # Perform statistical test between pre and post
            z_stat = (pre_beta - post_beta) / np.sqrt(pre_se**2 + post_se**2)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
            P_Values.append(p_value)
            significant_pairs.append(p_value < 0.05)
        
        # Create the grouped bar plot with blue and red colors instead of grey
        pre_color = '#1E88E5'  # Blue
        post_color = '#E53935'  # Red
        
        pre_bars = ax.bar(indices - bar_width/2, pre_betas, bar_width, 
                         color=pre_color, alpha=0.9, label='Pre-changepoint',
                         yerr=pre_cis, capsize=3, edgecolor='black', linewidth=0.5)
        
        post_bars = ax.bar(indices + bar_width/2, post_betas, bar_width,
                          color=post_color, alpha=0.9, label='Post-changepoint',
                          yerr=post_cis, capsize=3, edgecolor='black', linewidth=0.5)
        
        # Set consistent y-axis limits for all plots
        ax.set_ylim(global_min, global_max)
        
        # Add a legend to identify pre and post changepoint bars
        legend = ax.legend(
            [pre_bars, post_bars],
            ['Pre-changepoint', 'Post-changepoint'],
            loc='upper right',
            frameon=True,
            framealpha=0.9,
            edgecolor='#DDDDDD',
            fontsize=11,
            ncol=1
        )
        
        # Add significance indicators with improved styling and larger stars
        for i, (is_significant, P_Value) in enumerate(zip(significant_pairs, P_Values)):
            if is_significant:
                # Calculate height for significance bar - closer to the bars
                # Use a smaller offset from the highest error bar
                bar_top = max(pre_betas[i] + pre_cis[i], post_betas[i] + post_cis[i])
                sig_line_height = bar_top + 0.05 * (global_max - global_min)  # Smaller offset
                
                # Ensure the significance line doesn't go beyond the plot limits
                sig_line_height = min(sig_line_height, global_max - 0.1 * (global_max - global_min))
                
                # Draw a line connecting the bars
                ax.plot([indices[i] - bar_width/2, indices[i] + bar_width/2], 
                        [sig_line_height, sig_line_height], 'k-', linewidth=1.5)
                
                # Add larger asterisks with better styling - closer to the line
                if P_Value < 0.001:
                    ax.text(indices[i], sig_line_height + 0.02 * (global_max - global_min), '***', 
                            ha='center', va='bottom', fontsize=24, fontweight='bold')
                elif P_Value < 0.01:
                    ax.text(indices[i], sig_line_height + 0.02 * (global_max - global_min), '**', 
                            ha='center', va='bottom', fontsize=24, fontweight='bold')
                elif P_Value < 0.05:
                    ax.text(indices[i], sig_line_height + 0.02 * (global_max - global_min), '*', 
                            ha='center', va='bottom', fontsize=24, fontweight='bold')
                elif is_significant:
                    ax.text(indices[i], sig_line_height + 0.02 * (global_max - global_min), '*', 
                            ha='center', va='bottom', fontsize=24, fontweight='bold')
        
        # Set labels and title with improved styling
        ax.set_ylabel('Parameter Estimate', fontsize=14, fontweight='bold')
        
        # Format title based on dependent variable
        title_text = f'Pre vs Post Changepoint Parameter Estimates'
        if 'normalized_brain_score' in dv:
            title_text += ': Brain Score'
        elif 'spearman' in dv:
            title_text += ': ClickMe Score'
        else:
            title_text += f': {dv}'
            
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=15)
        
        # Set x-ticks at the center of each group with improved styling
        ax.set_xticks(indices)
        
        # Format predictor names for better readability
        formatted_predictors = []
        for p in sorted_predictors:
            # Capitalize first letter and replace underscores with spaces
            formatted_p = p.replace('_', ' ').capitalize()
            # Special cases
            if p == 'is_transformer':
                formatted_p = 'ViT'
            elif p == 'is_cnn':
                formatted_p = 'CNN'
            elif p == 'is_imagenet':
                formatted_p = 'ImageNet multi-label accuracy'
            elif p == 'is_internet_scale':
                formatted_p = 'Internet-scale vision'
            elif p == 'is_adversarial':
                formatted_p = 'Adversarially trained'
            elif p == 'is_language':
                formatted_p = 'Internet-scale vision & language'
            elif p == 'multi_label_acc':
                formatted_p = 'Multi-label Accuracy'
            elif p == 'total_params':
                formatted_p = 'Number of parameters'
            elif p == 'num_layers':
                formatted_p = 'Number of layers'
            elif p == 'input_size':
                formatted_p = 'Input size'
            formatted_predictors.append(formatted_p)
            
        # Rotate x-tick labels more horizontally for better readability
        ax.set_xticklabels(formatted_predictors, rotation=45, ha='right', fontsize=10)
        
        # Add a horizontal line at y=0 with improved styling
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7, zorder=0)
        
        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
        
        # Set y-axis to use fixed number of ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Style the remaining spines
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color('#333333')
        
        # Adjust layout with more bottom padding for rotated labels
        plt.tight_layout(pad=1.2, rect=[0, 0.05, 1, 0.95])
        
        # Save the figure in multiple formats
        save_path_png = os.path.join(save_dir, f'{dv}_pre_post_comparison.png')
        save_path_pdf = os.path.join(save_dir, f'{dv}_pre_post_comparison.pdf')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()
        
        # Save the statistical test results
        stat_results = []
        for i, predictor in enumerate(sorted_predictors):
            pre_data = dv_results[(dv_results['Predictor'] == predictor) & 
                         (dv_results['Data_Subset'] == 'pre')]
            post_data = dv_results[(dv_results['Predictor'] == predictor) & 
                                  (dv_results['Data_Subset'] == 'post')]
            
            if len(pre_data) == 0 or len(post_data) == 0:
                continue
                
            pre_beta = pre_data['Beta'].values[0]
            post_beta = post_data['Beta'].values[0]
            
            # Calculate standard errors
            if 'Std_Error' in pre_data.columns and not np.isnan(pre_data['Std_Error'].values[0]):
                pre_se = pre_data['Std_Error'].values[0]
                post_se = post_data['Std_Error'].values[0]
            else:
                pre_p = pre_data['p_Value'].values[0]
                post_p = post_data['p_Value'].values[0]
                pre_t = max(abs(stats.norm.ppf(pre_p / 2)), 0.001)
                post_t = max(abs(stats.norm.ppf(post_p / 2)), 0.001)
                pre_se = abs(pre_beta) / pre_t
                post_se = abs(post_beta) / post_t
            
            # Calculate z-statistic and p-value
            z_stat = (pre_beta - post_beta) / np.sqrt(pre_se**2 + post_se**2)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            stat_results.append({
                'Predictor': predictor,
                'Pre_Beta': pre_beta,
                'Post_Beta': post_beta,
                'Pre_CI': 1.96 * pre_se,
                'Post_CI': 1.96 * post_se,
                'Z_Statistic': z_stat,
                'P_Value': p_value,
                'Significant': p_value < 0.05
            })
        
        if stat_results:
            stat_df = pd.DataFrame(stat_results)
            stat_df.to_csv(os.path.join(save_dir, f'{dv}_pre_post_stats.csv'), index=False)
            
            # Create a heatmap of the pre-post differences with improved styling
            # Use fixed dimensions for consistency
            plt.figure(figsize=(6.5, max(5, len(stat_results) * 0.4)))  # Fixed width, height scales with content
            
            # Prepare data for heatmap
            heatmap_data = pd.DataFrame({
                'Predictor': stat_df['Predictor'],
                'Pre-Post Difference': stat_df['Pre_Beta'] - stat_df['Post_Beta'],
                'P_Value': stat_df['P_Value']
            })
            
            # Sort by absolute difference
            heatmap_data['Abs_Diff'] = abs(heatmap_data['Pre-Post Difference'])
            heatmap_data = heatmap_data.sort_values('Abs_Diff', ascending=False)
            
            # Create colormap with significance indicators
            cmap = plt.cm.coolwarm
            
            # Plot heatmap
            ax = plt.subplot(111)
            im = ax.imshow(
                heatmap_data['Pre-Post Difference'].values.reshape(-1, 1),
                cmap=cmap,
                aspect='auto',
                vmin=-max(abs(heatmap_data['Pre-Post Difference'])),
                vmax=max(abs(heatmap_data['Pre-Post Difference']))
            )
            
            # Add colorbar with improved styling
            cbar = plt.colorbar(im)
            cbar.set_label('Pre-Post Difference', fontsize=12, fontweight='bold')
            
            # Format predictor names for better readability
            formatted_heatmap_predictors = []
            for p in heatmap_data['Predictor']:
                # Capitalize first letter and replace underscores with spaces
                formatted_p = p.replace('_', ' ').capitalize()
                # Special cases
                if p == 'is_transformer':
                    formatted_p = 'ViT'
                elif p == 'is_cnn':
                    formatted_p = 'CNN'
                elif p == 'is_imagenet':
                    formatted_p = 'ImageNet multi-label accuracy'
                elif p == 'is_internet_scale':
                    formatted_p = 'Internet-scale vision'
                elif p == 'is_adversarial':
                    formatted_p = 'Adversarially trained'
                elif p == 'is_language':
                    formatted_p = 'Internet-scale vision & language'
                elif p == 'multi_label_acc':
                    formatted_p = 'Multi-label Accuracy'
                elif p == 'total_params':
                    formatted_p = 'Number of parameters'
                elif p == 'num_layers':
                    formatted_p = 'Number of layers'
                elif p == 'input_size':
                    formatted_p = 'Input size'
                formatted_heatmap_predictors.append(formatted_p)
            
            # Set y-ticks (predictors) with improved styling
            ax.set_yticks(range(len(heatmap_data)))
            ax.set_yticklabels(formatted_heatmap_predictors, fontsize=12)
            
            # Remove x-ticks
            ax.set_xticks([])
            
            # Add significance indicators with improved styling - larger stars
            for i, P_Value in enumerate(heatmap_data['P_Value']):
                if P_Value < 0.001:
                    ax.text(0, i, '***', ha='center', va='center', fontsize=24, 
                           color='black', fontweight='bold')
            
            # Set title with improved styling
            title_text = f'Pre-Post Differences'
            if 'normalized_brain_score' in dv:
                title_text += ': Brain Score'
            elif 'spearman' in dv:
                title_text += ': ClickMe Score'
            else:
                title_text += f': {dv}'
                
            plt.title(title_text, fontsize=16, fontweight='bold', pad=15)
            
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Style the remaining spines
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(0.8)
                ax.spines[spine].set_color('#333333')
            
            # Save the heatmap in multiple formats
            plt.tight_layout()
            save_path_png = os.path.join(save_dir, f'{dv}_pre_post_heatmap.png')
            save_path_pdf = os.path.join(save_dir, f'{dv}_pre_post_heatmap.pdf')
            plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
            plt.savefig(save_path_pdf, bbox_inches='tight')
            plt.close()
            
        print(f"Saved publication-ready comparison plot and statistics for {dv} to {save_dir}")

# After all analyses are complete, create the pre vs post comparison plots
plot_pre_post_comparison(variable_combinations, all_anova_results, save_dir=anova_dir)

# Load the data
# Replace 'your_data.csv' with your actual file
df = joined_df

# Create a composite score for group 1 metrics using PCA
group1_metrics = ['multi_label_acc', 'results_sketch', 'results_imagenetv2_matched_frequency', 
                 'results_imagenet_r', 'results_imagenet_a']

# Make sure all required columns exist
for col in group1_metrics + ['spearman', 'normalized_brain_score']:
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in the dataset")

# Standardize the group 1 metrics
scaler = StandardScaler()
group1_data = scaler.fit_transform(df[group1_metrics].dropna())

# Apply PCA to get a composite score
pca = PCA(n_components=1)
group1_composite = pca.fit_transform(group1_data).flatten()

# Create a new dataframe with only the rows that have valid data for all metrics
valid_indices = df[group1_metrics + ['spearman', 'normalized_brain_score']].dropna().index
df_clean = df.loc[valid_indices].copy()
df_clean['group1_composite'] = group1_composite

# Function to calculate R² between variables
def calculate_r_squared(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.score(X, y)

# Calculate all pairwise and three-way R² values
# 1. Individual R² values
r2_1 = calculate_r_squared(df_clean[['group1_composite']], df_clean['group1_composite'])  # Always 1.0
r2_2 = calculate_r_squared(df_clean[['spearman']], df_clean['spearman'])  # Always 1.0
r2_3 = calculate_r_squared(df_clean[['normalized_brain_score']], df_clean['normalized_brain_score'])  # Always 1.0

# 2. Pairwise R² values
r2_12 = calculate_r_squared(df_clean[['group1_composite', 'spearman']], df_clean['group1_composite'])
r2_13 = calculate_r_squared(df_clean[['group1_composite', 'normalized_brain_score']], df_clean['group1_composite'])
r2_23 = calculate_r_squared(df_clean[['spearman', 'normalized_brain_score']], df_clean['spearman'])

# 3. Three-way R² value
r2_123 = calculate_r_squared(df_clean[['group1_composite', 'spearman', 'normalized_brain_score']], 
                             df_clean['group1_composite'])

# Calculate direct R² values (how much one explains the other)
r2_1_by_2 = calculate_r_squared(df_clean[['spearman']], df_clean['group1_composite'])
r2_1_by_3 = calculate_r_squared(df_clean[['normalized_brain_score']], df_clean['group1_composite'])
r2_2_by_1 = calculate_r_squared(df_clean[['group1_composite']], df_clean['spearman'])
r2_2_by_3 = calculate_r_squared(df_clean[['normalized_brain_score']], df_clean['spearman'])
r2_3_by_1 = calculate_r_squared(df_clean[['group1_composite']], df_clean['normalized_brain_score'])
r2_3_by_2 = calculate_r_squared(df_clean[['spearman']], df_clean['normalized_brain_score'])

# Calculate regions for Venn diagram
# A: Unique to group1_composite
# B: Unique to spearman
# C: Unique to normalized_brain_score
# AB: Shared between group1_composite and spearman
# AC: Shared between group1_composite and normalized_brain_score
# BC: Shared between spearman and normalized_brain_score
# ABC: Shared among all three

r2_1_by_23 = calculate_r_squared(df_clean[['spearman', 'normalized_brain_score']], df_clean['group1_composite'])
r2_2_by_13 = calculate_r_squared(df_clean[['group1_composite', 'normalized_brain_score']], df_clean['spearman'])
r2_3_by_12 = calculate_r_squared(df_clean[['group1_composite', 'spearman']], df_clean['normalized_brain_score'])

# Calculate regions for the Venn diagram
A = 1 - r2_1_by_23  # Variance in 1 not explained by 2 or 3
B = 1 - r2_2_by_13  # Variance in 2 not explained by 1 or 3
C = 1 - r2_3_by_12  # Variance in 3 not explained by 1 or 2

AB = r2_1_by_2 - (r2_1_by_23 - r2_1_by_3)  # Shared between 1 and 2 only
AC = r2_1_by_3 - (r2_1_by_23 - r2_1_by_2)  # Shared between 1 and 3 only
BC = r2_2_by_3 - (r2_2_by_13 - r2_2_by_1)  # Shared between 2 and 3 only

# The three-way intersection
ABC = r2_1_by_23 - ((r2_1_by_2 - (r2_1_by_23 - r2_1_by_3)) + (r2_1_by_3 - (r2_1_by_23 - r2_1_by_2)))

# Make sure all values are non-negative (there can be small negative values due to calculation precision)
regions = {'100': max(0, A), '010': max(0, B), '001': max(0, C), 
           '110': max(0, AB), '101': max(0, AC), '011': max(0, BC), 
           '111': max(0, ABC)}

# Create the Venn diagram
plt.figure(figsize=(10, 8))
v = venn3(subsets=regions, set_labels=('Performance Metrics', 'Spearman', 'Brain Score'))

# Add a title and adjust the appearance
plt.title('Variance Explained Between Metric Groups', fontsize=16)

# Add information about how much of each group is explained by others
txt = (f"Performance metrics explained by Spearman: {r2_1_by_2:.2f}\n"
       f"Performance metrics explained by Brain Score: {r2_1_by_3:.2f}\n"
       f"Performance metrics explained by both: {r2_1_by_23:.2f}\n\n"
       f"Spearman explained by Performance metrics: {r2_2_by_1:.2f}\n"
       f"Spearman explained by Brain Score: {r2_2_by_3:.2f}\n"
       f"Spearman explained by both: {r2_2_by_13:.2f}\n\n"
       f"Brain Score explained by Performance metrics: {r2_3_by_1:.2f}\n"
       f"Brain Score explained by Spearman: {r2_3_by_2:.2f}\n"
       f"Brain Score explained by both: {r2_3_by_12:.2f}\n\n"
       f"Shared variance among all three: {ABC:.2f}")

plt.figtext(0.1, -0.1, txt, fontsize=12, wrap=True)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Make room for the text

# Display the diagram
plt.savefig('variance_venn_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

# Print a summary of the variance sharing
print("Summary of variance sharing:")
print(f"Performance metrics (unique variance): {A:.2f}")
print(f"Spearman (unique variance): {B:.2f}")
print(f"Brain Score (unique variance): {C:.2f}")
print(f"Shared between Performance metrics and Spearman only: {AB:.2f}")
print(f"Shared between Performance metrics and Brain Score only: {AC:.2f}")
print(f"Shared between Spearman and Brain Score only: {BC:.2f}")
print(f"Shared among all three: {ABC:.2f}")

# Function to calculate the shared and unique variance between three variables
def calculate_variance_partitioning(df, var1, var2, var3):
    # Calculate the R² for each individual model
    X1 = sm.add_constant(df[[var1]])
    X2 = sm.add_constant(df[[var2]])
    X3 = sm.add_constant(df[[var3]])
    
    # Individual R² for each variable predicting the others
    model_1_2 = sm.OLS(df[var2], X1).fit()
    r2_1_2 = model_1_2.rsquared
    
    model_1_3 = sm.OLS(df[var3], X1).fit()
    r2_1_3 = model_1_3.rsquared
    
    model_2_1 = sm.OLS(df[var1], X2).fit()
    r2_2_1 = model_2_1.rsquared
    
    model_2_3 = sm.OLS(df[var3], X2).fit()
    r2_2_3 = model_2_3.rsquared
    
    model_3_1 = sm.OLS(df[var1], X3).fit()
    r2_3_1 = model_3_1.rsquared
    
    model_3_2 = sm.OLS(df[var2], X3).fit()
    r2_3_2 = model_3_2.rsquared
    
    # Models with pairs of predictors
    X12 = sm.add_constant(df[[var1, var2]])
    X13 = sm.add_constant(df[[var1, var3]])
    X23 = sm.add_constant(df[[var2, var3]])
    
    model_12_3 = sm.OLS(df[var3], X12).fit()
    r2_12_3 = model_12_3.rsquared
    
    model_13_2 = sm.OLS(df[var2], X13).fit()
    r2_13_2 = model_13_2.rsquared
    
    model_23_1 = sm.OLS(df[var1], X23).fit()
    r2_23_1 = model_23_1.rsquared
    
    # Calculate variance components
    # Unique variance of var1
    unique_var1 = r2_12_3 - r2_2_3
    
    # Unique variance of var2
    unique_var2 = r2_13_2 - r2_1_3
    
    # Unique variance of var3
    unique_var3 = r2_23_1 - r2_2_1
    
    # Shared variance between var1 and var2 (excluding var3)
    shared_var1_var2 = r2_1_2 + r2_2_1 - unique_var1 - unique_var2
    
    # Shared variance between var1 and var3 (excluding var2)
    shared_var1_var3 = r2_1_3 + r2_3_1 - unique_var1 - unique_var3
    
    # Shared variance between var2 and var3 (excluding var1)
    shared_var2_var3 = r2_2_3 + r2_3_2 - unique_var2 - unique_var3
    
    # Shared variance among all three variables
    total_shared = (r2_1_2 + r2_1_3 + r2_2_1 + r2_2_3 + r2_3_1 + r2_3_2) / 2 - unique_var1 - unique_var2 - unique_var3 - shared_var1_var2 - shared_var1_var3 - shared_var2_var3
    
    # Ensure non-negative values (due to potential estimation issues)
    unique_var1 = max(0, unique_var1)
    unique_var2 = max(0, unique_var2)
    unique_var3 = max(0, unique_var3)
    shared_var1_var2 = max(0, shared_var1_var2)
    shared_var1_var3 = max(0, shared_var1_var3)
    shared_var2_var3 = max(0, shared_var2_var3)
    total_shared = max(0, total_shared)
    
    # Return variance components
    return {
        'unique_var1': unique_var1,
        'unique_var2': unique_var2,
        'unique_var3': unique_var3,
        'shared_var1_var2': shared_var1_var2,
        'shared_var1_var3': shared_var1_var3, 
        'shared_var2_var3': shared_var2_var3,
        'total_shared': total_shared,
        'r2_1_2': r2_1_2,
        'r2_1_3': r2_1_3,
        'r2_2_1': r2_2_1,
        'r2_2_3': r2_2_3,
        'r2_3_1': r2_3_1,
        'r2_3_2': r2_3_2
    }

# Calculate variance components
variance_components = calculate_variance_partitioning(joined_df, 'multi_label_acc', 'spearman', 'normalized_brain_score')

# Extract components for the Venn diagram
venn_values = (
    variance_components['unique_var1'],
    variance_components['unique_var2'],
    variance_components['shared_var1_var2'],
    variance_components['unique_var3'],
    variance_components['shared_var1_var3'],
    variance_components['shared_var2_var3'],
    variance_components['total_shared']
)

# Create figure with fixed size
plt.figure(figsize=(10, 10))

# Create the Venn diagram with equal-sized circles
v = venn3(
    subsets=venn_values, 
    set_labels=('multi_label_acc', 'spearman', 'normalized_brain_score')
)

# Important: Format each label to show percentage with 2 decimal places
for label_id in ['100', '010', '001', '110', '101', '011', '111']:
    if v.get_label_by_id(label_id):
        current_value = float(v.get_label_by_id(label_id).get_text())
        # Use .2f to get exactly 2 decimal places, then add % sign
        percentage_str = f'{current_value*100:.2f}%'
        v.get_label_by_id(label_id).set_text(percentage_str)

plt.title('Shared Variance Between Metrics')
plt.show()

# Print the variance components with 2 decimal places
print(f"Unique variance of multi_label_acc: {variance_components['unique_var1']:.2%}")
print(f"Unique variance of spearman: {variance_components['unique_var2']:.2%}")
print(f"Unique variance of normalized_brain_score: {variance_components['unique_var3']:.2%}")
print(f"Shared variance between multi_label_acc and spearman: {variance_components['shared_var1_var2']:.2%}")
print(f"Shared variance between multi_label_acc and normalized_brain_score: {variance_components['shared_var1_var3']:.2%}")
print(f"Shared variance between spearman and normalized_brain_score: {variance_components['shared_var2_var3']:.2%}")
print(f"Shared variance among all three: {variance_components['total_shared']:.2%}")
