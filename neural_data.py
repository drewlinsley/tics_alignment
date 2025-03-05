import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from matplotlib.colors import LogNorm

# Set the style
plt.style.use('white')
sns.set_style("ticks")

# Load the data
neural_alignment = pd.read_csv('csvs/neural_alignment_george_central_it_131_170.csv')
model_analysis = pd.read_csv('csvs/model_analysis_gemini.csv')
model_metadata = pd.read_csv('csvs/model_metadata')

# Join the dataframes
joined_df = neural_alignment.merge(model_analysis, on='model_name', how='inner')
joined_df = joined_df.merge(model_metadata, left_on='model_name', right_on='model', how='inner')

# Create model type categories
def categorize_model_type(model_type):
    model_type = model_type.lower()
    if ('conv' in model_type or 'cnn' in model_type or 'resnet' in model_type or 'focal' in model_type):
        if 'vit' in model_type:
            return 'ConvViT'
        return 'CNN'
    elif ('vit' in model_type or 'transformer' in model_type or 'former' in model_type):
        return 'Visual Transformer'
    else:
        return 'Other'

# Create dataset categories
def categorize_dataset(dataset):
    dataset = str(dataset).lower()
    if any(d in dataset for d in ['yfc-semisl', 'in1k-dist', 'in1k-in1k', 'in1k-ap']):
        return 'ImageNet-1K'
    elif any(d in dataset for d in ['ig1b-wsl', 'in21k', 'in21k-selfsl', 'ig1b-swsl', 'jft300m-ns']):
        return 'Internet-scale'
    elif 'in1k-adv' in dataset:
        return 'Adversarial'
    else:
        return dataset

# Apply the categorization
joined_df['model_type_category'] = joined_df['model_type'].apply(categorize_model_type)
joined_df['dataset_category'] = joined_df['pretrain'].apply(categorize_dataset)

# Create combined category for coloring
joined_df['category'] = joined_df['model_type_category'] + ' - ' + joined_df['dataset_category']

# Calculate normalized brain score
joined_df['normalized_brain_score'] = joined_df['brain_score'] / joined_df['ceiling_score']

# Function to calculate pareto front
def compute_pareto_front(x, y, resolution=100):
    # Sort by x
    indices = np.argsort(x)
    x_sorted = x[indices]
    y_sorted = y[indices]
    
    # Initialize arrays for front
    x_front = [x_sorted[0]]
    y_front = [y_sorted[0]]
    
    # Calculate the pareto front (upper bound)
    max_y = y_sorted[0]
    for i in range(1, len(x_sorted)):
        if y_sorted[i] > max_y:
            max_y = y_sorted[i]
            x_front.append(x_sorted[i])
            y_front.append(y_sorted[i])
    
    # Interpolate for a smoother line
    if len(x_front) > 1:
        x_interp = np.linspace(min(x), max(x), resolution)
        model = LinearRegression()
        model.fit(np.array(x_front).reshape(-1, 1), y_front)
        y_interp = model.predict(x_interp.reshape(-1, 1))
        return x_interp, y_interp
    
    return np.array(x_front), np.array(y_front)

# Create a function to make the plots
def create_plot(x_column, y_column='normalized_brain_score', title=None, save_path=None):
    plt.figure(figsize=(10, 7))
    
    # Create scatter plot
    ax = sns.scatterplot(
        data=joined_df,
        x=x_column,
        y=y_column,
        hue='category',
        alpha=0.7,
        edgecolor='black',
        linewidth=1,
        s=100
    )
    
    # Calculate and plot pareto front
    x_data = joined_df[x_column].values
    y_data = joined_df[y_column].values
    valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
    if sum(valid_indices) > 1:
        x_front, y_front = compute_pareto_front(x_data[valid_indices], y_data[valid_indices])
        plt.plot(x_front, y_front, color='black', alpha=0.5, linewidth=2)
    
    # Set plot styling
    plt.xlabel(x_column, fontsize=14)
    plt.ylabel('Inferotemporal cortex\nBrain Score', fontsize=14)
    if title:
        plt.title(title, fontsize=16)
    
    # Despine right and top
    sns.despine(right=True, top=True)
    
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title='Model Type - Dataset', 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Create output directory if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Create the five scatter plots
create_plot('results-imagenet', 
           title='ImageNet Accuracy vs Brain Score', 
           save_path=os.path.join(output_dir, 'imagenet_vs_brain.png'))

create_plot('results-sketch', 
           title='Sketch Accuracy vs Brain Score', 
           save_path=os.path.join(output_dir, 'sketch_vs_brain.png'))

create_plot('results-imagenetv2-matched-frequency', 
           title='ImageNetV2 Accuracy vs Brain Score', 
           save_path=os.path.join(output_dir, 'imagenetv2_vs_brain.png'))

create_plot('results-imagenet-r', 
           title='ImageNet-R Accuracy vs Brain Score', 
           save_path=os.path.join(output_dir, 'imagenet_r_vs_brain.png'))

create_plot('results-imagenet-a', 
           title='ImageNet-A Accuracy vs Brain Score', 
           save_path=os.path.join(output_dir, 'imagenet_a_vs_brain.png'))

# Define the variables to analyze
dependent_variables = [
    'normalized_brain_score',
    'results-imagenet',
    'results-sketch',
    'results-imagenetv2-matched-frequency',
    'results-imagenet-r',
    'results-imagenet-a'
]

# Define the categorical and continuous predictors
categorical_predictors = [
    'model_type_category',
    'dataset_category',
    'category'  # combined predictor
]

continuous_predictors = [
    'input_size',
    'total_params',
    'num_transformer_layers',
    'num_normalizations',
    'num_skip_connections',
    'num_convolutional_layers',
    'num_dense_layers',
    'final_receptive_field_size',
    'num_strieds',
    'num_max_pools'
]

# Create a directory for ANOVA results
anova_dir = 'anova_results'
os.makedirs(anova_dir, exist_ok=True)

# Initialize a DataFrame to store ANOVA results for reporting
all_anova_results = pd.DataFrame(columns=['Dependent_Variable', 'Predictor', 'F_Value', 'p_Value', 'R_Squared'])

# Run ANOVA analyses for each dependent variable
for dv in dependent_variables:
    print(f"\nAnalyzing {dv}...")
    
    # Filter out rows with missing values for this dependent variable
    df_filtered = joined_df[~joined_df[dv].isna()].copy()
    
    if len(df_filtered) < 5:
        print(f"Not enough data for {dv}, skipping...")
        continue
    
    # 1. Run ANOVA for categorical predictors
    for predictor in categorical_predictors:
        df_cat = df_filtered[~df_filtered[predictor].isna()].copy()
        
        if len(df_cat[predictor].unique()) < 2:
            print(f"Not enough unique values in {predictor} for {dv}, skipping...")
            continue
        
        try:
            # Run ANOVA
            model = ols(f'{dv} ~ C({predictor})', data=df_cat).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Save results
            f_value = anova_table.iloc[0, 0]
            p_value = anova_table.iloc[0, 1]
            r_squared = model.rsquared
            
            all_anova_results = all_anova_results.append({
                'Dependent_Variable': dv,
                'Predictor': predictor,
                'F_Value': f_value,
                'p_Value': p_value,
                'R_Squared': r_squared
            }, ignore_index=True)
            
            print(f"ANOVA for {predictor}: F={f_value:.2f}, p={p_value:.4f}, R²={r_squared:.4f}")
            
            # Create and save a box plot for each significant categorical relationship (p < 0.05)
            if p_value < 0.05:
                plt.figure(figsize=(12, 6))
                ax = sns.boxplot(x=predictor, y=dv, data=df_cat)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.title(f'Effect of {predictor} on {dv}\nF={f_value:.2f}, p={p_value:.4f}, R²={r_squared:.4f}')
                plt.tight_layout()
                plt.savefig(os.path.join(anova_dir, f'boxplot_{dv}_{predictor}.png'), dpi=300)
                plt.close()
        
        except Exception as e:
            print(f"Error analyzing {predictor} for {dv}: {e}")
    
    # 2. Run regression analyses for continuous predictors
    for predictor in continuous_predictors:
        df_cont = df_filtered[~df_filtered[predictor].isna()].copy()
        
        if df_cont[predictor].std() == 0:
            print(f"No variation in {predictor} for {dv}, skipping...")
            continue
        
        try:
            # Run linear regression
            X = sm.add_constant(df_cont[predictor])
            model = sm.OLS(df_cont[dv], X).fit()
            
            # Extract results
            f_value = model.fvalue
            p_value = model.f_pvalue
            r_squared = model.rsquared
            
            all_anova_results = all_anova_results.append({
                'Dependent_Variable': dv,
                'Predictor': predictor,
                'F_Value': f_value,
                'p_Value': p_value,
                'R_Squared': r_squared
            }, ignore_index=True)
            
            print(f"Regression for {predictor}: F={f_value:.2f}, p={p_value:.4f}, R²={r_squared:.4f}")
            
            # Create and save a regression plot for each significant relationship (p < 0.05)
            if p_value < 0.05:
                plt.figure(figsize=(10, 6))
                sns.regplot(x=predictor, y=dv, data=df_cont, scatter_kws={'alpha':0.7})
                plt.title(f'Effect of {predictor} on {dv}\nF={f_value:.2f}, p={p_value:.4f}, R²={r_squared:.4f}')
                plt.tight_layout()
                plt.savefig(os.path.join(anova_dir, f'regplot_{dv}_{predictor}.png'), dpi=300)
                plt.close()
        
        except Exception as e:
            print(f"Error analyzing {predictor} for {dv}: {e}")

# Create a heatmap of all significant relationships
# Sort by R-squared to identify most important predictors
significant_results = all_anova_results[all_anova_results['p_Value'] < 0.05].copy()

if len(significant_results) > 0:
    # Save the significant results to a CSV
    significant_results.to_csv(os.path.join(anova_dir, 'significant_relationships.csv'), index=False)
    
    # Create a pivot table for the heatmap
    heatmap_data = significant_results.pivot_table(
        index='Predictor', 
        columns='Dependent_Variable',
        values='R_Squared',
        fill_value=0
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f')
    plt.title('R-squared Values for Significant Predictors (p < 0.05)')
    plt.tight_layout()
    plt.savefig(os.path.join(anova_dir, 'r_squared_heatmap.png'), dpi=300)
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
    plt.title('F-Values for Significant Predictors (p < 0.05)')
    plt.tight_layout()
    plt.savefig(os.path.join(anova_dir, 'f_value_heatmap.png'), dpi=300)
    plt.close()
    
    # Create summary bar plots of R-squared values for each dependent variable
    for dv in dependent_variables:
        dv_results = significant_results[significant_results['Dependent_Variable'] == dv].copy()
        
        if len(dv_results) > 0:
            dv_results = dv_results.sort_values('R_Squared', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='R_Squared', y='Predictor', data=dv_results)
            plt.title(f'Predictors of {dv} (Ranked by R-squared)')
            plt.tight_layout()
            plt.savefig(os.path.join(anova_dir, f'r_squared_ranking_{dv}.png'), dpi=300)
            plt.close()
else:
    print("No significant relationships found.")

print(f"\nAll ANOVA analyses complete. Results saved to {anova_dir}/")
