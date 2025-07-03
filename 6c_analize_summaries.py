#!/usr/bin/env python3
"""
Script to analyze and plot the summary CSV files from the air pollution model comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import seaborn as sns

def load_summary_data(summary_file: str = 'SUMMARY.csv', 
                     by_hour_file: str = 'SUMMARY_BY_HOUR.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the summary CSV files.
    
    Args:
        summary_file: Path to the model performance summary CSV
        by_hour_file: Path to the RMSE by hour CSV
        
    Returns:
        Tuple of (summary_df, by_hour_df) DataFrames
    """
    try:
        summary_df = pd.read_csv(summary_file)
        print(f"Loaded summary data: {len(summary_df)} models")
    except FileNotFoundError:
        print(f"Warning: {summary_file} not found")
        summary_df = pd.DataFrame()
    
    try:
        by_hour_df = pd.read_csv(by_hour_file)
        print(f"Loaded by-hour data: {len(by_hour_df)} records")
    except FileNotFoundError:
        print(f"Warning: {by_hour_file} not found")
        by_hour_df = pd.DataFrame()
    
    return summary_df, by_hour_df

def create_comprehensive_plot(summary_df: pd.DataFrame, 
                            by_hour_df: pd.DataFrame,
                            output_file: str = 'model_analysis_summary.png',
                            dpi: int = 300) -> None:
    """
    Create a comprehensive matplotlib figure with multiple subplots.
    
    Args:
        summary_df: DataFrame with model performance summary
        by_hour_df: DataFrame with RMSE by hour data
        output_file: Output file path for the plot
        dpi: DPI for the output image
    """
    # Set up the figure with a grid layout
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid of subplots (3 rows, 3 columns)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Summary - Bar plot of Mean RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    if not summary_df.empty:
        bars = ax1.bar(range(len(summary_df)), summary_df['Mean_RMSE'], 
                      color=sns.color_palette("husl", len(summary_df)))
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Mean RMSE')
        ax1.set_title('Mean RMSE by Model')
        ax1.set_xticks(range(len(summary_df)))
        ax1.set_xticklabels(summary_df['Model_ID'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No summary data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Mean RMSE by Model')
    
    # 2. RMSE Range (Min-Max) by Model
    ax2 = fig.add_subplot(gs[0, 1])
    if not summary_df.empty:
        x_pos = np.arange(len(summary_df))
        ax2.errorbar(x_pos, summary_df['Mean_RMSE'], 
                    yerr=[summary_df['Mean_RMSE'] - summary_df['Min_RMSE'], 
                          summary_df['Max_RMSE'] - summary_df['Mean_RMSE']],
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Range by Model')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(summary_df['Model_ID'], rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'No summary data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('RMSE Range by Model')
    
    # 3. Model Names and IDs Table
    ax3 = fig.add_subplot(gs[0, 2])
    if not summary_df.empty:
        # Create a table showing model names and IDs
        ax3.axis('tight')
        ax3.axis('off')
        
        # Prepare table data
        table_data = []
        for _, row in summary_df.iterrows():
            model_id = row['Model_ID']
            # Try to get model name if available, otherwise use ID
            model_name = row.get('Model_Name', model_id) if 'Model_Name' in row else model_id
            table_data.append([model_id, model_name])
        
        # Create table
        table = ax3.table(cellText=table_data,
                         colLabels=['Model ID', 'Model Name'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.4, 0.6])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax3.set_title('Model Names and IDs', pad=20)
    else:
        ax3.text(0.5, 0.5, 'No summary data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Model Names and IDs')
    
    # 4. RMSE by Hour - Line plot for all models
    ax4 = fig.add_subplot(gs[1, :])
    if not by_hour_df.empty:
        # Remove rows with NaN RMSE values
        clean_df = by_hour_df.dropna(subset=['RMSE'])
        
        if not clean_df.empty:
            # Plot each model as a separate line
            for model_id in clean_df['Model_ID'].unique():
                model_data = clean_df[clean_df['Model_ID'] == model_id]
                ax4.plot(model_data['Predicted_Hour'], model_data['RMSE'], 
                        marker='o', linewidth=2, markersize=6, label=model_id)
            
            ax4.set_xlabel('Predicted Hour')
            ax4.set_ylabel('RMSE')
            ax4.set_title('RMSE by Predicted Hour - All Models')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No valid RMSE data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('RMSE by Predicted Hour - All Models')
    else:
        ax4.text(0.5, 0.5, 'No by-hour data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('RMSE by Predicted Hour - All Models')
    
    # 5. Model Performance Heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    if not summary_df.empty:
        # Create a heatmap of key metrics
        heatmap_data = summary_df[['Mean_RMSE', 'Std_RMSE', 'Min_RMSE', 'Max_RMSE']].values
        im = ax5.imshow(heatmap_data.T, cmap='RdYlGn_r', aspect='auto')
        ax5.set_xticks(range(len(summary_df)))
        ax5.set_xticklabels(summary_df['Model_ID'], rotation=45, ha='right')
        ax5.set_yticks(range(4))
        ax5.set_yticklabels(['Mean_RMSE', 'Std_RMSE', 'Min_RMSE', 'Max_RMSE'])
        ax5.set_title('Model Performance Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
        cbar.set_label('RMSE Value')
    else:
        ax5.text(0.5, 0.5, 'No summary data available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Model Performance Heatmap')
    
    # 6. RMSE Distribution Box Plot
    ax6 = fig.add_subplot(gs[2, 1])
    if not by_hour_df.empty:
        clean_df = by_hour_df.dropna(subset=['RMSE'])
        if not clean_df.empty:
            # Create box plot
            model_data_list = [clean_df[clean_df['Model_ID'] == model_id]['RMSE'].values 
                              for model_id in clean_df['Model_ID'].unique()]
            model_labels = clean_df['Model_ID'].unique()
            
            bp = ax6.boxplot(model_data_list, labels=model_labels, patch_artist=True)
            
            # Color the boxes
            colors = sns.color_palette("husl", len(model_labels))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax6.set_xlabel('Models')
            ax6.set_ylabel('RMSE')
            ax6.set_title('RMSE Distribution by Model')
            ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, 'No valid RMSE data available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('RMSE Distribution by Model')
    else:
        ax6.text(0.5, 0.5, 'No by-hour data available', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('RMSE Distribution by Model')
    
    # 7. Model Ranking by Performance
    ax7 = fig.add_subplot(gs[2, 2])
    if not summary_df.empty:
        # Sort by mean RMSE (lower is better)
        ranked_df = summary_df.sort_values('Mean_RMSE')
        bars = ax7.barh(range(len(ranked_df)), ranked_df['Mean_RMSE'],
                       color=sns.color_palette("RdYlGn_r", len(ranked_df)))
        ax7.set_yticks(range(len(ranked_df)))
        ax7.set_yticklabels(ranked_df['Model_ID'])
        ax7.set_xlabel('Mean RMSE')
        ax7.set_title('Model Ranking (Lower RMSE = Better)')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax7.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=8)
    else:
        ax7.text(0.5, 0.5, 'No summary data available', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Model Ranking')
    
    # Add overall title
    fig.suptitle('Air Pollution Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Save the figure
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Analysis plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def print_summary_statistics(summary_df: pd.DataFrame, by_hour_df: pd.DataFrame) -> None:
    """
    Print summary statistics to console.
    
    Args:
        summary_df: DataFrame with model performance summary
        by_hour_df: DataFrame with RMSE by hour data
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if not summary_df.empty:
        print(f"\nModel Names and IDs ({len(summary_df)} models):")
        print("-" * 50)
        print(f"{'Model ID':<15} {'Model Name':<35}")
        print("-" * 50)
        for _, row in summary_df.iterrows():
            model_id = row['Model_ID']
            # Try to get model name if available, otherwise use ID
            model_name = row.get('Model_Name', model_id) if 'Model_Name' in row else model_id
            print(f"{model_id:<15} {model_name:<35}")
        
        print(f"\nModel Performance Summary ({len(summary_df)} models):")
        print("-" * 40)
        print(f"Best performing model: {summary_df.loc[summary_df['Mean_RMSE'].idxmin(), 'Model_ID']} "
              f"(Mean RMSE: {summary_df['Mean_RMSE'].min():.4f})")
        print(f"Worst performing model: {summary_df.loc[summary_df['Mean_RMSE'].idxmax(), 'Model_ID']} "
              f"(Mean RMSE: {summary_df['Mean_RMSE'].max():.4f})")
        print(f"Average Mean RMSE across all models: {summary_df['Mean_RMSE'].mean():.4f}")
        print(f"Standard deviation of Mean RMSE: {summary_df['Mean_RMSE'].std():.4f}")
    
    if not by_hour_df.empty:
        clean_df = by_hour_df.dropna(subset=['RMSE'])
        if not clean_df.empty:
            print(f"\nRMSE by Hour Analysis ({len(clean_df)} valid records):")
            print("-" * 40)
            print(f"Number of unique models: {clean_df['Model_ID'].nunique()}")
            print(f"Number of unique hours: {clean_df['Predicted_Hour'].nunique()}")
            print(f"Hour range: {clean_df['Predicted_Hour'].min()} to {clean_df['Predicted_Hour'].max()}")
            print(f"Overall RMSE statistics:")
            print(f"  Mean: {clean_df['RMSE'].mean():.4f}")
            print(f"  Std:  {clean_df['RMSE'].std():.4f}")
            print(f"  Min:  {clean_df['RMSE'].min():.4f}")
            print(f"  Max:  {clean_df['RMSE'].max():.4f}")

def main() -> None:
    """Main function to run the analysis."""
    print("Loading summary data...")
    summary_df, by_hour_df = load_summary_data()
    
    if summary_df.empty and by_hour_df.empty:
        print("No data files found. Please run the dashboard script first to generate SUMMARY.csv and SUMMARY_BY_HOUR.csv")
        return
    
    print_summary_statistics(summary_df, by_hour_df)
    
    print("\nCreating comprehensive analysis plot...")
    create_comprehensive_plot(summary_df, by_hour_df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
