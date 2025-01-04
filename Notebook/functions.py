# We decided to create this py. file to store our functions and make our code 'cleaner'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 3.1. Segmentation based on Economic Value

# Functions to assign R, F, and M scores

# Function to assign R score
def r_score(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1

# Function to assign F and M scores
def fm_score(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

# Define a function to segment customers based on RFM scores
def segment_customer(row):
    if row['R_score'] == 4 and row['F_score'] == 4 and row['M_score'] == 4:
        return 'Champion'
    elif row['R_score'] >= 3 and row['F_score'] >= 3 and row['M_score'] >= 3:
        return 'Loyal Customer'
    elif row['R_score'] >= 3 and row['F_score'] >= 2 and row['M_score'] >= 2:
        return 'Potential Loyalist'
    elif row['R_score'] <= 2 and row['F_score'] >= 2 and row['M_score'] >= 2:
        return 'At Risk'
    elif row['R_score'] >= 2 and row['F_score'] <= 2 and row['M_score'] <= 2:
        return 'Need Attention'
    elif row['R_score'] <= 2 and row['F_score'] <= 2 and row['M_score'] <= 2:
        return 'Causal Shopper'
    elif row['R_score'] == 1 and row['F_score'] <= 2 and row['M_score'] <= 2:
        return 'Lost Customer'
    else:
        return 'Others'

# 3.4. Cluster Analysis and Profiling 

def cluster_profiles(df, label_columns, figsize, 
                     cmap="tab10",
                     compare_titles=None):
    """
    Pass df with labels columns of one or multiple clustering labels. 
    Then specify this label columns to perform the cluster profile according to them.
    """
    
    if compare_titles == None:
        compare_titles = [""]*len(label_columns)
        
    fig, axes = plt.subplots(nrows=len(label_columns), 
                             ncols=2, 
                             figsize=figsize, 
                             constrained_layout=True,
                             squeeze=False)
    for ax, label, titl in zip(axes, label_columns, compare_titles):
        # Filtering df
        drop_cols = [i for i in label_columns if i!=label]
        dfax = df.drop(drop_cols, axis=1)
        
        # Getting the cluster centroids and counts
        centroids = dfax.groupby(by=label, as_index=False).mean()
        counts = dfax.groupby(by=label, as_index=False).count().iloc[:,[0,1]]
        counts.columns = [label, "counts"]
        
        # Setting Data
        pd.plotting.parallel_coordinates(centroids, 
                                            label, 
                                            color = sns.color_palette(cmap),
                                            ax=ax[0])



        sns.barplot(x=label, 
                    hue=label,
                    y="counts", 
                    data=counts, 
                    ax=ax[1], 
                    palette=sns.color_palette(cmap),
                    legend=False
                    )

        #Setting Layout
        handles, _ = ax[0].get_legend_handles_labels()
        cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
        ax[0].annotate(text=titl, xy=(0.95,1.1), xycoords='axes fraction', fontsize=13, fontweight = 'heavy') 
        ax[0].axhline(color="black", linestyle="--")
        ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=13)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), 
                              rotation=40,
                              ha='right'
                              )
        
        ax[0].legend(handles, cluster_labels,
                     loc='center left', bbox_to_anchor=(1, 0.5), title=label
                     ) # Adaptable to number of clusters
        
        ax[1].set_xticks([i for i in range(len(handles))])
        ax[1].set_xticklabels(cluster_labels)
        ax[1].set_xlabel("")
        ax[1].set_ylabel("Absolute Frequency")
        ax[1].set_title("Cluster Sizes - {} Clusters".format(len(handles)), fontsize=13)
        
        
    
    # plt.subplots_adjust(hspace=0.4, top=0.90)
    plt.suptitle("Cluster Simple Profiling", fontsize=23)
    plt.show()

# Assessment of the clustering solution

# Function to get the total of sum squares to calculate R2
# using R² to evaluate how well the clustering solution explains the variability in the data 
def get_ss(df):
    ss = np.sum(df.var() * (df.count() - 1))
    return ss  # return sum of sum of squares of each df variable

def get_ss_variables(df):
    """Get the SS for each variable
    """
    ss_vars = df.var() * (df.count() - 1)
    return ss_vars

def r2_variables(df, labels):
    """Get the R² for each variable
    """
    sst_vars = get_ss_variables(df)
    ssw_vars = np.sum(df.groupby(labels).apply(get_ss_variables))
    return 1 - ssw_vars/sst_vars


# PROFILING

def plot_top_value_counts_by_cluster(df, column, cluster_column='merged_labels', top_n=3):

    cluster_value_counts = {}
    
    # Iterate over each unique cluster
    for cluster in df[cluster_column].unique():
        # Filter the data for the current cluster
        cluster_data = df[df[cluster_column] == cluster]
        
        # Calculate value counts and get the top N
        value_counts_df = cluster_data[column].value_counts().head(top_n).reset_index()
        value_counts_df.columns = [column, 'count']
        
        # Store the DataFrame in the dictionary
        cluster_value_counts[cluster] = value_counts_df

    for cluster, df in cluster_value_counts.items():
        print(f"Cluster {cluster} top {top_n} value counts for '{column}':")
        df.plot(x=column, y='count', kind='bar', title=f"Cluster {cluster} - Top {top_n} Value Counts for '{column}'",
                edgecolor='black', legend=False)
        plt.ylabel('Count')
        plt.tick_params(axis='x', labelrotation=0)
        plt.show()