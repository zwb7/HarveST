"""
Example usage of HarveST package
"""

import os
from datetime import datetime
from harvest import Harvest
import torch
import numpy as np
import random


def make_run_output_dir(base_dir: str, data_path: str) -> str:
    """Create a per-sample, per-run output directory."""
    sample_id = os.path.basename(os.path.normpath(data_path)) or "unknown_sample"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, sample_id, run_id)


def setup_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
setup_seed(2023)
def main():
    # Example 1: Full pipeline from raw data (commented out for testing)
    """
    print("Running HarveST preprocessing and clustering...")
    
    # Initialize HarveST
    harvest = Harvest(
        config="config.yaml",
        output_dir="./results/preprocess_cluster",
        random_seed=2023,
        device="cuda:7",
    )
    
    # Preprocess data
    preprocessed_data = harvest.pre_process(
        data_path="/path/to/visium/data",
        n_top_genes=3000,
        spatial_params={"p": 1.0, "radius": 150}
    )
    
    # Perform clustering
    results = harvest.cluster(
        n_clusters=7,
        plot_results=True,
        save_results=True
    )
    
    print("Analysis completed!")
    print(f"Results saved to: {harvest.output_dir}")
    """
    
    # Example 2: Using preprocessed matrices with original data
    print("Running clustering on preprocessed data...")

    # Paths to your data
    # matrix_dir = "./harvest_results"  # Directory with preprocessed matrices
    matrix_dir = "../Dataset/LIBD/151674/preprocessed/"  # Directory with preprocessed matrices
    
    original_data_path = "../Dataset/LIBD/151674/"  # Original Visium data
    output_dir = make_run_output_dir("./results/cluster", original_data_path)
    
    harvest2 = Harvest(output_dir=output_dir, device="cuda:7")
    
    # Check if required directories and files exist
    if not os.path.exists(matrix_dir):
        print(f"Matrix directory {matrix_dir} does not exist!")
        print("Please run preprocessing first or change the matrix_dir path.")
        return
        
    if not os.path.exists(original_data_path):
        print(f"Original data path {original_data_path} does not exist!")
        print("Please provide the correct path to original Visium data.")
        return
    
    # Check if required preprocessed files exist
    required_files = [
        "df_gene2gene.csv",
        "df_cell2cell.csv", 
        "df_cell2gene.csv",
        "feat_cell.npy",
        "feat_gene.npy"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(matrix_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        print("Please run preprocessing first to generate these files.")
        return
    
    # Load preprocessed data with original data path
    harvest2.load_preprocessed_data(
        matrix_dir=matrix_dir,
        data_path=original_data_path,
        count_file="filtered_feature_bc_matrix.h5",  # Adjust if needed
        n_top_genes=3000,  # Should match preprocessing parameters
        load_ground_truth=True
    )
    
    # Run clustering
    results2 = harvest2.cluster(
        n_clusters=7,  # Adjust based on your data
        plot_results=True,  # Now can generate proper spatial plots
        save_results=True
    )
    
    print("Clustering completed!")
    print(f"Results saved to: {harvest2.output_dir}")
    
    # Print some results summary
    if "adata" in results2:
        adata = results2["adata"]
        print(f"\nResults summary:")
        print(f"- Data shape: {adata.shape}")
        print(f"- Clustering methods available: {[col for col in adata.obs.columns if 'clust' in col or 'mclust' in col]}")
        
        if 'Ground Truth' in adata.obs:
            # Print ARI scores if ground truth is available
            from sklearn.metrics.cluster import adjusted_rand_score
            
            if 'svm1_or_clust_refined' in adata.obs:
                ari = adjusted_rand_score(adata.obs['svm1_or_clust_refined'], adata.obs['Ground Truth'])
                print(f"- Best ARI score (svm1_or_clust_refined): {ari:.4f}")

if __name__ == "__main__":
    main()
