"""
Example usage of HarveST package for spatial gene discovery
"""

from harvest import Harvest

def example_spatial_gene_discovery():
    """Example of spatial gene discovery using HarveST."""
    print("Running spatial gene discovery...")
    
    # Initialize HarveST (assuming you have preprocessed data)
    harvest = Harvest(output_dir="./results/spatial_genes", device="cuda:7")
    
    # Load preprocessed data
    matrix_dir = "../Data/151674/preprocessed"  # Directory with preprocessed matrices
    original_data_path = "../Data/151674/"  # Original Visium data
    
    harvest.load_preprocessed_data(
        matrix_dir=matrix_dir,
        data_path=original_data_path,
        count_file="filtered_feature_bc_matrix.h5",
        n_top_genes=3000,
        load_ground_truth=True
    )
    
    # Discover spatial genes for all domains
    svg_results = harvest.discover_spatial_genes(
        domain_column='Ground Truth',
        restart_prob=0.1,
        p_value_threshold=0.01,
        num_randomizations=100,  # Reduced for faster example
        use_preprocessed_networks=True
    )
    
    # Print results summary
    print("\nSpatial Gene Discovery Results:")
    for domain, genes_df in svg_results.items():
        if not genes_df.empty:
            print(f"Domain {domain}: Found {len(genes_df)} significant genes")
            top_genes = genes_df.head(3)['Gene'].tolist()
            print(f"  Top genes: {', '.join(top_genes)}")
    
    return svg_results

def main():
    # Example of spatial gene discovery
    svg_results = example_spatial_gene_discovery()

if __name__ == "__main__":
    main()
