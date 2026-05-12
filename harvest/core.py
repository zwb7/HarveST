"""
Main HarveST class providing the public API
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from typing import Optional, Dict, Any, Union, List

from .preprocessing import DataPreprocessor
from .model import GraphNeuralNetwork
from .clustering import ClusterTrainer
from .utils import setup_logger, setup_seed, Config


class Harvest:
    """
    HarveST: A Graph Neural Network for Spatial Transcriptomics Clustering
    
    This class provides the main interface for spatial transcriptomics data
    preprocessing and clustering using graph neural networks.
    """
    
    def __init__(self, config: Optional[Union[str, Dict]] = None, 
                 output_dir: str = "./results",
                 random_seed: int = 2023,
                 device: Optional[str] = None):
        """
        Initialize HarveST instance.
        
        Parameters:
        -----------
        config : str or dict, optional
            Path to config file or config dictionary
        output_dir : str
            Output directory for results
        random_seed : int
            Random seed for reproducibility
        device : str, optional
            Device to use. Defaults to 'cuda:7' when not specified.
        """
        # Load configuration
        if config is None:
            self.config = self._get_default_config()
        elif isinstance(config, str):
            self.config = Config.load_config(config)
        else:
            self.config = config
            
        # Set up directories
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger(output_dir)
        
        # Set random seed
        self.random_seed = random_seed
        setup_seed(random_seed)
        
        # Set device
        if device is None:
            device = "cuda:7"
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
            if ":" in device:
                device_index = int(device.split(":", 1)[1])
                if device_index >= torch.cuda.device_count():
                    raise RuntimeError(
                        f"Requested device '{device}', but only {torch.cuda.device_count()} CUDA devices are visible."
                    )
        self.device = torch.device(device)
            
        self.logger.info(f"HarveST initialized with device: {self.device}")
        
        # Initialize data containers
        self.adata = None
        self.preprocessed_data = {}
        self.model = None
        self.clustering_results = {}
        
    def pre_process(self, 
                   data_path: str,
                   count_file: str = "filtered_feature_bc_matrix.h5",
                   n_top_genes: int = 3000,
                   n_bins: int = 5,
                   spatial_params: Optional[Dict] = None,
                   parallel_mi: bool = True,
                   n_jobs: int = -1,
                   load_ground_truth: bool = True,
                   truth_file_suffix: str = "_truth.txt") -> Dict[str, Any]:
        """
        Preprocess spatial transcriptomics data.
        
        Parameters:
        -----------
        data_path : str
            Path to the spatial transcriptomics data directory
        count_file : str
            Name of the count matrix file
        n_top_genes : int
            Number of top highly variable genes to select
        n_bins : int
            Number of bins for expression discretization
        spatial_params : dict, optional
            Parameters for spatial distance calculation
        parallel_mi : bool
            Whether to use parallel computation for mutual information
        n_jobs : int
            Number of parallel jobs (-1 for all CPUs)
        load_ground_truth : bool
            Whether to try loading ground truth labels
        truth_file_suffix : str
            Suffix for ground truth file
            
        Returns:
        --------
        dict : Preprocessed data including feature matrices and relationships
        """
        self.logger.info("Starting data preprocessing...")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            output_dir=self.output_dir,
            logger=self.logger,
            random_seed=self.random_seed
        )
        
        # Load and preprocess data
        self.adata = preprocessor.load_visium(data_path, count_file)
        
        # Try to load ground truth if requested
        if load_ground_truth:
            self._try_load_ground_truth(data_path, truth_file_suffix)
        
        adata_processed = preprocessor.preprocess_data(self.adata, n_top_genes)
        
        # Extract features
        feat_cell, feat_gene, cell_mapping, gene_mapping = preprocessor.extract_features(adata_processed)
        
        # Generate relationships
        self.logger.info("Generating gene-gene mutual information...")
        mi_matrix = preprocessor.generate_mutual_information(feat_gene, parallel_mi, n_jobs)
        
        self.logger.info("Generating cell-gene expression matrix...")
        expr_matrix = preprocessor.generate_expression_matrix(adata_processed, n_bins)
        
        self.logger.info("Generating cell-cell spatial relationships...")
        if spatial_params is None:
            spatial_params = {"p": 1.0, "radius": 150, "start": 1, "end": 300}
        adj_matrix = preprocessor.generate_spatial_adjacency(adata_processed, **spatial_params)
        
        # Store preprocessed data
        self.preprocessed_data = {
            "cell_features": feat_cell,
            "gene_features": feat_gene,
            "adjacency_matrix": adj_matrix,
            "mutual_information": mi_matrix,
            "expression_matrix": expr_matrix,
            "cell_mapping": cell_mapping,
            "gene_mapping": gene_mapping,
            "adata_processed": adata_processed  # Store the processed AnnData
        }
        
        # Save preprocessing results
        self._save_preprocessing_results()
        
        self.logger.info("Data preprocessing completed successfully!")
        return self.preprocessed_data
    
    def cluster(self,
                n_clusters: int,
                model_params: Optional[Dict] = None,
                training_params: Optional[Dict] = None,
                clustering_params: Optional[Dict] = None,
                plot_results: bool = True,
                save_results: bool = True,
                adata: Optional = None) -> Dict[str, Any]:
        """
        Perform clustering on preprocessed data.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to identify
        model_params : dict, optional
            Model architecture parameters
        training_params : dict, optional
            Training parameters
        clustering_params : dict, optional
            Clustering and refinement parameters
        plot_results : bool
            Whether to generate spatial plots
        save_results : bool
            Whether to save results to files
        adata : optional
            AnnData object to use. If None, uses stored data.
            
        Returns:
        --------
        dict : Clustering results including labels and evaluation metrics
        """
        if not self.preprocessed_data:
            raise ValueError("Data must be preprocessed first. Call pre_process() method or load_preprocessed_data().")
            
        self.logger.info("Starting clustering analysis...")
        
        # Use provided adata or create a minimal one from preprocessed data
        if adata is not None:
            working_adata = adata
        elif "adata_processed" in self.preprocessed_data:
            working_adata = self.preprocessed_data["adata_processed"]
        else:
            # Create a minimal AnnData object from preprocessed data
            self.logger.info("Creating minimal AnnData object from preprocessed data...")
            working_adata = self._create_minimal_adata()
        
        # Set default parameters
        if model_params is None:
            model_params = self._get_default_model_params()
        if training_params is None:
            training_params = self._get_default_training_params()
        if clustering_params is None:
            clustering_params = self._get_default_clustering_params()
            
        # Update n_clusters and n_top_genes
        clustering_params["n_clusters"] = n_clusters
        model_params["n_top_genes"] = self.preprocessed_data["gene_features"].shape[0]
        
        # Initialize trainer
        trainer = ClusterTrainer(
            config={**model_params, **training_params, **clustering_params},
            device=self.device,
            logger=self.logger,
            output_dir=self.output_dir
        )
        
        # Convert data to tensors
        tensors = self._prepare_tensors()
        
        # Run clustering pipeline
        results = trainer.run_pipeline(
            adata=working_adata,
            **tensors
        )
        
        self.clustering_results = results
        
        # Generate plots if requested
        if plot_results:
            self._generate_plots(results["adata"])
            
        # Save results if requested
        if save_results:
            self._save_clustering_results(results)
            
        self.logger.info("Clustering analysis completed successfully!")
        return results
    def discover_spatial_genes(self, 
                          domain_column: str = 'Ground Truth',
                          restart_prob: float = 0.5,
                          p_value_threshold: float = 0.01,
                          num_randomizations: int = 500,
                          domains: Optional[List[str]] = None,
                          use_preprocessed_networks: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Discover domain-specific spatial variable genes using Random Walk with Restart.
        
        Parameters:
        -----------
        domain_column : str
            Column in adata.obs containing domain information
        restart_prob : float
            Restart probability for random walk with restart
        p_value_threshold : float
            Threshold for significant p-values
        num_randomizations : int
            Number of random walks for null distribution
        domains : List[str], optional
            List of specific domains to analyze (if None, analyze all domains)
        use_preprocessed_networks : bool
            Whether to use networks from preprocessing (if available)
            
        Returns:
        --------
        dict : Dictionary mapping domain names to DataFrames of significant genes
        """
        if not self.preprocessed_data and not use_preprocessed_networks:
            raise ValueError("Data must be preprocessed first or use_preprocessed_networks must be True.")
        
        self.logger.info("Starting spatial gene discovery...")
        
        # Initialize SpatialGeneDiscovery
        from .spatial_gene_discovery import SpatialGeneDiscovery
        
        sgd = SpatialGeneDiscovery(
            data_dir=self.output_dir,  # Use output dir as working directory
            output_dir=os.path.join(self.output_dir, "spatial_genes"),
            logger=self.logger
        )
        
        # Use existing adata if available
        if "adata_processed" in self.preprocessed_data:
            sgd.adata = self.preprocessed_data["adata_processed"]
            sgd.num_cells = sgd.adata.shape[0]
            sgd.num_genes = sgd.adata.shape[1]
            
            # Extract domains
            if domain_column in sgd.adata.obs.columns:
                sgd.domains = sgd.adata.obs[domain_column].unique().tolist()
            else:
                raise ValueError(f"Domain column '{domain_column}' not found in data.")
        else:
            raise ValueError("No processed AnnData available. Please run preprocessing first.")
        
        # Use preprocessed networks if available
        if use_preprocessed_networks and self.preprocessed_data:
            # Create integrated network from preprocessed matrices
            cc_matrix = self.preprocessed_data["adjacency_matrix"]
            gg_matrix = self.preprocessed_data["mutual_information"] 
            cg_matrix = self.preprocessed_data["expression_matrix"]
            
            # Normalize matrices
            cg_norm = sgd._normalize_matrix(cg_matrix)
            gg_norm = sgd._normalize_matrix(gg_matrix)
            cc_norm = sgd._normalize_matrix(cc_matrix)
            
            # Create integrated adjacency matrix
            sgd.adj_matrix = np.zeros((sgd.num_cells + sgd.num_genes, 
                                    sgd.num_cells + sgd.num_genes))
            
            sgd.adj_matrix[:sgd.num_cells, :sgd.num_cells] = cc_norm
            sgd.adj_matrix[:sgd.num_cells, sgd.num_cells:] = cg_norm
            sgd.adj_matrix[sgd.num_cells:, :sgd.num_cells] = cg_norm.T
            sgd.adj_matrix[sgd.num_cells:, sgd.num_cells:] = gg_norm
            
            # Create gene name mapping
            sgd.gene_name_mapping = {gene: idx for idx, gene in enumerate(sgd.adata.var.index)}
            
            self.logger.info("Using preprocessed networks for spatial gene discovery")
        else:
            raise ValueError("Network data must be loaded separately when use_preprocessed_networks=False")
        
        # Find domain-specific genes
        results = sgd.find_domain_specific_genes(
            domain_column=domain_column,
            restart_prob=restart_prob,
            p_value_threshold=p_value_threshold,
            num_randomizations=num_randomizations,
            domains=domains
        )
        
        self.logger.info("Spatial gene discovery completed!")
        return results
    def load_preprocessed_data(self, 
                            matrix_dir: str,
                            data_path: str,
                            count_file: str = "filtered_feature_bc_matrix.h5",
                            adata_file: Optional[str] = None,
                            n_top_genes: int = 3000,
                            load_ground_truth: bool = True,
                            truth_file_suffix: str = "_truth.txt") -> Dict[str, Any]:
        """
        Load previously preprocessed data from directory and reconstruct AnnData.
        
        Parameters:
        -----------
        matrix_dir : str
            Directory containing preprocessed matrices
        data_path : str
            Path to the original spatial transcriptomics data directory
        count_file : str
            Name of the count matrix file
        adata_file : str, optional
            Path to saved AnnData file. If None, reconstructs from original data.
        n_top_genes : int
            Number of top highly variable genes (should match preprocessing)
        load_ground_truth : bool
            Whether to try loading ground truth labels
        truth_file_suffix : str
            Suffix for ground truth file
            
        Returns:
        --------
        dict : Loaded preprocessed data with proper AnnData object
        """
        self.logger.info(f"Loading preprocessed data from {matrix_dir}")
        
        # Load preprocessed matrices
        df_mi = pd.read_csv(os.path.join(matrix_dir, "df_gene2gene.csv"), index_col=0)
        df_adj = pd.read_csv(os.path.join(matrix_dir, "df_cell2cell.csv"), index_col=0)
        df_expr = pd.read_csv(os.path.join(matrix_dir, "df_cell2gene.csv"), index_col=0)
        feat_cell = np.load(os.path.join(matrix_dir, "feat_cell.npy"))
        feat_gene = np.load(os.path.join(matrix_dir, "feat_gene.npy"))
    
        self.preprocessed_data = {
            "cell_features": feat_cell,
            "gene_features": feat_gene,
            "adjacency_matrix": df_adj.values,
            "mutual_information": df_mi.values,
            "expression_matrix": df_expr.values
        }
        
        # Try to load existing processed AnnData file first
        adata_loaded = False
        if adata_file and os.path.exists(adata_file):
            self.logger.info(f"Loading preprocessed AnnData from {adata_file}")
            try:
                adata_processed = sc.read_h5ad(adata_file)
                
                # Verify the loaded AnnData matches the preprocessed matrices
                if self._verify_adata_consistency(adata_processed, feat_cell, feat_gene):
                    self.preprocessed_data["adata_processed"] = adata_processed
                    self.logger.info("Preprocessed AnnData loaded and verified successfully")
                    adata_loaded = True
                else:
                    self.logger.warning("Loaded AnnData doesn't match preprocessed matrices. Will reconstruct.")
            except Exception as e:
                self.logger.warning(f"Failed to load preprocessed AnnData: {e}. Will reconstruct.")
        
        # If no valid AnnData loaded, reconstruct from original data
        if not adata_loaded:
            self.logger.info(f"Reconstructing AnnData from original data: {data_path}")
            
            # Initialize preprocessor
            preprocessor = DataPreprocessor(
                output_dir=self.output_dir,
                logger=self.logger,
                random_seed=self.random_seed
            )
            
            # Load original data
            self.adata = preprocessor.load_visium(data_path, count_file)
            
            # Try to load ground truth if requested
            if load_ground_truth:
                self._try_load_ground_truth(data_path, truth_file_suffix)
            
            # Preprocess data to match the loaded matrices
            adata_processed = preprocessor.preprocess_data(self.adata, n_top_genes)
            
            # Verify that the reconstructed data matches the loaded matrices
            if not self._verify_adata_consistency(adata_processed, feat_cell, feat_gene):
                self.logger.warning(
                    "Reconstructed AnnData doesn't match preprocessed matrices. "
                    "This might be due to different preprocessing parameters or data versions."
                )
                # Still use the reconstructed data but warn user
            
            self.preprocessed_data["adata_processed"] = adata_processed
            
            # Extract mappings for consistency
            cell_mapping = dict(zip(adata_processed.obs.index, range(len(adata_processed.obs.index))))
            gene_mapping = dict(zip(adata_processed.var.index, range(len(adata_processed.var.index))))
            
            self.preprocessed_data.update({
                "cell_mapping": cell_mapping,
                "gene_mapping": gene_mapping
            })
            
            self.logger.info("AnnData reconstructed successfully from original data")
        
        self.logger.info("Preprocessed data loaded successfully!")
        return self.preprocessed_data

    def _verify_adata_consistency(self, adata, feat_cell: np.ndarray, feat_gene: np.ndarray) -> bool:
        """
        Verify that AnnData object is consistent with preprocessed feature matrices.
        
        Parameters:
        -----------
        adata : AnnData
            AnnData object to verify
        feat_cell : np.ndarray
            Cell feature matrix
        feat_gene : np.ndarray
            Gene feature matrix
            
        Returns:
        --------
        bool : True if consistent, False otherwise
        """
        try:
            # Check dimensions
            if adata.shape[0] != feat_cell.shape[0]:
                self.logger.warning(f"Cell count mismatch: AnnData {adata.shape[0]} vs features {feat_cell.shape[0]}")
                return False
                
            if adata.shape[1] != feat_gene.shape[0]:
                self.logger.warning(f"Gene count mismatch: AnnData {adata.shape[1]} vs features {feat_gene.shape[0]}")
                return False
            
            # Check if spatial coordinates exist
            if 'spatial' not in adata.obsm:
                self.logger.warning("No spatial coordinates found in AnnData")
                return False
                
            # Check if highly variable genes annotation exists
            if 'highly_variable' not in adata.var:
                self.logger.warning("No highly variable genes annotation found in AnnData")
                return False
                
            self.logger.info("AnnData consistency check passed")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error during AnnData consistency check: {e}")
            return False

    def _try_load_ground_truth(self, data_path: str, truth_file_suffix: str):
        """Try to load ground truth labels if available."""
        try:
            # Extract section ID from path
            section_id = os.path.basename(data_path.rstrip('/'))
            truth_file = os.path.join(data_path, f"{section_id}{truth_file_suffix}")
            
            if os.path.exists(truth_file):
                self.logger.info(f"Loading ground truth from {truth_file}")
                ann_df = pd.read_csv(truth_file, sep='\t', header=None, index_col=0)
                ann_df.columns = ['Ground Truth']
                self.adata.obs['Ground Truth'] = ann_df.loc[self.adata.obs_names, 'Ground Truth']
                self.logger.info("Ground truth labels loaded successfully")
            else:
                self.logger.info(f"No ground truth file found at {truth_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load ground truth: {e}")
    
    def _create_minimal_adata(self) -> 'sc.AnnData':
        """Create a minimal AnnData object from preprocessed data."""
        self.logger.info("Creating minimal AnnData object...")
        
        # Get dimensions
        n_cells = self.preprocessed_data["cell_features"].shape[0]
        n_genes = self.preprocessed_data["gene_features"].shape[0]
        
        # Create minimal AnnData
        X = self.preprocessed_data["cell_features"]
        adata = sc.AnnData(X=X)
        
        # Add basic obs and var info
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        
        # Create mock spatial coordinates if needed
        spatial_coords = np.random.rand(n_cells, 2) * 100  # Mock coordinates
        adata.obsm["spatial"] = spatial_coords
        
        # Add highly variable genes annotation
        adata.var['highly_variable'] = True
        
        self.logger.info(f"Created minimal AnnData with shape {adata.shape}")
        return adata
    
    def _try_load_ground_truth(self, data_path: str, truth_file_suffix: str):
        """Try to load ground truth labels if available."""
        try:
            # Extract section ID from path
            section_id = os.path.basename(data_path.rstrip('/'))
            truth_file = os.path.join(data_path, f"{section_id}{truth_file_suffix}")
            
            if os.path.exists(truth_file):
                self.logger.info(f"Loading ground truth from {truth_file}")
                ann_df = pd.read_csv(truth_file, sep='\t', header=None, index_col=0)
                ann_df.columns = ['Ground Truth']
                self.adata.obs['Ground Truth'] = ann_df.loc[self.adata.obs_names, 'Ground Truth']
                self.logger.info("Ground truth labels loaded successfully")
            else:
                self.logger.info("No ground truth file found")
        except Exception as e:
            self.logger.warning(f"Failed to load ground truth: {e}")
    
    def _prepare_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert preprocessed data to PyTorch tensors."""
        # Normalize adjacency matrix
        adj = torch.tensor(self.preprocessed_data["adjacency_matrix"], dtype=torch.float).to(self.device)
        adj = adj + torch.eye(adj.shape[0]).to(self.device)  # 添加自环
        adj[adj != 0] = 1.0
        
        # Normalize mutual information matrix
        mi = torch.tensor(self.preprocessed_data["mutual_information"], dtype=torch.float).to(self.device)
        # Add identity matrix to MI
        mi = mi + torch.eye(mi.shape[0]).to(self.device)
        row_sums = mi.sum(dim=1, keepdim=True)
        mi = mi / row_sums
        
        # Normalize expression matrix
        expr = torch.tensor(self.preprocessed_data["expression_matrix"], dtype=torch.float).to(self.device)
        row_sums = expr.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        expr = expr / row_sums
        
        return {
            "cell_features": torch.tensor(self.preprocessed_data["cell_features"], 
                                        dtype=torch.float).to(self.device),
            "gene_features": torch.tensor(self.preprocessed_data["gene_features"], 
                                        dtype=torch.float).to(self.device),
            "adj": adj,
            "mi": mi,
            "expr": expr
        }
    
    def _save_preprocessing_results(self):
        """Save preprocessing results to files."""
        # Save feature matrices
        np.save(os.path.join(self.output_dir, "feat_cell.npy"), 
                self.preprocessed_data["cell_features"])
        np.save(os.path.join(self.output_dir, "feat_gene.npy"), 
                self.preprocessed_data["gene_features"])
        
        # Save relationship matrices
        pd.DataFrame(self.preprocessed_data["mutual_information"]).to_csv(
            os.path.join(self.output_dir, "df_gene2gene.csv"))
        pd.DataFrame(self.preprocessed_data["adjacency_matrix"]).to_csv(
            os.path.join(self.output_dir, "df_cell2cell.csv"))
        pd.DataFrame(self.preprocessed_data["expression_matrix"]).to_csv(
            os.path.join(self.output_dir, "df_cell2gene.csv"))
        
        # Save processed AnnData if available
        if "adata_processed" in self.preprocessed_data:
            adata_file = os.path.join(self.output_dir, "adata_processed.h5ad")
            self.preprocessed_data["adata_processed"].write(adata_file)
            self.logger.info(f"Processed AnnData saved to {adata_file}")
    
    def _save_clustering_results(self, results: Dict):
        """Save clustering results."""
        # Save AnnData object
        results["adata"].write(os.path.join(self.output_dir, "clustering_results.h5ad"))
        
        # Save embeddings
        if "embeddings" in results:
            np.save(os.path.join(self.output_dir, "embeddings.npy"), results["embeddings"])
    
    def _generate_plots(self, adata):
        """Generate spatial plots for clustering results."""
        from .clustering.cluster_analysis import ClusterAnalysis
        
        plot_keys = ["mclust", "svm1_or_clust_refined"]
        ClusterAnalysis.plot_spatial(adata, "harvest_results", plot_keys)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "n_top_genes": 3000,
            "spatial_params": {"p": 1.0, "radius": 150},
            "model_params": {
                "hidden_2": 256,
                "hidden_3": 64,
                "coor_ratio": 0.8
            },
            "training_params": {
                "epochs": 2000,
                "learning_rate": 0.001,
                "weight_decay": 0.000001,
                "g_loss_w": 1.0
            },
            "clustering_params": {
                "pca_n_components": 30,
                "svm_ratio": 0.5,
                "refine_radius": 50,
                "cluster_seed": 2020
            }
        }
    
    def _get_default_model_params(self) -> Dict:
        """Get default model parameters."""
        return {
            "hidden_2": 256,
            "hidden_3": 64,
            "coor_ratio": 0.8
        }
    
    def _get_default_training_params(self) -> Dict:
        """Get default training parameters."""
        return {
            "epochs": 2000,
            "learning_rate": 0.001,
            "weight_decay": 0.000001,
            "g_loss_w": 1.0,
            "log_interval": 100,
            "train_seed": 2023
        }
    
    def _get_default_clustering_params(self) -> Dict:
        """Get default clustering parameters."""
        return {
            "pca_n_components": 30,
            "svm_ratio": 0.5,
            "refine_radius": 50,
            "cluster_seed": 2020
        }
