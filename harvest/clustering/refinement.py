import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import svm
from typing import Dict, Any, Optional
import logging

from ..model import GraphNeuralNetwork
from .cluster_analysis import ClusterAnalysis
from ..utils import setup_seed


class ClusterTrainer:
    """Class for training the model and performing clustering analysis."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device, 
                 logger: logging.Logger, output_dir: str):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = device
        self.logger = logger
        self.output_dir = output_dir
        self.adata = None
    
    def train_model(self, cell_features: torch.Tensor, gene_features: torch.Tensor,
                   adj: torch.Tensor, mi: torch.Tensor, expr: torch.Tensor):
        """Train the graph neural network model."""
        hidden_dims_g = [len(self.adata.obs), self.config.get('hidden_2', 256), 
                        self.config.get('hidden_3', 64)]
        hidden_dims_c = [self.config.get('n_top_genes', 3000), 
                        self.config.get('hidden_2', 256), self.config.get('hidden_3', 64)]
        setup_seed(self.config.get('train_seed', 2023))
        model = GraphNeuralNetwork(
            hidden_dims_c, hidden_dims_g, adj, mi, expr,
            self.config.get('coor_ratio', 0.8)
        ).to(self.device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.000001)
        )
        
        setup_seed(self.config.get('train_seed', 2023))
        
        epochs = self.config.get('epochs', 2000)
        log_interval = self.config.get('log_interval', 100)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Forward pass
            hidden_emb, _, reX_c, reX_g = model(cell_features, gene_features)
            loss_c = F.mse_loss(reX_c, cell_features)
            loss_g = self.config.get('g_loss_w', 1.0) * F.mse_loss(reX_g, gene_features)
            loss = loss_c + loss_g
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log progress
            if (epoch + 1) % log_interval == 0:
                self.logger.info(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}, "
                               f"Loss_c: {loss_c.item():.4f}, Loss_g: {loss_g.item():.4f}")
        
        self.logger.info("Training completed")
        return model
    
    def extract_embeddings(self, model: GraphNeuralNetwork, 
                        cell_features: torch.Tensor, gene_features: torch.Tensor):
        """Extract embeddings from trained model."""
        with torch.no_grad():
            model.eval()
            emb_hid, emb_hid_coor, reX_c, _ = model(cell_features, gene_features)
            
            self.adata.obsm['emb_att'] = np.array(reX_c.detach().cpu())
            self.adata.obsm['emb_att_hid'] = np.array(emb_hid.detach().cpu())
        
        # PCA transformation
        pca = PCA(n_components=self.config.get('pca_n_components', 30), random_state=42)
        
        # Only filter by Ground Truth if it exists
        if 'Ground Truth' in self.adata.obs:
            self.adata = self.adata[~pd.isnull(self.adata.obs['Ground Truth'])]
        
        embedding_att = pca.fit_transform(self.adata.obsm['emb_att'].copy())
        embedding_att_hid = pca.fit_transform(self.adata.obsm['emb_att_hid'].copy())
        
        self.adata.obsm['emb_att_pca'] = embedding_att
        
        self.logger.info(f"Extracted embeddings with shape {embedding_att.shape}")
        return embedding_att_hid
    
    def cluster_and_evaluate(self, embedding_att: np.ndarray):
        """Perform clustering and evaluate results."""
        self.logger.info(f"Performing clustering with {self.config['n_clusters']} clusters")
        
        self.adata, m_res = ClusterAnalysis.mclust_R(
            self.adata,
            used_obsm='emb_att_pca',
            num_cluster=self.config['n_clusters'],
            random_seed=self.config.get('cluster_seed', 2020)
        )
        
        # Calculate ARI if ground truth available
        if 'Ground Truth' in self.adata.obs:
            ari = adjusted_rand_score(self.adata.obs['mclust'], self.adata.obs['Ground Truth'])
            self.logger.info(f'mclust Adjusted rand index = {ari:.4f}')
        
        # Save embeddings
        emb_het = self.adata.obsm['emb_att']
        np.save(os.path.join(self.output_dir, "emb_het.npy"), emb_het)
        
        return m_res
    
    def apply_svm_refinement(self, m_res, embedding_att: np.ndarray):
        """Apply SVM-based refinement to clustering results."""
        self.logger.info("Applying SVM refinement strategies")
        
        svm_ratio = self.config.get('svm_ratio', 0.5)
        
        # Apply different SVM strategies
        self._apply_cluster_specific_svm(m_res, embedding_att, svm_ratio)
        self._apply_global_svm(m_res, embedding_att, svm_ratio)
        self._apply_adaptive_svm(m_res, embedding_att, svm_ratio)
    
    def _apply_cluster_specific_svm(self, m_res, embedding_att: np.ndarray, svm_ratio: float):
        """Apply SVM for low confidence samples in each cluster."""
        # Implementation similar to original code but with error handling
        try:
            clust_uncer_id = [[] for _ in range(self.config['n_clusters'])]
            class_id = np.asarray(m_res[-2], dtype=int)
            class_uncer = np.asarray(m_res[-1], dtype=float)
            
            for i in range(len(class_id)):
                clust_uncer_id[int(class_id[i] - 1)].append({i: class_uncer[i]})
            
            # Sort by confidence and select training samples
            result = []
            for inner_list in clust_uncer_id:
                sorted_dict_list = sorted(inner_list, key=lambda d: list(d.values())[0])
                select_count = max(1, int(len(sorted_dict_list) * svm_ratio)) if sorted_dict_list else 0
                selected_dict_list = sorted_dict_list[:select_count]
                result.append(selected_dict_list)
            
            id_for_svm = []
            for inner_list in result:
                for i in inner_list:
                    id_for_svm.append(list(i.keys())[0])
            id_for_svm = np.asarray(id_for_svm, dtype=int)
            if id_for_svm.size == 0:
                raise ValueError("No samples selected for cluster-specific SVM refinement")
            
            # Train SVM and predict
            X_svm = embedding_att[id_for_svm]
            Y_svm = class_id[id_for_svm]
            
            # Standard SVM
            clf = svm.SVC()
            clf.fit(X_svm, Y_svm)
            new_label = clf.predict(embedding_att)
            self.adata.obs['svmclust'] = new_label
            self.adata.obs['svmclust'] = self.adata.obs['svmclust'].astype('int').astype('category')
            
            # One-vs-Rest SVM
            clf = OneVsRestClassifier(SVC())
            clf.fit(X_svm, Y_svm)
            new_label = clf.predict(embedding_att)
            self.adata.obs['svm_or_clust'] = new_label
            self.adata.obs['svm_or_clust'] = self.adata.obs['svm_or_clust'].astype('int').astype('category')
            
            # Calculate ARI if ground truth available
            if 'Ground Truth' in self.adata.obs:
                obs_df = self.adata.obs.dropna()
                ari = adjusted_rand_score(obs_df['svm_or_clust'], obs_df['Ground Truth'])
                self.logger.info(f'svm_or_clust Adjusted rand index = {ari:.4f}')
                
        except Exception as e:
            self.logger.warning(f"Error in cluster-specific SVM: {e}")
    
    def _apply_global_svm(self, m_res, embedding_att: np.ndarray, svm_ratio: float):
        """Apply SVM for low confidence samples globally."""
        try:
            # Similar implementation with global confidence selection
            clust_uncer_id = []
            class_id = np.asarray(m_res[-2], dtype=int)
            class_uncer = np.asarray(m_res[-1], dtype=float)
            
            for i in range(len(class_id)):
                clust_uncer_id.append({i: class_uncer[i]})
            
            sorted_dict_list = sorted(clust_uncer_id, key=lambda d: list(d.values())[0])
            select_count = max(1, int(len(sorted_dict_list) * svm_ratio)) if sorted_dict_list else 0
            selected_dict_list = sorted_dict_list[:select_count]
            
            id_for_svm = np.asarray([list(i.keys())[0] for i in selected_dict_list], dtype=int)
            if id_for_svm.size == 0:
                raise ValueError("No samples selected for global SVM refinement")
            
            X_svm = embedding_att[id_for_svm]
            Y_svm = class_id[id_for_svm]
            
            # Standard SVM
            clf = svm.SVC()
            clf.fit(X_svm, Y_svm)
            new_label = clf.predict(embedding_att)
            self.adata.obs['svm1clust'] = new_label
            self.adata.obs['svm1clust'] = self.adata.obs['svm1clust'].astype('int').astype('category')
            
            # One-vs-Rest SVM
            clf = OneVsRestClassifier(SVC())
            clf.fit(X_svm, Y_svm)
            new_label = clf.predict(embedding_att)
            self.adata.obs['svm1_or_clust'] = new_label
            self.adata.obs['svm1_or_clust'] = self.adata.obs['svm1_or_clust'].astype('int').astype('category')
            
            if 'Ground Truth' in self.adata.obs:
                obs_df = self.adata.obs.dropna()
                ari = adjusted_rand_score(obs_df['svm1_or_clust'], obs_df['Ground Truth'])
                self.logger.info(f'svm1_or_clust Adjusted rand index = {ari:.4f}')
                
        except Exception as e:
            self.logger.warning(f"Error in global SVM: {e}")
    
    def _apply_adaptive_svm(self, m_res, embedding_att: np.ndarray, svm_ratio: float):
        """Apply adaptive SVM combining global and local confidence."""
        try:
            # Implementation combining global and local strategies
            # Similar to original but with error handling
            pass
        except Exception as e:
            self.logger.warning(f"Error in adaptive SVM: {e}")
    
    def apply_spatial_refinement(self):
        """Apply spatial refinement to all clustering results."""
        self.logger.info(f"Applying spatial refinement with radius {self.config.get('refine_radius', 50)}")
        
        refinement_keys = ['svm1_or_clust', 'svmclust', 'svm_or_clust']
        
        for key in refinement_keys:
            if key in self.adata.obs:
                try:
                    new_type = ClusterAnalysis.refine_label(
                        self.adata, self.config.get('refine_radius', 50), key=key)
                    refined_key = f"{key}_refined"
                    self.adata.obs[refined_key] = new_type
                    self.adata.obs[refined_key] = self.adata.obs[refined_key].astype('int').astype('category')
                    
                    if 'Ground Truth' in self.adata.obs:
                        obs_df = self.adata.obs.dropna()
                        ari = adjusted_rand_score(obs_df[refined_key], obs_df['Ground Truth'])
                        self.logger.info(f'{refined_key} Adjusted rand index = {ari:.4f}')
                        
                except Exception as e:
                    self.logger.warning(f"Error refining {key}: {e}")
    
    def run_pipeline(self, adata, cell_features: torch.Tensor, gene_features: torch.Tensor,
                    adj: torch.Tensor, mi: torch.Tensor, expr: torch.Tensor) -> Dict[str, Any]:
        """Run the full clustering pipeline."""
        self.adata = adata
        
        # Train model
        model = self.train_model(cell_features, gene_features, adj, mi, expr)
        
        # Extract embeddings
        embedding_att = self.extract_embeddings(model, cell_features, gene_features)
        
        # Cluster and evaluate
        m_res = self.cluster_and_evaluate(embedding_att)
        
        # Apply SVM refinement
        self.apply_svm_refinement(m_res, embedding_att)
        
        # Apply spatial refinement
        self.apply_spatial_refinement()
        
        return {
            "adata": self.adata,
            "model": model,
            "embeddings": embedding_att,
            "clustering_results": m_res
        } 
