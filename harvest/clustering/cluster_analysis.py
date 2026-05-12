import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import ot
from sklearn.metrics.cluster import adjusted_rand_score
from typing import Optional, List, Union
import logging


class ClusterAnalysis:
    """Class for clustering analysis and evaluation."""
    
    @staticmethod
    def mclust_R(adata, num_cluster: int, modelNames: str = 'EEE', 
                used_obsm: str = 'emb_pca', random_seed: int = 2020):
        """Perform clustering using R's mclust package."""
        np.random.seed(random_seed)
        import rpy2.robjects as robjects
        robjects.r.library("mclust")
        from rpy2.robjects import FloatVector
        
        r_random_seed = robjects.r['set.seed']
        r_random_seed(random_seed)
        rmclust = robjects.r['Mclust']
        embedding = np.asarray(adata.obsm[used_obsm], dtype=np.float64, order="C")
        r_matrix = robjects.r.matrix(
            FloatVector(embedding.ravel(order="C")),
            nrow=embedding.shape[0],
            ncol=embedding.shape[1],
            byrow=True,
        )
        res = rmclust(r_matrix, num_cluster, modelNames)
        mclust_res = np.array(res[-2])
        
        adata.obs['mclust'] = mclust_res
        adata.obs['mclust'] = adata.obs['mclust'].astype('int')
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
        
        return adata, res
    
    @staticmethod
    def refine_label(adata, radius: int = 50, key: str = 'label') -> List[str]:
        """Refine cluster labels based on spatial neighbors."""
        n_neigh = radius
        new_type = []
        old_type = adata.obs[key].values
        
        # Calculate distance
        position = adata.obsm['spatial']
        distance = ot.dist(position, position, metric='euclidean')
        n_cell = distance.shape[0]
        
        for i in range(n_cell):
            vec = distance[i, :]
            index = vec.argsort()
            neigh_type = []
            for j in range(1, n_neigh+1):
                neigh_type.append(old_type[index[j]])
            max_type = max(neigh_type, key=neigh_type.count)
            new_type.append(max_type)
        
        new_type = [str(i) for i in list(new_type)]
        return new_type
    
    @staticmethod
    def plot_spatial(adata, section_id: str, cluster_keys: Optional[Union[str, List[str]]] = None):
        """Create and save spatial plots for clustering results."""
        if cluster_keys is None:
            cluster_keys = ["svm1_or_clust_refined"]
        elif isinstance(cluster_keys, str):
            cluster_keys = [cluster_keys]
        
        plt.rcParams["figure.figsize"] = (10, 8)
        
        for key in cluster_keys:
            if key not in adata.obs:
                continue
            
            fig, ax = plt.subplots()
            
            # Plot prediction
            sc.pl.spatial(adata, basis="spatial", color=key, show=False, ax=ax,
                         legend_fontoutline=2, legend_fontsize=15, legend_loc=None)
            
            # Remove spines and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set labels
            ax.set_xlabel('HarveST', fontsize=25)
            ax.set_ylabel('')
            
            # Save figure
            filename = f"HarveST_{section_id}_{key}-formal.png"
            plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close() 
