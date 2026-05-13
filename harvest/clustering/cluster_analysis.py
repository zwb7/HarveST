import os
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
        from rpy2.robjects import FloatVector, StrVector
        
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
        r_set_rownames = robjects.r["rownames<-"]
        r_set_colnames = robjects.r["colnames<-"]
        r_matrix = r_set_rownames(
            r_matrix, StrVector([str(i) for i in range(embedding.shape[0])])
        )
        r_matrix = r_set_colnames(
            r_matrix, StrVector([f"dim_{i}" for i in range(embedding.shape[1])])
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
    def plot_spatial(
        adata,
        section_id: str,
        cluster_keys: Optional[Union[str, List[str]]] = None,
        output_file: Optional[str] = None,
    ):
        """Create and save one comparison figure for clustering results."""
        if cluster_keys is None:
            cluster_keys = ["svm1_or_clust_refined"]
        elif isinstance(cluster_keys, str):
            cluster_keys = [cluster_keys]

        available_keys = [key for key in cluster_keys if key in adata.obs]
        if not available_keys:
            return None

        fig_width = 8 * len(available_keys)
        fig, axes = plt.subplots(1, len(available_keys), figsize=(fig_width, 8), squeeze=False)

        for ax, key in zip(axes.flat, available_keys):
            sc.pl.spatial(
                adata,
                basis="spatial",
                color=key,
                show=False,
                ax=ax,
                legend_fontoutline=2,
                legend_fontsize=12,
                legend_loc=None,
            )

            # Remove spines and ticks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            title = key
            if "Ground Truth" in adata.obs:
                obs_df = adata.obs[[key, "Ground Truth"]].dropna()
                if not obs_df.empty:
                    ari = adjusted_rand_score(obs_df[key], obs_df["Ground Truth"])
                    title = f"{key}\nARI = {ari:.4f}"

            ax.set_title(title, fontsize=18)
            ax.set_xlabel('HarveST', fontsize=18)
            ax.set_ylabel('')

        if output_file is None:
            output_file = f"HarveST_{section_id}_clustering_comparison.png"
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)
        return output_file
