# HarveST: Heterogeneous Graph Learning Framework for Revealing Spatial Transcriptomics Patterns

HarveST is a comprehensive Python package for spatial transcriptomics data analysis that combines graph neural networks with attention mechanisms for spatial domain identification and spatial variable gene discovery.

##  Key Features

### Spatial Domain Clustering

- **Graph Neural Networks**: Advanced GNN with attention mechanisms for heterogeneous networks
- **Multi-level Refinement**: Various clustering and spatial refinement strategies
- **Performance Evaluation**: Built-in metrics (ARI, NMI) for clustering assessment

### Spatial Variable Gene Discovery

- **Random Walk with Restart (RWR)**: Identify domain-specific spatial variable genes
- **Statistical Significance**: Empirical p-value calculation with null distribution
- **Multi-domain Analysis**: Simultaneous analysis across all spatial domains

### Comprehensive Visualization

- **Spatial Plots**: High-quality spatial domain and gene expression visualizations
- **Interactive Analysis**: Jupyter notebook for complete workflow
- **Publication-ready Figures**: Automated generation of analysis plots

## Data Availability

All data used in the paper are publicly available at https://doi.org/10.5281/zenodo.16883462.

## Installation

### Conda Environment Installation

```bash
# 1. Clone the repository
git clone https://github.com/Seven595/HarveST.git
cd HarveST

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate harvest-env

# 3. Install the package in editable mode
pip install -e .

# 4. Get the example data
cd ./Data/151674 && unzip preprocessed.zip

```

## Quick Start

### Complete Analysis Workflow

For a comprehensive analysis combining both clustering and spatial gene discovery, see our interactive Jupyter notebook:

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/HarveST_Complete_Analysis.ipynb
```

### Basic Clustering Analysis

```python
from harvest import Harvest

# Initialize HarveST
harvest = Harvest(output_dir="./results")

# Preprocess data
data = harvest.pre_process(
    data_path="../Data/151674",
    n_top_genes=3000
)

# Perform clustering
results = harvest.cluster(
    n_clusters=7,
    plot_results=True
)
```

### Spatial Variable Gene Discovery

```python
# Discover domain-specific genes
svg_results = harvest.discover_spatial_genes(
    domain_column='Ground Truth',
    restart_prob=0.1,
    p_value_threshold=0.01,
    num_randomizations=100
)

# Print results
for domain, genes_df in svg_results.items():
    print(f"Domain {domain}: {len(genes_df)} significant genes")
```

## Examples and Tutorials

### Available Examples

- **`examples/example_usage_cluster.py`**: Clustering analysis examples
- **`examples/example_usage_svg.py`**: Spatial variable gene discovery examples
- **`notebooks/HarveST_Complete_Analysis.ipynb`**: Complete interactive workflow

### Running Examples

```bash
# Clustering analysis
python examples/example_usage_cluster.py --preprocessed

# Spatial gene discovery
python examples/example_usage_svg.py --direct-interface

# Interactive complete analysis
jupyter notebook notebooks/HarveST_Complete_Analysis.ipynb
```

## Core Modules

### Data Preprocessing

- **10x Visium Support**: Native support for 10x Visium spatial transcriptomics data
- **H5AD Compatibility**: Load and process AnnData objects
- **Feature Engineering**: Automated highly variable gene selection and normalization
- **Network Construction**: Cell-cell, gene-gene, and cell-gene relationship matrices

### Graph Neural Network

- **Heterogeneous Architecture**: Separate processing for cell and gene features
- **Attention Mechanisms**: Multi-level attention for improved learning
- **Spatial Awareness**: Integration of spatial coordinates and gene expression

### Clustering Methods

- **Multiple Algorithms**: mclust, SVM-based refinement, spatial refinement
- **Hierarchical Approach**: From initial clustering to spatially-aware refinement
- **Evaluation Metrics**: Comprehensive performance assessment

### Spatial Gene Discovery

- **RWR Algorithm**: Random walk with restart on heterogeneous networks
- **Statistical Testing**: Empirical p-values with randomization testing
- **Domain Specificity**: Identify genes specific to each spatial domain

## Analysis Outputs

### Clustering Results

- **Spatial Domain Maps**: Visualization of identified spatial domains
- **Performance Metrics**: ARI, NMI scores comparing to ground truth
- **Refined Clusters**: Multiple clustering strategies with spatial refinement
- **Embeddings**: Low-dimensional representations for downstream analysis

### Spatial Gene Discovery Results

- **Gene Rankings**: Ranked lists of significant genes per domain
- **Statistical Metrics**: Scores, p-values, and fold changes
- **Spatial Expression Maps**: Visualization of top genes across tissue
- **CSV Outputs**: Detailed results tables for further analysis



## Requirements

- **Python** ≥ 3.8
- **PyTorch** == 1.13.1+cuda11.7
- **scanpy** ≥ 1.8.0
- **scikit-learn** ≥ 1.0.0
- **pandas** ≥ 1.3.0
- **numpy** ≥ 1.20.0
- **matplotlib** ≥ 3.5.0
- **seaborn** ≥ 0.11.0
- **tqdm** ≥ 4.60.0
- **scipy** ≥ 1.7.0
- **pot** (for optimal transport)
- **rpy2** (for R interface)

See `environment.yml` for the complete list of dependencies.

## Project Structure

```angelscript
HarveST/
├── assets/                    # Static assets and example figures
│   └── figures/
├── Data/                      # Example data kept in-place for script compatibility
│   └── 151674/
├── docs/                      # Repository notes and structure docs
├── harvest/                    # Main package
│   ├── core.py                # Main HarveST class
│   ├── spatial_gene_discovery.py  # SVG discovery module
│   ├── preprocessing/         # Data preprocessing modules
│   ├── model/                # Graph neural network models
│   ├── clustering/           # Clustering algorithms
│   └── utils/                # Utility functions
├── examples/                  # Example scripts and local config
│   ├── example_usage_cluster.py
│   ├── example_usage_svg.py
│   └── config.yaml
├── notebooks/                 # Interactive analysis notebooks
│   └── HarveST_Complete_Analysis.ipynb
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the spatial transcriptomics community for valuable feedback
- Built on top of excellent packages: scanpy, PyTorch, scikit-learn
- Inspired by advances in graph neural networks and spatial biology
