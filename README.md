# mappertools

Various helper functions for doing Mapper analysis.
This package has been designed mainly for use with [kmapper](https://kepler-mapper.scikit-tda.org/).

## Dependencies
mappertools requires:

  - python
  - numpy
  - scipy
  - scikit-learn
  - pandas
  - networkx
  - pyclustering
  - matplotlib

## Install
Install from release package:

0. Navigate to the [releases page](https://github.com/emerson-escolar/mappertools/releases)
1. Download a dist package "`mappertools-*.*.*.tar.gz`", NOT "Source code"
2. Run `pip install mappertools-*.*.*.tar.gz` in command line

Alternatively, for developers install using the following.

## Develop
Install from source:

```
git clone https://github.com/emerson-escolar/mappertools
cd mappertools
pip install -e .
```

## Usage

1. mappertools/mapper contains several functions that may be useful for constructing mapper graphs. For example:
   - clustering.EPCover: "equalized projection cover"
   - hierarchical_clustering.HeuristicHierarchical: hierarchical clustering with automated heuristics to determine number of clusters
   - distances, filters, clustering: as labeled
2. mappertools/features contains several functions for analyzing mapper graphs
   - flare_balls: Compute "flareness" of entities in Mapper graph using the proposed definition in Escolar et al., "Mapping Firms' Locations in Technological Space"
   - flare_tree: Compute "flares" in G using the 0-persistent homology of centrality filtration.
   - mapper_stats: compute some summary statistics of a mapper graph


## License

GPL-3.0
