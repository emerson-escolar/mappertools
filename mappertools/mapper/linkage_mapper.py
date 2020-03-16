import warnings

warnings.warn(
    "linkage_mapper will be deprecated soon, use hierarchical_clustering instead",
    PendingDeprecationWarning
)

from mappertools.mapper.hierarchical_clustering import *
