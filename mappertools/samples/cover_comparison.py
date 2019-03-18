import numpy as np
from kmapper import cover

from mappertools import covers

def overlap(c1, c2):
    ints = set(c1).intersection(set(c2))
    return len(ints) / max(len(c1), len(c2))

def print_stats(cube_entries):
    for i, hypercube in enumerate(cube_entries):
        print("There are %s points in cube %s/%s" % (hypercube.shape[0], i, len(bins)))
        #print("members: ", list(hypercube[:,1]))
    for i, (c1, c2) in enumerate(zip(cube_entries, cube_entries[1:])):
        print("Cardinality overlap %s" % (overlap(c1[:,0], c2[:,0])))


#data
data = np.random.normal(size=(100))
data = data[:,np.newaxis]
lens = data
ids = np.array([x for x in range(lens.shape[0])])
lens = np.c_[ids, lens]



## KMAPPER
# divided into 10 bins:
cov = cover.Cover(10,0.25)

## imitate usage in kmapper.py KeplerMapper.map:
print("***KMAPPER COVER***")
bins = cov.fit(lens)
bins = list(bins)  # extract list from generator
cube_entries = cov.transform(lens)
print_stats(cube_entries)
print()

## EMAPPER
print("***EQUALIZED PROJECTION COVER***")
epcov = covers.EPCover(10,0.5)
epcov.fit(lens)
cube_entries = epcov.transform(lens)
print_stats(cube_entries)
