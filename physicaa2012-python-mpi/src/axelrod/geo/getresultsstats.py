#!/usr/bin/env python
##############################################################################
#
# getresultsstats.py - get stats about lattice, graphs, cultures
#
# File:    getresultsstats.py
# Author:  Alex Stivala
# Created: September 2014
#
##############################################################################

"""

Load full culture, lattice, network files written for some runs by
multiruninitmain.py etc. and use the functions in lib.py to get some
of the stats for those full results (they are already in a single row
in the reults.csv file but this can regenerate them for a specific run
given the full data).

Usage:
    getresultsstats.py resultsbasename

    resultsbasename is naem without extension of the results file; uses
     the .C, .L and .adj files

E.g.:

    getresultsstats.py results/500/results9-n500-q5-beta_p10-beta_s1-theta0.97-init_random_prob1.0-n_immutable0-50000

NB some informatoin is obtained from the filename e.g. value of n, q, etc.
so the filename must be that produced by the multiruninitmain.py script.
"""

import sys,os,glob
import getopt
import csv
import itertools
from numpy import array
import lib
                 
#-----------------------------------------------------------------------------
#
# functions
#
#-----------------------------------------------------------------------------


                 
#-----------------------------------------------------------------------------
#
# main
#
#-----------------------------------------------------------------------------

def usage(progname):
    """
    print usage msg and exit
    """
    sys.stderr.write("usage: " + progname + " resultsbasename\n")
    sys.exit(1)

def main():
    """
    See usage message in module header block
    """
    try:
        opts,args = getopt.getopt(sys.argv[1:], "")
    except:
        usage(sys.argv[0])
    for opt,arg in opts:
        usage(sys.argv[0])

    if len(args) != 1:
        usage(sys.argv[0])

    basename = sys.argv[1]

    # example filename:
    # results9-n500-q5-beta_p10-beta_s1-theta0.97-init_random_prob1.0-n_immutable0-50000
    sfilename = os.path.basename(basename)
#    print(sfilename)#XXX
    splitname = sfilename.split('-')
    modelmpirank = int(splitname[0][len('results'):])# not needed just check
    n = int(splitname[1][1:])
    q = int(splitname[2][1:])
    beta_p = int(splitname[3][len('beta_p'):])
    beta_s = int(splitname[4][len('beta_s'):])
    theta = float(splitname[5][len('theta'):])
    try:
        init_random_prob = float(splitname[6][len('init_random_prob'):])
    except ValueError:
        init_random_prob = float('nan')
    try:
        n_immutable = int(splitname[7][len('n_immutable')])
    except:
        n_immutable = 0
    try:
        time = int(splitname[8])
    except IndexError:
        time = -1
    

    # get value of m (lattice size) by reading from the results0.csv file
    resultsdir = os.path.dirname(basename)
#    print(glob.glob(os.path.join(resultsdir, 'results0.csv')))#XXX
    firstrow = list(itertools.islice(csv.reader(open(glob.glob(os.path.join(resultsdir, 'results0.csv'))[0])),1))[0]
    m = int(firstrow[1])

    # get value of F by lengh of vectors in .S file
    sfile = basename + '.S'
    F = None
    cultures = []
    for row in csv.reader(open(sfile)):
        culture_vector = array([int(x) for x in row[1][1:-1].split()])
        cultures.append(culture_vector)
        if F == None:
            F = len(culture_vector)
        else:
            assert len(culture_vector) == F
  
    (G, L, C, D, tend) = lib.loadNetwork(basename)

    density = G.density()
    cluster_coeff = G.transitivity_undirected()

    # graph with only edges of weight 1.0
    G1 = G.subgraph_edges(G.es.select(weight_eq = 1.0),
                          delete_vertices=False)
    g1_density = G1.density()
    g1_cluster_coeff = G1.transitivity_undirected()

    (overallMooreNeighbourhoodDiversity,
     overallMooreNeighbourhoodImmutableDiversity,
     overallMooreNeighbourhoodMutableDiversity) = (
        lib.calcMooreNeighbourhoodDiversity(n, m, F, n_immutable, C, L)
        )
    (mooreNeighbourhoodIdentityDiversity,
     mooreNeighbourhoodIdentityImmutableDiversity,
     mooreNeighbourhoodIdentityMutableDiversity) = (
        lib.calcMooreNeighbourhoodIdentityDiversity(n, m, F, n_immutable, C, L)
        )

    if G.ecount() > 0:
        communities = G.community_fastgreedy(weights="weight")
        num_communities = float(len(communities.as_clustering())) / n
        largest_community = float(max(communities.as_clustering().sizes())) / n 
        communities = communities.as_clustering()
        # modularity of the community structure 
        community_modularity = G.modularity(communities,
                  weights=[G.es()[i]["weight"] for i in xrange(G.ecount())])
    else:
        communities = [[x] for x in range(n)] 
        num_communities = 1.0
        largest_community = 1.0 / n
        community_modularity = float('nan')


    # Find the cultural clustering and the number of cultures
    vertex_label, num_cultures, cultures = lib.cultures_in(C, range(n))
    
    num_cultures = float(num_cultures) / n
    
    overall_diversity = lib.calcDiversityOfCluster(C, range(n))
        
    # Calculate social clustering of culture as the modularity of the network
    # based on cultural membership
    if G.ecount() > 0:
        social_clustering = G.modularity(vertex_label,weights="weight")
    else:
        social_clustering = float('nan')

    
    sys.stdout.write('n = %d, m = %d, F = %d, q = %d, beta_p = %d, beta_s = %d, theta = %f, init_random_prob = %f, n_immutable = %d\n, time = %d\n' % (n, m, F, q, beta_p, beta_s, theta, init_random_prob, n_immutable, time))
    sys.stdout.write('density = %f, cluster_coeff = %f, overallMooreNeighbourhoodDiversity = %f, mooreNeighourhoodIdentityDiversity = %f, overallMooreNeighbourhoodImmutableDiversity = %f\n' % (density, cluster_coeff, overallMooreNeighbourhoodDiversity, mooreNeighbourhoodIdentityDiversity, overallMooreNeighbourhoodImmutableDiversity))
    sys.stdout.write('g1_density = %f, g1_cluster_coeff = %f\n' %(g1_density, g1_cluster_coeff))

    sys.stdout.write('num_communities = %f, largest_community = %f\n' %(num_communities, largest_community))
    sys.stdout.write('community_modularity = %f\n' %(community_modularity))
    sys.stdout.write('num_cultures = %f, overall_diversity = %f, social_clustering = %f\n' %(num_cultures,overall_diversity, social_clustering))
    
if __name__ == "__main__":
    main()


