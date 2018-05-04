#  Copyright (C) 2011 Jens Pfau <jpfau@unimelb.edu.au>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

# modified by ADS to use MPI and seed srand with time+pid and to also 
# ensure unique tmp files so safe for parallel execution with MPI Python
# NB this involved extra command line pameter to model to not compatible
# with unmodified versions

import os, re, glob, sys
from random import randrange, random
from igraph import Graph, ADJ_UNDIRECTED
from numpy import array, nonzero, zeros, count_nonzero
from stats import ss, lpearsonr, mean



import time

#import model
import hackmodel

import csv
import commands
import ConfigParser

# location of R scripts to run
SCRIPTPATH = os.path.abspath(os.path.dirname(sys.argv[0])) + '/../../../../../scripts'

# Randomly draw one element from s based on the distribution given in w.
def randsample(s,w):
    cum = 0.0
    randTmp = random() * sum(w)
    
    for i in range(len(w)):
        cum += w[i]
        if randTmp <= cum:
            return s[i]






# Calculates the assortativity for the given graph.
def assortativity(graph, degrees=None):
    if degrees is None: degrees = graph.degree()
    degrees_sq = [deg**2 for deg in degrees]

    m = float(graph.ecount())
    
    if m == 0:
        return 0
    
    num1, num2, den1 = 0, 0, 0
    for source, target in graph.get_edgelist():
        num1 += degrees[source] * degrees[target]
        num2 += degrees[source] + degrees[target]
        den1 += degrees_sq[source] + degrees_sq[target]

    num1 /= m
    den1 /= 2*m
    num2 = (num2 / (2*m)) ** 2

    if (den1 - num2) == 0:
        return 0

    return (num1 - num2) / (den1 - num2)


# Returns a culture-dependent membership vector of the nodes in vector c and
# the number of cultures found overall among agents in c.
def cultures_in(C,c):
    # Label all nodes with their culture, whereby agents with the same culture
    # get the same label
    vertex_label = list([0]*len(c))
    
    cultures = list()
    
    vertex_label[0] = 0
    cultures.append(C[c[0]])
    current_label = 1
    for i in range(1,len(c)):        
        prev = -1
        for j in range(i):
            if len(nonzero(C[c[i]]-C[c[j]])[0]) == 0:
                prev = j
                break
        if prev >= 0:
            vertex_label[i] = vertex_label[prev]
        else:
            vertex_label[i] = current_label
            cultures.append(C[c[i]])
            current_label = current_label + 1
    
    
    # The number of cultures can be determined by the current pointer to the
    # label to be assigned next
    return vertex_label, current_label, cultures

# Calculates the diversity within those agents' cultures in C that are given in
# the list cluster.
def calcDiversityOfCluster(C, cluster):
    n = len(cluster)
    
    diversity = 0.0
    for i in range(n):
        for j in range(i):
            diversity += 1.0 - hackmodel.similarity(C[cluster[i]], C[cluster[j]])
    
    if n == 1:
        return 0.0
    else:
        return diversity / (n*(n-1)/2)


# Calculates the modal/prototype culture among the group of agents in C that is
# determined by the list cluster.
def calcModal(C, cluster, F, q):
    M = zeros((F, q))
    
    modal = array([0] * F)
    
    for i in range(len(cluster)):
        for j in range(F):
            M[j, C[cluster[i]][j]] += 1
    
    for i in range(F):
        max = 0
        maxJ = 0
        for j in range(q):
            if M[i,j] > max:
                maxJ = j
                max = M[i,j]
            modal[i] = maxJ
            
    return modal


# Calculates the within-cluster and between-cluster diversity as well as
# the within-/-between-cluster diversity ratio and the mapping of cluster size
# to diversity of the clusters given in the list clusters, whose cultures are
# defined in C.
def calcDiversity(C, clusters, F, q):
    wcd = 0.0
    modals = list()
    size_x_div = list()
   
    for i in range(len(clusters)):
        div = calcDiversityOfCluster(C, clusters[i])
        wcd += div
        modals.append(calcModal(C, clusters[i], F, q))
        size_x_div.append([len(clusters[i]), div])
        
    wcd = wcd / len(clusters)
    bcd = calcDiversityOfCluster(modals, range(len(modals)))
    
    if wcd == 0.0 or wcd == float('nan'):
        diversity_ratio = float('nan')
    else:
        diversity_ratio = bcd / wcd
    
    return wcd, bcd, diversity_ratio, size_x_div
    

# Calculates the Pearson correlation between two list, left and right.
def calcCorrelation(network, left, right):
    if network and ss(left) != 0 and ss(right) != 0 and max(left) != 0 and max(right) != 0:
        return lpearsonr(left,right)[0]
    else:
        return float('nan')    


# Calculate cophenetic correlation coefficient (runs external R script -
# easier than getting r2py to work)
def calc_cophenetic_cc(C):
    """
    Calcualte cophenetic correlation coefficient
    Parameters:
       C - list of numpy.array vectors
    Return value:  
       cophnetic correlation coefficient

    Expects output of R script to be a single line like:
 
    cophenetic_cc =  0.4546
    """
    tmpcsvfile = os.tempnam()
    csv.writer(open(tmpcsvfile, 'w')).writerows(C)
    r_out = os.popen('Rscript ' + SCRIPTPATH + '/' + 'compute_cophenetic_cc.R ' +
                     tmpcsvfile, 'r')
    cc = None
    for line in r_out:
        sline = line.split()
        if len(sline) == 3 and sline[0] == 'cophenetic_cc':
            try:
                cc = float(sline[2])
            except ValueError:
                cc = float('NaN')
    r_out.close()
    os.unlink(tmpcsvfile)
    return cc


# Calculate the number of connected components in the culture graph at
# value of theta, i.e. the graph where any two cultures with similarity
# >= theta are connected by an edge
def calcCultureNumComponents(cultures, theta, n):
    """
    Calculate the number of connected components in the culture graph at
    value of theta, i.e. the graph where any two cultures with similarity
     >= theta are connected by an edge

    Parametrs:
       cultures - list of numpy.array vectors
       theta - value of culture similarity threshold theta
       n - number of agents

    Return value:  
      Number of connected components in culture graph with edges
      when similarity >= theta (normalized by dividing by number of agens)
      nb also must have similarity > 0 since that is required for any
      interaction in Axelrod model
    """
    assert(len(cultures) == n)
    culture_graph = Graph(len(cultures))
    culture_graph.add_edges(
            [(i, j) for i in xrange(len(cultures)) for j in xrange(i)
                if (hackmodel.similarity(cultures[i], cultures[j]) >= theta
                     and hackmodel.similarity(cultures[i], cultures[j]) > 0.0) ]
                           )
    normalized_num_components = float(len(culture_graph.components())) / n
    return normalized_num_components


def calcMooreNeighbourhoodIdentityDiversity(n, m, F, n_immutable, C, L):
    """
    Neighbourhood diversity (statistics about lattice spatial regions) as 
    used in Neal & Neal (2013), in which just the fracton of neighbours with
    different cultures are used (univariate there i.e.  just one attribute,
    multivariate here, i.e. culture vector):
    (Neal & Neal (2013) probably uses Moore neighbourhood (8 neighbours) like
    the 'Segregation' library model in NetLogo (Wilensky 1997), but we should
    also use von Neumann neighbourhood as used in Axelrod (1997)):
    
    mooreNeighbourhoodIdentityDiversity: 
        fraction of neighbours in Moore neighbourhood with different
        culture

    mooreNeighbourhoodImmutableIdentityDiversity: 
        (as used in Segregation model) fraction of neighbours in Moore
        neighbourhood with different immutable culture

    mooreNeighbourhoodMutableIdentityDiversity: 
        fraction of neighbours in von Neumann neighbourhood with different
        mutable culture
        
    Parameters:
       n - number of agents
       m - lattic dimension
       F - cultur vector dimension
       n_immutable - number of immutable features at start of culture vector
       C - cultures, list of culture vectors (one for each agent)
       L - lattice, list of (x,y) tuples, one for each agent

    Return value:
       tuple (mooreNeighbourhoodIdentityDiversity,
              mooreNeighbourhoodImmutableIdentityDiversity,
              mooreNeighbourhoodMutableIdentityDiversity)

    """
    assert(len(C) == n)
    assert(len(L) == n)
    assert(len(C[0]) == F)
    assert(n_immutable <= F)
           
    # make dict to look up agent at lattice position
    # lattice_dict[(x,y)] is agent at (x,y)
    lattice_dict = dict([(L[i], i) for i in xrange(n)])
    
    # The Moore neighbourhood of a lattice position is the 8 surrounding
    # positions, i.e. those with a Chebysev distance (chessboard distance) of 1

    total_neighbours = 0
    total_diff_neighbours = 0
    for i in xrange(n):
        # FIXME: very inefficient, should just hardcode list of 8 neighbour
        # co-ordinates here (checkng for boundaries) instead of huge list
        # comprehension clause
        neighbours = [lattice_dict[(x,y)] for x in xrange(m)
                      for y in xrange(m)
                      if lattice_dict.has_key((x,y)) and
                      hackmodel.chebyshev_distance(L[i], (x,y)) == 1]
        diff_neighbours = [j for j in neighbours
                              if count_nonzero(C[i] != C[j]) > 0]
        total_neighbours += len(neighbours)
        total_diff_neighbours += len(diff_neighbours)
    mooreNeighbourhoodIdentityDiversity = (float(total_diff_neighbours)/
                                           float(total_neighbours))


    if n_immutable > 0:
        total_neighbours = 0
        total_diff_neighbours = 0
        for i in xrange(n):
            # FIXME: very inefficient, should just hardcode list of 8 neighbour
            # co-ordinates here (checkng for boundaries) instead of huge list
            # comprehension clause
            neighbours = [lattice_dict[(x,y)] for x in xrange(m)
                          for y in xrange(m)
                          if lattice_dict.has_key((x,y)) and
                          hackmodel.chebyshev_distance(L[i], (x,y)) == 1]
            diff_neighbours = [j for j in neighbours
                                  if count_nonzero(C[i][:n_immutable] !=
                                                   C[j][:n_immutable]) > 0]
            total_neighbours += len(neighbours)
            total_diff_neighbours += len(diff_neighbours)
        mooreNeighbourhoodImmutableIdentityDiversity = (
            float(total_diff_neighbours)/
            float(total_neighbours) )
    else:
        mooreNeighbourhoodImmutableIdentityDiversity = float('nan')


    if F - n_immutable > 0:
        total_neighbours = 0
        total_diff_neighbours = 0
        for i in xrange(n):
            # FIXME: very inefficient, should just hardcode list of 8 neighbour
            # co-ordinates here (checkng for boundaries) instead of huge list
            # comprehension clause
            neighbours = [lattice_dict[(x,y)] for x in xrange(m)
                          for y in xrange(m)
                          if lattice_dict.has_key((x,y))
                          and hackmodel.chebyshev_distance(L[i], (x,y)) == 1]
            diff_neighbours = [j for j in neighbours
                                  if count_nonzero(C[i][n_immutable:] !=
                                                   C[j][n_immutable:]) > 0]
            total_neighbours += len(neighbours)
            total_diff_neighbours += len(diff_neighbours)
        mooreNeighbourhoodMutableIdentityDiversity = (
            float(total_diff_neighbours)/
            float(total_neighbours) )
    else:
        mooreNeighbourhoodMutableIdentityDiversity = float('nan')


    return (mooreNeighbourhoodIdentityDiversity,
            mooreNeighbourhoodImmutableIdentityDiversity,
            mooreNeighbourhoodMutableIdentityDiversity)
            
 

def calcMooreNeighbourhoodDiversity(n, m, F, n_immutable, C, L):
    """
    Neighbourhood diversity (statistics about lattice spatial regions)
    using average distance between culture vectors in the Moore neighbourhood
    of each agent.

    Note there are two averages here: first for every agent the average culture
    distance from that agent of all others in its Moore neighbourhood,
    and then this is averaged over all agents.

    
    overallMooreNeighbourhoodDiversity: 
        average distance between culture vectors in Moore neighbourhood

    overallMooreNeighbourhoodImmutableDiversity: 
        average distance between immutable features in culture
        vectors in Moore neighbourhood

    overallMooreNeighbourhoodMutableDiversity: 
        fraction of neighbours in von Neumann neighbourhood with different
        mutable culture
        
    Parameters:
       n - number of agents
       m - lattic dimension
       F - cultur vector dimension
       n_immutable - number of immutable features at start of culture vector
       C - cultures, list of culture vectors (one for each agent)
       L - lattice, list of (x,y) tuples, one for each agent

    Return value:
       tuple (overallMooreNeighbourhoodDiversity,
              overallMooreNeighbourhoodImmutableDiversity,
              overallMooreNeighbourhoodMutableDiversity)

    """
    assert(len(C) == n)
    assert(len(L) == n)
    assert(len(C[0]) == F)
    assert(n_immutable <= F)
           
    # make dict to look up agent at lattice position
    # lattice_dict[(x,y)] is agent at (x,y)
    lattice_dict = dict([(L[i], i) for i in xrange(n)])
    
    # The Moore neighbourhood of a lattice position is the 8 surrounding
    # positions, i.e. those with a Chebysev distance (chessboard distance) of 1

    overallMooreNeighbourhoodDiversity = 0.0
    overallMooreNeighbourhoodImmutableDiversity = 0.0
    overallMooreNeighbourhoodMutableDiversity = 0.0
    for i in xrange(n):
        # FIXME: very inefficient, should just hardcode list of 8 neighbour
        # co-ordinates here (checkng for boundaries) instead of huge list
        # comprehension clause
        neighbours = [lattice_dict[(x,y)] for x in xrange(m)
                      for y in xrange(m)
                      if lattice_dict.has_key((x,y)) and
                      hackmodel.chebyshev_distance(L[i], (x,y)) == 1]

        if len(neighbours) > 0:
            overallMooreNeighbourhoodDiversity += (sum([ (1.0 - 
                                             hackmodel.similarity(C[i], C[j]))
                                             for j in neighbours])
                                       / len(neighbours))
            if n_immutable > 0:
                overallMooreNeighbourhoodImmutableDiversity += (sum([ (1.0 - 
                  hackmodel.similarity(C[i][:n_immutable], C[j][:n_immutable])) 
                                                           for j in neighbours])
                                                     / len(neighbours))
            if F - n_immutable > 0:
                    overallMooreNeighbourhoodMutableDiversity += (sum([ (1.0 - 
                  hackmodel.similarity(C[i][n_immutable:], C[j][n_immutable:])) 
                                                          for j in neighbours])
                                                    / len(neighbours))
    overallMooreNeighbourhoodDiversity /= n
    if n_immutable > 0:
        overallMooreNeighbourhoodImmutableDiversity /= n
    else:
        overallMooreNeighbourhoodImmutableDiversity = float('nan')
    if F - n_immutable > 0:
        overallMooreNeighbourhoodMutableDiversity /= n
    else:
        overallMooreNeighbourhoodMutableDiversity = float('nan')

    return (overallMooreNeighbourhoodDiversity,
            overallMooreNeighbourhoodImmutableDiversity,
            overallMooreNeighbourhoodMutableDiversity)



            
    
def calcNumAndMaxSizeOfLatticeCulturalRegions(n, C, L):
    """
    Compute the number of regions and the largest region size, where
    region here is the traditional Axelrod region on the lattice,
    i.e. agents with the same culture in contiguous sites on the 
    lattice (a 'cultural region' in Axelrod's terminology)

    Parameters:
       n - number of agents
       C - cultures, list of culture vectors (one for each agent)
       L - lattice, list of (x,y) tuples, one for each agent

    Return value:
       tuple (numRegions, maxRegionSize): number of cultural regions and
       maximum size of a cultural region.
       Both are normlized by dividing by n.
    """
    assert(len(C) == n)
    assert(len(L) == n)
    # we can conventiently compute these by building a 'cultural region graph'
    # which is the graph where two nodes (agents) are connected exactly
    # when both (a) they have the same culture and (b) they are adjacent
    # on the lattice (their Manhattan distance is 1).
    cultural_region_graph = Graph(n)
    cultural_region_graph.add_edges(
            [(i, j) for i in xrange(n) for j in xrange(i)
                if hackmodel.similarity(C[i], C[j]) == 1 and
                   hackmodel.manhattan_distance(L[i], L[j]) == 1 ] 
    )
    components = cultural_region_graph.components()
    normalized_num_regions = float(len(components)) / n
    normalized_max_region_size = float(max(components.sizes())) / n
    return (normalized_num_regions, normalized_max_region_size)


# Gets all relevant statistics about the graph, about culture and location of agents
# and writes this information to a file.
def writeStatistics(statsWriter, F, phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b,
                    r, s, t, q, theta, init_random_prob,
                    n_immutable,run, G, L, C, m, toroidal, network, timestep=-1, lastG = None, lastL = None, lastC = None,
                    componentClusterWriter = None, communityClusterWriter = None, cultureClusterWriter = None, ultrametricWriter =None, correlation = False, differences = None, noise = -1, radius = -1):
    n = G.vcount()
    
    if init_random_prob == None:
        init_random_prob = "NA"

    pre = [n, m, F, phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b, r, s, t, q, theta, init_random_prob, n_immutable]
    
    if radius != -1:
        pre.append(radius)

    if noise != -1:
        pre.append(noise)    
    
    pre.append(run)

    if timestep != -1:
        pre.append(timestep)     
    
    # Calculate graph metrics
    first = time.time()
    avg_path_length = G.average_path_length(unconn=False)
    print 'path length: ', time.time() - first
    
    first = time.time()
#    if correlation:
#        dia = G.diameter()
#    else:
#        dia = float('nan')
    dia = float('nan')
    print 'diameter: ', time.time() - first
    #den = float(G.ecount())/(n*(n-1)/2)
    
    first = time.time()
    avg_degree = mean(G.degree(range(n)))
    print 'avg degree: ', time.time() - first
    
    first = time.time()
    cluster_coeff = G.transitivity_undirected()  # global clustering coefficient
    local_cluster_coeff = G.transitivity_avglocal_undirected() # avg local cc
    print 'cluster: ', time.time() - first
    
    first = time.time()
    ass = assortativity(G)
    print 'assortativity: ', time.time() - first

    first = time.time()
    density = G.density()
    print 'density: ', time.time() - first
    
    first = time.time()
    # Calculate correlation between different spaces
    if correlation:
        k = 0
        physical = [0] * (n*(n-1)/2)
        social = [0] * (n*(n-1)/2)
        culture = [0] * (n*(n-1)/2)
        for i in range(n):
            for j in range(i):
                physical[k] = 1.0 - hackmodel.distance(L[i], L[j], m, toroidal)[1]
                social[k] = G.es[G.get_eid(i,j)]["weight"] if G.are_connected(i,j) else 0
                culture[k] = hackmodel.similarity(C[i], C[j])
                k = k+1
        
        corr_soc_phy = calcCorrelation(network, social, physical)
        corr_soc_cul = calcCorrelation(network, social, culture)
        corr_phy_cul = calcCorrelation(network, physical, culture)  
    else:
        corr_soc_phy = float('nan')
        corr_soc_cul = float('nan')
        corr_phy_cul = float('nan')
    print 'correlation: ', time.time() - first
        
    first = time.time()
    # Find the cultural clustering and the number of cultures
    vertex_label, num_cultures, cultures = cultures_in(C, range(n))
    
    num_cultures = float(num_cultures) / n
    
    overall_diversity = calcDiversityOfCluster(C, range(n))
 
    
    # The size of the largest culture can be determined by finding the
    # vertex label that appears most often
    size_culture = 0
    for i in range(len(vertex_label)):
        if vertex_label.count(i) > size_culture:
            size_culture = vertex_label.count(i)
    
    size_culture = float(size_culture) / n   
    print 'cultures: ', time.time() - first
    
    # Calculate components etc
    if network:
        first = time.time()
        components = G.components()
        num_components = float(len(components)) / n
        
        largest_component = float(max(components.sizes())) / n
        print 'components: ', time.time() - first
        
        if G.ecount() > 0:
            first = time.time()
            communities = G.community_fastgreedy(weights="weight")
#            num_communities = float(len(communities)) / n
            num_communities = float(len(communities.as_clustering())) / n
            largest_community = float(max(communities.as_clustering().sizes())) / n 
            communities = communities.as_clustering()
            # modularity of the community structure 
            community_modularity = G.modularity(communities,
                      weights=[G.es()[i]["weight"] for i in xrange(G.ecount())])
            print 'communities ', time.time() - first
        else:
            communities = [[x] for x in range(n)] 
            num_communities = 1.0
            largest_community = 1.0 / n
            community_modularity = float('nan')
            
        first = time.time()
        # calculate intra-component diversity 
        within_component_diversity, between_component_diversity, component_diversity_ratio, component_size_x_div = calcDiversity(C, components, F, q)
                    
            
        # calculate intra-community diversity
        within_community_diversity, between_community_diversity, community_diversity_ratio, community_size_x_div = calcDiversity(C, communities, F, q)
        print 'inter+intra community diversity: ', time.time() - first
           
        
        if componentClusterWriter != None:
            for tmp in component_size_x_div:
                componentClusterWriter.writerow(pre + tmp)        

        if communityClusterWriter != None:
            for tmp in community_size_x_div:
                communityClusterWriter.writerow(pre + tmp)
        
        if cultureClusterWriter != None:
            accounted_for = set([])
            for i in range(len(vertex_label)):
                if vertex_label[i] not in accounted_for:
                    cultureClusterWriter.writerow(pre + [vertex_label.count(vertex_label[i])])
                    accounted_for.add(vertex_label[i])               

        if ultrametricWriter != None:
            # calculate degree of ultrametricity of the culture vectors
            cophenetic_cc = calc_cophenetic_cc(C)
            ultrametricWriter.writerow(pre + [cophenetic_cc])


        
        # Calculate social clustering of culture as the modularity of the network
        # based on cultural membership
        if G.ecount() > 0:
            first = time.time()
            social_clustering = G.modularity(vertex_label, weights="weight")
            print 'social clustering: ', time.time() - first
        else:
            social_clustering = float('nan')
            
        social_closeness = 0.0
        overall_closeness = 0.0                      
            

        # calculate the graph stats (avg degtree, cluster
        # coefficient, etc.) as above, but with the graph consisting
        # of only edges with weight 1.0 instead of the whole graph
        # G1 is the undirected graph consisting of edges with weight 1.0 in G
        # in R: G1 <- subgraph.edges(G, eids=which(E(G)$weight == 1.0), delete.vertices=FALSE)
        first = time.time()
        G1 = G.subgraph_edges(G.es.select(weight_eq = 1.0),
                              delete_vertices=False)
        print 'get weight 1 subgraph G1: ', time.time() - first
        # Calculate graph metrics
        first = time.time()
        g1_avg_path_length = G1.average_path_length(unconn=False)
        print 'G1 path length: ', time.time() - first

        first = time.time()
        g1_dia = float('nan')
        print 'G1 diameter: ', time.time() - first

        first = time.time()
        g1_avg_degree = mean(G1.degree(range(n)))
        print 'G1 avg degree: ', time.time() - first

        first = time.time()
        g1_cluster_coeff = G1.transitivity_undirected() # global cc
        g1_local_cluster_coeff = G1.transitivity_avglocal_undirected() # avg local cc
        print 'G1 cluster: ', time.time() - first

        first = time.time()
        g1_ass = assortativity(G1)
        print 'G1 assortativity: ', time.time() - first

        first = time.time()
        g1_density = G1.density()
        print 'G1 density: ', time.time() - first


        first = time.time()
        # Calculate correlation between different spaces
        if correlation:
            k = 0
            g1_physical = [0] * (n*(n-1)/2)
            g1_social = [0] * (n*(n-1)/2)
            g1_culture = [0] * (n*(n-1)/2)
            for i in range(n):
                for j in range(i):
                    g1_physical[k] = 1.0 - hackmodel.distance(L[i], L[j], m, toroidal)[1]
                    g1_social[k] = G1.es[G1.get_eid(i,j)]["weight"] if G1.are_connected(i,j) else 0
                    g1_culture[k] = hackmodel.similarity(C[i], C[j])
                    k = k+1

            g1_corr_soc_phy = calcCorrelation(network, social, physical)
            g1_corr_soc_cul = calcCorrelation(network, social, culture)
            g1_corr_phy_cul = calcCorrelation(network, physical, culture)  
        else:
            g1_corr_soc_phy = float('nan')
            g1_corr_soc_cul = float('nan')
            g1_corr_phy_cul = float('nan')
        print 'G1 correlation: ', time.time() - first

        g1_components = G1.components()
        g1_num_components = float(len(g1_components)) / n
        
        g1_largest_component = float(max(g1_components.sizes())) / n
        
        if G1.ecount() > 0:
            g1_communities = G1.community_fastgreedy(weights="weight")
            g1_num_communities = float(len(g1_communities.as_clustering())) / n
            g1_largest_community = float(max(g1_communities.as_clustering().sizes())) / n 
            g1_communities = g1_communities.as_clustering()
            # modularity of the community structure 
            g1_community_modularity = G1.modularity(g1_communities,
                     weights=[G1.es()[i]["weight"] for i in xrange(G1.ecount())])

        else:
            g1_communities = [[x] for x in range(n)] 
            g1_num_communities = 1.0
            g1_largest_community = 1.0 / n
            g1_community_modularity = float('nan')
            
        
        # calculate intra-component diversity 
        g1_within_component_diversity, g1_between_component_diversity, g1_component_diversity_ratio, g1_component_size_x_div = calcDiversity(C, g1_components, F, q)
                    
            
        # calculate intra-community diversity
        g1_within_community_diversity, g1_between_community_diversity, g1_community_diversity_ratio, g1_community_size_x_div = calcDiversity(C, g1_communities, F, q)


        # Calculate social clustering of culture as the modularity of the network
        # based on cultural membership
        if G1.ecount() > 0:
            g1_social_clustering = G1.modularity(vertex_label, weights="weight")
        else:
            g1_social_clustering = float('nan')
        
    else:
        num_components = float('nan')
        largest_component = float('nan')
        num_communities = float('nan')
        largest_community = float('nan')
        within_component_diversity = float('nan')
        within_community_diversity = float('nan')
        between_component_diversity = float('nan')
        between_community_diversity = float('nan')
        component_diversity_ratio = float('nan')
        community_diversity_ratio = float('nan')
        social_clustering = float('nan')
        overall_closeness = float('nan')
        social_closeness = float('nan')
        overall_closeness = float('nan')
        community_modularity = float('nan')

        g1_num_components = float('nan')
        g1_largest_component = float('nan')
        g1_num_communities = float('nan')
        g1_largest_community = float('nan')
        g1_within_component_diversity = float('nan')
        g1_within_community_diversity = float('nan')
        g1_between_component_diversity = float('nan')
        g1_between_community_diversity = float('nan')
        g1_component_diversity_ratio = float('nan')
        g1_community_diversity_ratio = float('nan')
        g1_social_clustering = float('nan')
        g1_community_modularity = float('nan')
        
    print 'communities: ', time.time() - first       
    

    physical_closeness = 0.0  
    
    # indices 0 - 9
    stats = [avg_path_length, dia, avg_degree, cluster_coeff, corr_soc_phy, corr_soc_cul, corr_phy_cul, num_cultures, size_culture, overall_diversity, ass]    
    
    # indices 10 - 12
    stats = stats + [num_components, largest_component, within_component_diversity, between_component_diversity, component_diversity_ratio]
    
    # indices 13 - 15
    stats = stats + [num_communities, largest_community, within_community_diversity, between_community_diversity, community_diversity_ratio]
    
    # indices 16 - 19
    stats = stats + [social_clustering, social_closeness, physical_closeness, overall_closeness]
    
    
    
    physicalStability = 0.0
    socialStability = 0.0
    culturalStability = 0.0
    
    first = time.time()
    if lastG is not None:
        
        for i in range(n):
            physicalStability += 1 - hackmodel.distance(L[i], lastL[i], m, toroidal)[1]
            culturalStability += hackmodel.similarity(C[i], lastC[i])
            
            for j in range(i):        
                socialStability += 1.0 - abs((G.es[G.get_eid(i,j)]["weight"] if G.are_connected(i,j) else 0)
                                       - (lastG.es[lastG.get_eid(i,j)]["weight"] if lastG.are_connected(i,j) else 0))   
        
        physicalStability = physicalStability / n
        culturalStability = culturalStability / n
        socialStability = socialStability / (n*(n-1)/2)    
    print 'stability: ', time.time() - first
                
        
    stats = stats + [physicalStability, socialStability, culturalStability]

    first = time.time()
    num_culture_components = calcCultureNumComponents(C, theta, n)
    print 'culture components:', time.time() - first

    stats += [num_culture_components]


    first = time.time()
    (num_regions, max_region_size) = calcNumAndMaxSizeOfLatticeCulturalRegions(
                                        n, C, L)
    print 'cultural regions:', time.time() - first
    stats += [num_regions, max_region_size]


    first = time.time()
    (mooreNeighbourhoodIdentityDiversity,
     mooreNeighbourhoodImmutableIdentityDiversity,
     mooreNeighbourhoodMutableIdentityDiversity) = (
        calcMooreNeighbourhoodIdentityDiversity(n, m, F, n_immutable, C, L)
        )
     
    print 'Moore neighbourhood identity diversity:', time.time() - first
    stats += (mooreNeighbourhoodIdentityDiversity,
              mooreNeighbourhoodImmutableIdentityDiversity,
              mooreNeighbourhoodMutableIdentityDiversity)


    first = time.time()
    (overallMooreNeighbourhoodDiversity,
     overallMooreNeighbourhoodImmutableDiversity,
     overallMooreNeighbourhoodMutableDiversity) = (
        calcMooreNeighbourhoodDiversity(n, m, F, n_immutable, C, L)
        )
     
    print 'Moore neighbourhood diversity:', time.time() - first
    stats += (overallMooreNeighbourhoodDiversity,
              overallMooreNeighbourhoodImmutableDiversity,
              overallMooreNeighbourhoodMutableDiversity)


    # add graph and community stats for G1 graph (G with only weight=1 edges)
    stats += [g1_avg_path_length, g1_dia, g1_avg_degree, g1_cluster_coeff, g1_corr_soc_phy, g1_corr_soc_cul, g1_corr_phy_cul, g1_ass]    
    
    stats = stats + [g1_num_components, g1_largest_component, g1_within_component_diversity, g1_between_component_diversity, g1_component_diversity_ratio]
    
    stats = stats + [g1_num_communities, g1_largest_community, g1_within_community_diversity, g1_between_community_diversity, g1_community_diversity_ratio]
    
    stats = stats + [g1_social_clustering]
    stats = stats + [density, g1_density]
    stats = stats + [community_modularity, g1_community_modularity]
    stats = stats + [local_cluster_coeff, g1_local_cluster_coeff]

    # Add distribution over cultural differences in every step
    if differences is not None:
        stats += differences
            
    
    if statsWriter != None:
        statsWriter.writerow(pre + stats)

    return stats







# Write a given degree distribution to a file.
def writeHist(dir, fname, d, n):
    f = open(os.path.join(dir, fname + '.hist'), 'w')
    writer = csv.writer(f)
    writer.writerow(d)
    f.close()
    
    
    
    
    
# Write the network, culture and location of agents to a file.    
def writeNetwork(G, L, C, dir, fname):
    # Write adjacency matrix of graph with weights
    G.write(os.path.join(dir , fname + '.adj'), format="adjacency", attribute="weight", default=0.0)
    
    # Write the locations of agents
    f = open(os.path.join(dir, fname + '.L'), 'wb')
    writer = csv.writer(f)
    
    for i in range(len(L)):
        writer.writerow(L[i])
    
    f.close()
    
    # Write the culture of agents
    f = open(os.path.join(dir, fname + '.C'), 'wb')
    writer = csv.writer(f)
    
    for i in range(len(C)):
        writer.writerow(C[i])
    
    f.close()
    
    # Write the present cultures
    f = open(os.path.join(dir, fname + '.S'), 'wb')
    writer = csv.writer(f)

    i = 0
    for c in cultures_in(C, range(G.vcount()))[2]:
        writer.writerow([i, c])
        i = i+1
    f.close()    
    

# Determines the svn revisions for all files in this working copy
# and write this information to the provided configuration file.   
def findRevisions(config, path):
    config.add_section('revisions')
    for file in [x for x in index(path) if 'svn' not in x and 'pyc' not in x]:
        #out = commands.getoutput('svn info ' + file)
        #revMatch = re.search('Revision: ([0-9.-]*)', out)
        out = commands.getoutput('cd ' + path + '; git log --oneline ' + file + '|head -1')
        revMatch = re.search('^([0-9a-fA-F]+)', out)
        fileMatch = re.search('[\S]*(\/axelrod\/[\S]*)', file) 
        if revMatch != None:
            config.set('revisions', fileMatch.group(1), revMatch.group(1))
        else:
            config.set('revisions', fileMatch.group(1), 'n/a')
 
 
    
# Lists all files in the directory recursively.  
def index(directory):
    # like os.listdir, but traverses directory trees
    stack = [directory]
    files = []
    while stack:
        directory = stack.pop()
        for file in os.listdir(directory):
            fullname = os.path.join(directory, file)
            files.append(fullname)
            if os.path.isdir(fullname) and not os.path.islink(fullname):
                stack.append(fullname)
    return files


# Determines the diff between the working copy and the latest svn revision
# for all files in this project.
def findDiffs(path, outpath):
    out = commands.getoutput('svn diff ' + path)
    file = open(outpath + 'svn.diff', 'w')
    file.write(out)
    file.close()



# Loads agents' locations, cultures, and the social network from files in
# the directory given by the argument path.
def loadNetwork(path, network = True, end = False):   
    L = list()
    reader = csv.reader(open(path + '.L'))                
    for row in reader:
        L.append((int(row[0]), int(row[1])))
    
    C = list()
    reader = csv.reader(open(path + '.C'))          
    for row in reader:
        C.append(array([int(x) for x in row]))
        
    if end:    
        infile = open(path + '.Tend', "r")
        tend = infile.readline()
    else:
        tend = 0
        
    if network:
        G = Graph.Read_Adjacency(path + '.adj', attribute="weight", mode=ADJ_UNDIRECTED)
    else:
        G = Graph(len(C))        
    if G.ecount() == 0:
        G.es["weight"] = []        
        
    
    if os.path.exists(path + '.D') and os.path.isfile(path + '.D'):
        reader = csv.reader(open(path + '.D'))
        for row in reader:
            D = [int(x) for x in row]

    else:
        D = None
        
    
    return G, L, C, D, tend
        
    
    
# Runs the C++ version of the model.
def modelCpp(G,L,C,tmax,n,m,F,q,r,s,toroidal,network,t,directedMigration,
             phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b, modelpath,
             r_2, s_2, phy_mob_a_2, phy_mob_b_2, soc_mob_a_2, soc_mob_b_2, modelCallback = None, noise = -1, end = False, k = 0.01, theta = 0.0,
             n_immutable = 0,
             no_migration = False, radius = -1):
    
    if toroidal: 
        toroidalS = '1'
    else:
        toroidalS = '0'
        
    if network: 
        networkS = '1'
    else:
        networkS = '0'         
        
    if directedMigration: 
        directedMigrationS = '1'
    else:
        directedMigrationS = '0'

    tmpdir = os.tempnam(None, 'exp')
    os.mkdir(tmpdir)
    writeNetwork(G, L, C, tmpdir, 'tmp')
    
    # If all cells are occupied, there cannot need to be any migration
    if n == m*m:
        s = 0.0

    if no_migration:  # if no_migration is set, disable migration 
        s = 0.0
    
    options = [tmax,n,m,F,q,r,s,toroidalS,networkS,t,directedMigrationS,
               phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b, 
               r_2, s_2, phy_mob_a_2, phy_mob_b_2, soc_mob_a_2, soc_mob_b_2, k,
               tmpdir + '/' + 'tmp', theta, n_immutable]
    

    if radius != -1:
        options.append(radius)

    if noise != -1:
        options.append(noise)
        
    
    if modelCallback is not None:
        f = open(tmpdir + '/tmp.T', 'wb')
        writer = csv.writer(f)
        writer.writerow([len(modelCallback.call_list)])
        writer.writerow(modelCallback.call_list)        
        f.close()        
        #options += modelCallback.call_list            
    
    first = time.time()
    output = commands.getoutput(modelpath + ' ' + ' '.join([str(x) for x in options]))
    #XXX output = commands.getoutput(modelpath + ' ' + ' '.join([str(x) for x in options]) + ' | tee -a model_stdout.txt')
    print output
    print 'model: ', time.time() - first
    
    if modelCallback is not None:
        for iteration in modelCallback.call_list:
            if iteration != tmax:
                G2, L2, C2, D2, tmp = loadNetwork(tmpdir + '/tmp-' + str(iteration), network, end = False)
                if D2 is not None:
                    modelCallback.call(G2, L2, C2, iteration, D2)
                else:
                    modelCallback.call(G2, L2, C2, iteration)
    
    G2, L2, C2, D2, tend = loadNetwork(tmpdir + '/tmp', network, end = end)    
    
    for filename in glob.glob(os.path.join(tmpdir, "*")):
        os.remove(filename)
    os.rmdir(tmpdir)
    
    if D2 is not None:
        return G2, L2, C2, D2
    elif end:
        return G2, L2, C2, tend
    else:
        return G2, L2, C2, tmax
    
    
    

        

#    # Run single runs for drawing networks
#    for n in n_list:
#        G = Graph(n)     
#    
#        # Set random positions of agents
#        L = list()
#        for i in range(n):
#            while True:
#                x = (randrange(m), randrange(m))
#                if x not in L:
#                    L.append(x)
#                    break
#       
#        G.es["weight"] = []
#          
#        for (r,s) in rs_list:         
#            for t in t_list:                 
#                for q in q_list:
#                    # Create culture vector for all agents
#                    C = [array([randrange(q) for k in range(F)]) for l in range(n)]
#                    
#                    # Run the model                  
#                    if cppmodel is not None:
#                        G2, L2, C2 = modelCpp(G.copy(),list(L),C,tmax,n,m,F,q,r,s,toroidal,network,t,
#                                              directedMigration,phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b, cppmodel)                                                             
#                    else:     
#                        G2, L2, C2 = model.model(G.copy(),list(L),C,tmax,m,r,s,toroidal,network,
#                                               phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b,t,directedMigration)                    
#                                   
#                    
#                    # Print network and write out statistics
#                    file = 'results-n' + str(n) + '-r' + str(r) + '-s' + str(s) + '-t' + str(t) + '-q' + str(q)
#                    writeNetwork(G2,L2,C2,dir + str(n) + '/', file)
     
            
        
