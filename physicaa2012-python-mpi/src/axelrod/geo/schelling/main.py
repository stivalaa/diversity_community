#  Copyright (C) 2011 Jens Pfau <jpfau@unimelb.edu.au>
#  Copyright (C) 2014 Alex Stivala <stivalaa@unimelb.edu.au>
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

# This verison is from running models base on the Neal & Neal (2014)
# Schelling segragation model with social networks e.g.
# lattice-schelling-network/model
# This version does
# multiple runs from same initialization.

# reusing theta to mean desired level of similarity in this model

import warnings # so we can suppress the annoying tempnam 'security' warning

import sys
import os,errno
sys.path.append( os.path.abspath(os.path.dirname(sys.argv[0])).replace('/schelling','') )
#sys.path.append( os.path.abspath(os.path.dirname(sys.argv[0])).replace('geo/schelling','geo') )

from numpy import array
from igraph import Graph
from random import randrange,sample,random
import math
import glob
from time import strftime,localtime,time

import csv
#import model

import lib
import ConfigParser
import commands

from mpi4py import MPI

import neutralevolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_processes = comm.Get_size()
    
# Runs the C++ version of the model.
def modelCpp(G,L,C,tmax,n,m,F,q,
             modelpath,
             modelCallback, moore_radius, sigma):

    
    tmpdir = os.tempnam(None, 'exp')
    os.mkdir(tmpdir)
    lib.writeNetwork(G, L, C, tmpdir, 'tmp')
    
    options = [tmax,n,m,F,q,
               tmpdir + '/' + 'tmp', moore_radius, sigma]
    
    if modelCallback is not None:
        f = open(tmpdir + '/tmp.T', 'wb')
        writer = csv.writer(f)
        writer.writerow([len(modelCallback.call_list)])
        writer.writerow(modelCallback.call_list)        
        f.close()        
        #options += modelCallback.call_list            
    
    first = time()
    output = commands.getoutput(modelpath + ' ' + ' '.join([str(x) for x in options]))
    #XXX output = commands.getoutput(modelpath + ' ' + ' '.join([str(x) for x in options]) + ' | tee -a model_stdout.txt')
    print output
    print 'model: ', time() - first
    
    if modelCallback is not None:
        for iteration in modelCallback.call_list:
            if iteration != tmax:
                G2, L2, C2, D2, tmp = lib.loadNetwork(tmpdir + '/tmp-' + str(iteration), network, end = False)
                if D2 is not None:
                    modelCallback.call(G2, L2, C2, iteration, D2)
                else:
                    modelCallback.call(G2, L2, C2, iteration)
    
    G2, L2, C2, D2, tend = lib.loadNetwork(tmpdir + '/tmp', network = True, end = end)    
    
    for filename in glob.glob(os.path.join(tmpdir, "*")):
        os.remove(filename)
    os.rmdir(tmpdir)
    
    if D2 is not None:
        return G2, L2, C2, D2
    elif end:
        return G2, L2, C2, tend
    else:
        return G2, L2, C2, tmax


def writeConfig(scriptpath, runs, dir, tmax, F, m, toroidal, network, 
             beta_p_list, beta_s_list, 
             n_list, q_list, 
             directedMigration, cppmodel = None, time_list = None, 
             theta_list = None, evolved_init_culture = False,
             evolved_init_culture2 = False,
             prototype_init_culture = False,
             trivial_ultrametric_init_culture = False,
             init_culture_csv_file = None,
             init_random_prob_list = None,
             resumeFrom = None, 
             no_migration = False,
             cluster_k =None,
             ef2_r = None,
             moore_radius = None
             ):
    # Write out current configuration       
    for n in n_list:
        try:
            os.mkdir(dir + str(n))
        except OSError,e:
            if e.errno == errno.EEXIST:
                pass
            
        # Save the diff between the current working copy and the latest svn revision
        lib.findDiffs(scriptpath, dir + str(n) + '/')

        # Write out the parameters for this experiment
        config = ConfigParser.RawConfigParser()

        config.add_section('environment')
        config.set('environment', 'starttime', strftime("%d%b%Y-%H:%M:%S", localtime()))
        config.set('environment', 'mpiname' ,comm.Get_name())
        config.set('environment', 'mpiprocessorname' ,MPI.Get_processor_name())
        config.set('environment', 'mpiprocesses' ,mpi_processes)
        config.set('environment', 'mpirank', rank)

        # Determine the svn revisions of all files and write them into the config file
        lib.findRevisions(config, scriptpath)

        # Write out all parameters
        config.add_section('paras')
        if cppmodel is not None:
            config.set('paras', 'cpp', commands.getoutput(cppmodel + ' -v'))
        else:
            config.set('paras', 'cpp', False)
        config.set('paras', 'runs', str(runs))
        config.set('paras', 'tmax', str(tmax))
        config.set('paras', 'F', str(F))
        config.set('paras', 'm', str(m))
        config.set('paras', 'toroidal', str(toroidal))
        config.set('paras', 'network', str(network))
        config.set('paras', 'beta_p_list', ','.join([str(x) for x in beta_p_list]))
        config.set('paras', 'beta_s_list', ','.join([str(x) for x in beta_s_list]))        
        config.set('paras', 'n', str(n))
        config.set('paras', 'directed_migration', str(directedMigration))
        config.set('paras', 'q_list', ','.join([str(x) for x in q_list]))
        config.set('paras', 'theta_list', ','.join([str(x) for x in theta_list]))
        if time_list is not None:
            config.set('paras', 'time_list', ','.join([str(x) for x in time_list]))
        config.set('paras', 'evolved_init_culture', str(evolved_init_culture))
        config.set('paras', 'evolved_init_culture2', str(evolved_init_culture2))
        config.set('paras', 'prototype_init_culture', str(prototype_init_culture))
        config.set('paras', 'trivial_ultrametric_init_culture', str(trivial_ultrametric_init_culture))
        if prototype_init_culture:
            config.set('paras', 'cluster_k', str(cluster_k))
        if evolved_init_culture2:
            config.set('paras', 'ef2_r', str(ef2_r))
        if evolved_init_culture or evolved_init_culture2 or prototype_init_culture:
            config.set('paras', 'init_random_prob_list', ','.join([str(x) for x in init_random_prob_list]))
        if init_culture_csv_file is not None:
            config.set('paras', 'read_init_culture', init_culture_csv_file)
        if resumeFrom != None:
            config.set('paras', 'resumeFrom', str(resumeFrom))
            writemode = 'ab'  # append to config file
        else:
            writemode = 'wb'  # overwrite config file
        config.set('paras', 'no_migration', str(no_migration));
        config.set('paras', 'moore_radius', str(moore_radius))
        config.write(open(dir + str(n) + '/' + 'parameter' + str(rank) + '.cfg', writemode))   





class ModelCallback:
    def __init__(self, graph_list, statsWriter, F, phy_mob_a, phy_mob_b, 
                 soc_mob_a, soc_mob_b, r, s, t, q, theta, init_random_prob,
                 n_immutable, i,
                 toroidal,network,
                 componentClusterWriter, communityClusterWriter, cultureClusterWriter, ultrametricWriter, moore_radius):
        self.call_list = graph_list
        self.statsWriter = statsWriter
        self.F = F
        self.phy_mob_a = phy_mob_a
        self.phy_mob_b = phy_mob_b
        self.soc_mob_a = soc_mob_a
        self.soc_mob_b = soc_mob_b
        self.r = r
        self.s = s
        self.t = t
        self.q = q
        self.theta = theta
        self.init_random_prob = init_random_prob
        self.n_immutable = n_immutable
        self.i = i
        self.toroidal = toroidal
        self.network = network
        self.lastG = None
        self.lastL = None
        self.lastC = None
        self.componentClusterWriter = componentClusterWriter
        self.communityClusterWriter = communityClusterWriter
        self.cultureClusterWriter = cultureClusterWriter
        self.ultrametricWriter = ultrametricWriter
        self.moore_radius = moore_radius
        
        
    def call(self, G, L, C, iteration):                        
        # Get statistics for this run
        lib.writeStatistics(self.statsWriter, self.F, self.phy_mob_a, 
                              self.phy_mob_b, self.soc_mob_a, self.soc_mob_b,
                              self.r,self.s,self.t,self.q,self.theta,
                              self.init_random_prob,
                              self.n_immutable,
                              self.i,G,L,C,m,
                              self.toroidal,
                              self.network,iteration,self.lastG, self.lastL, self.lastC, 
                              componentClusterWriter = self.componentClusterWriter, 
                              communityClusterWriter = self.communityClusterWriter, 
                              cultureClusterWriter= self.cultureClusterWriter,
                              ultrametricWriter = self.ultrametricWriter,
                              correlation = False)
        
        self.lastG = G.copy()
        self.lastL = list(L)
        
        self.lastC = list()
        for c in C:
            self.lastC.append(c.copy())



def scenario(scriptpath, runs, dir, tmax, F, m, toroidal, network, 
             beta_p_list, beta_s_list, 
             n_list, q_list, time_list,
             directedMigration=True, cppmodel = None, end = False,
             evolved_init_culture = False, 
             evolved_init_culture2 = False,
             prototype_init_culture = False,
             trivial_ultrametric_init_culture = False,
             sigma_list = None,
             initial_culture = None,
             init_culture_csv_file = None,
             init_random_prob_list = None,
             cluster_k = None,
             ef2_r = None,
             moore_radius = None,
             resumeFrom = None):   

    assert not (initial_culture != None and evolved_init_culture)
    assert not (initial_culture != None and evolved_init_culture2)
    assert not (initial_culture != None and prototype_init_culture)
    assert not (initial_culture != None and trivial_ultrametric_init_culture)
    assert not (evolved_init_culture and evolved_init_culture2)
    assert not (evolved_init_culture and prototype_init_culture)
    assert not (evolved_init_culture2 and prototype_init_culture)
    assert not (evolved_init_culture and trivial_ultrametric_init_culture)
    assert not (evolved_init_culture2 and trivial_ultrametric_init_culture)
    assert not (prototype_init_culture and trivial_ultrametric_init_culture)
    assert not (not prototype_init_culture and cluster_k != None)
    assert not (prototype_init_culture and cluster_k == None)
    assert not (not evolved_init_culture2 and ef2_r != None)
    assert not (evolved_init_culture2 and ef2_r == None)
    # TODO probably should have had an integer init_culture_type or something
    # instead of now having all these booleans...
    if not (evolved_init_culture or evolved_init_culture2 or prototype_init_culture or trivial_ultrametric_init_culture):
        init_random_prob_list = [None] # not used if not evolved_init_culture
    
    if cppmodel is not None:
        print "Using c++ version with ", cppmodel
    else:
        print "Using python version"
        
    if end:
        time_list = []
    
    try:
        os.mkdir(dir)       
    except OSError,e:
        if e.errno == errno.EEXIST:
            pass
      
  
    writeConfig(scriptpath, runs, dir, tmax, F, m, toroidal, network, 
             beta_p_list, beta_s_list,
             n_list, q_list, 
             directedMigration, cppmodel, time_list, sigma_list,
             evolved_init_culture, evolved_init_culture2,
             prototype_init_culture,
             trivial_ultrametric_init_culture,
             init_culture_csv_file,
             init_random_prob_list, resumeFrom, no_migration, cluster_k, ef2_r,
             moore_radius)
    
        

    statsWriter = {}  # dict indexec by n
    componentClusterWriter = {} # dict indexed by n
    communityClusterWriter = {} # dict indexed by n
    cultureClusterWriter = {} # dict indexed by n
    ultrametricWriter = {} # dict indexed by n

    param_list = [] # list of parameter tuples: build list then run each
    param_count = 0
    for n in n_list:
        # Create the file into which statistics are written
        results_prefix = '/results' + str(rank)
        resultsfilename = dir + str(n) + results_prefix + '.csv'
        if resumeFrom != None:
            print 'Resuming from ', resumeFrom
            print 'Appending results to ', resultsfilename
            csvwritemode = 'a'  # append if resuming 
        else:
            print 'Clean start'
            print 'Writing results to ', resultsfilename
            csvwritemode = 'w'  # overwite if staring from beginning
        file = open(resultsfilename, csvwritemode)
        statsWriter[n] = csv.writer(file)
        
        componentClusterWriterFile= open(dir + str(n) + results_prefix + '-size_x_div-components.csv', csvwritemode)
        communityClusterWriterFile= open(dir + str(n) + results_prefix + '-size_x_div-communities.csv', csvwritemode)
        cultureClusterWriterFile = open(dir + str(n) + results_prefix +'-cultures.csv', csvwritemode)
        ultrametricWriterFile = open(dir + str(n) + results_prefix + '-ultrametricity.csv', csvwritemode)

        componentClusterWriter[n] = csv.writer(componentClusterWriterFile)
        communityClusterWriter[n] = csv.writer(communityClusterWriterFile)
        cultureClusterWriter[n] = csv.writer(cultureClusterWriterFile)
        ultrametricWriter[n] = csv.writer(ultrametricWriterFile)
            
        for q in q_list:
            for beta_p in beta_p_list:
                for beta_s in beta_s_list:              
                    for sigma in sigma_list:
                        for init_random_prob in init_random_prob_list:
                            param_count += 1
                            if resumeFrom is not None:
                                if n == resumeFrom[0] and q == resumeFrom[1] and  beta_p == resumeFrom[2] and beta_s == resumeFrom[3] and sigma == resumeFrom[4] and init_random_prob == resumeFrom[5]:
                                    resumeFrom = None
                                else:
                                    continue                               

                            param_list.append((n, q, beta_p, beta_s, sigma, init_random_prob))


    print len(param_list)*runs,'of total',param_count*runs,'models to run'
    print int(math.ceil(float(len(param_list)*runs)/mpi_processes)),' models per MPI task'

    
    # now that we have list of parameter tuples, execute in parallel
    # using MPI: each MPI process will process 
    # ceil(num_jobs / mpi_processes) of the parmeter tuples in the list
    num_jobs = len(param_list)
    job_i = rank
    while job_i < num_jobs:
        start_time = time()
        (n, q, beta_p, beta_s, sigma, init_random_prob) = param_list[job_i]
        # Create the graph
        G = Graph(n)
        G.es["weight"] = []
        
        #print rank, n, q, beta_p, beta_s, theta, init_random_prob, i 
        # In this format, the tuple after 'rank 0: '
        # can be cut&pasted into resumeFrom: command line
        sys.stdout.write('rank %d: %d,%d,%f,%f,%f,%s,%d\n' %
                         (rank,n,q,beta_p,beta_s,sigma,str(init_random_prob),0))

        assert(n <= m**2) # lattice must have enough positions for agents

        first = time()
        # Set random positions of agents
        startL = [(x,y) for x in range(m) for y in range(m)]
        L = list()
        for j in range(n):
            idx = randrange(len(startL))
            L.append(startL[idx])
            del startL[idx]
        print 'init positions: ', time() - first

        first = time()
        # Create culture vector for all agents
        if initial_culture:
            C = initial_culture
            assert(n == len(C))
            assert(F == len(C[0]))
            
        elif evolved_init_culture:
          # 'evolve' culture vectors, then perturb each element
          # with probability init_random_prob, picking a random value for
          # that element,
          initialC = [randrange(q) for k in range(F)]
          C = neutralevolution.ef(initialC, q, [],
                                  int(math.ceil(math.log(n,2))),
                                  1.0, 1.0)[:n]
          C = [array(v) for v in neutralevolution.perturb(array(C), q, init_random_prob).tolist()]  # have to convert array to list and then each inner list back to array to from 2d array for pertrub() to list of 1d arrays

        elif evolved_init_culture2:
          # 'evolve' culture vectors (with ef2 function not original ef),
          # mutating ef2_r  traits (equiv to ef2() if r=1,
          # efault original ef2() if r=F/2) each step (not one trait)
          # then perturb each element
          # with probability init_random_prob, picking a random value for
          # that element,
          initialC = [randrange(q) for k in range(F)]
          C = neutralevolution.ef2(initialC, q, [],
                                  int(math.ceil(math.log(n,2))),
                                  ef2_r)[:n]
          C = [array(v) for v in neutralevolution.perturb(array(C), q, init_random_prob).tolist()]  # have to convert array to list and then each inner list back to array to from 2d array for pertrub() to list of 1d arrays

        elif prototype_init_culture:
          # 'evolve' culture vectors by 'reverse k-means' generating k
          # prototype vectors then randomly createding other near htme by
          # mutating 1/2 of all traits each step ,
          # then perturb each element
          # with probability init_random_prob, picking a random value for
          # that element,
          initialC = [randrange(q) for k in range(F)]
          C = neutralevolution.prototype_evolve(F, q, n, cluster_k, F/2)
          C = [array(v) for v in neutralevolution.perturb(array(C), q, init_random_prob).tolist()]  # have to convert array to list and then each inner list back to array to from 2d array for pertrub() to list of 1d arrays

        elif trivial_ultrametric_init_culture:
          # create ultrametric init culture that is perfectly ultrametric
          # but trivial in that all vectors have the same hamming distance
          C = neutralevolution.trivial_ultrametric(F, q, n)
          C = [array(v) for v in neutralevolution.perturb(array(C), q, init_random_prob).tolist()]  # have to convert array to list and then each inner list back to array to from 2d array for pertrub() to list of 1d arrays
          
        else: # uniform random culture vectors
          C = [array([randrange(q) for k in range(F)]) for l in range(n)]

        # write initial conditions to a file
        filename = 'results' + str(rank) + '-n' + str(n) + '-q' + str(q) + '-beta_p' + str(beta_p) + '-beta_s' + str(beta_s) + '-theta' + str(sigma) + '-init_random_prob' + str(init_random_prob)
        lib.writeNetwork(G,L,C, dir+str(n), filename + '-' + str(0))
        print 'init cultures: ', time() - first

        for i in range(runs):
            sys.stdout.write('rank %d run %d\n' % (rank, i))

            graphCallback = ModelCallback(time_list, statsWriter[n], F, 
                1, beta_p, 1, beta_s, 
                1, 1, -1, q, sigma, init_random_prob, F, i, toroidal, network,
                componentClusterWriter[n], communityClusterWriter[n], cultureClusterWriter[n], ultrametricWriter[n], moore_radius)

            # write initial stats to stats files
            # (original version did not do this, now have to account for 
            # iteratino 0 in results for for end (run to equilibrium) runs,
            # rather than just final iteration number when equilibrium reached)
            if end: # in time (not equilibrium) mode, already writes at iter=0
                first = time()
                graphCallback.call(G, L, C, 0)
                print 'initial graphCallback:', time() - first

            print 'setup time: ',time() - start_time
            # Run the model
            G2, L2, C2, tend = modelCpp(G.copy(),L,C,tmax,n,m,F,q,
                                  cppmodel,
                                  graphCallback, 
                                  moore_radius, sigma)

            process_time_start = time()
            first = time()
            if end:
                graphCallback.call(G2, L2, C2, tend)
                iterno = tend
            else:
                graphCallback.call(G2, L2, C2, tmax)
                iterno = tmax
            print 'post modelCallback: ', time() - first


            file.flush()
            componentClusterWriterFile.flush()
            communityClusterWriterFile.flush()
            cultureClusterWriterFile.flush()
            ultrametricWriterFile.flush()


            if i == 0:
              # write final conditions to a file
              filename = 'results' + str(rank) + '-n' + str(n) + '-q' + str(q) + '-beta_p' + str(beta_p) + '-beta_s' + str(beta_s) + '-theta' + str(sigma) + '-init_random_prob' + str(init_random_prob)
              lib.writeNetwork(G2,L2,C2, dir+str(n), filename + '-' + str(iterno))
        print 'post process time: ', time() - process_time_start
        job_i += mpi_processes

    file.close()
    componentClusterWriterFile.close()
    communityClusterWriterFile.close()
    cultureClusterWriterFile.close()
    ultrametricWriterFile.close()


if __name__ == '__main__':   

    # tmpdir() annoyingly gives 'security' warning on stderr, as does
    # tmpnam(), unless we add these filterwarnings() calls.
    warnings.filterwarnings('ignore', 'tempdir', RuntimeWarning)
    warnings.filterwarnings('ignore', 'tempnam', RuntimeWarning)

    try:
        import psyco
        #psyco.log()
        psyco.full()
    except ImportError:
        print "Psyco not installed or failed execution."    

    # Find the path to the source code
    scriptpath = os.path.abspath(os.path.dirname(sys.argv[0])).replace('/geo/schelling', '/')     
  
    #q_list =  [2]
    q_list =  [2,3,4,5]
  
    beta_p_list = [10]
    beta_s_list = [1]
   
    init_random_prob_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
 
    runs = 50
    

    tmax = 100000000
    time_list = [0, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 2000000, 5000000, 10000000, 50000000, 100000000]

    m = 25
    F = 1

    moore_radius = 1

    # reusing theta to mean sigma, mean overlap (tolerance) threshold in this model
#    sigma_list = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5]
#    sigma_list = [0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.029,0.03,0.031,0.032,0.033,0.034,0.035,0.036,0.037,0.038,0.039,0.04,0.041,0.042,0.043,0.044,0.045,0.046,0.047,0.048,0.049,0.05,0.051,0.052,0.053,0.054,0.055,0.056,0.057,0.058,0.059,0.06,0.061,0.062,0.063,0.064,0.065,0.066,0.067,0.068,0.069,0.07,0.071,0.072,0.073,0.074,0.075,0.076,0.077,0.078,0.079,0.08,0.081,0.082,0.083,0.084,0.085,0.086,0.087,0.088,0.089,0.09,0.091,0.092,0.093,0.094,0.095,0.096,0.097,0.098,0.099,0.1,0.101,0.102,0.103,0.104,0.105,0.106,0.107,0.108,0.109,0.11,0.111,0.112,0.113,0.114,0.115,0.116,0.117,0.118,0.119,0.12,0.121,0.122,0.123,0.124,0.125,0.126,0.127,0.128,0.129,0.13,0.131,0.132,0.133,0.134,0.135,0.136,0.137,0.138,0.139,0.14,0.141,0.142,0.143,0.144,0.145,0.146,0.147,0.148,0.149,0.15,0.151,0.152,0.153,0.154,0.155,0.156,0.157,0.158,0.159,0.16,0.161,0.162,0.163,0.164,0.165,0.166,0.167,0.168,0.169,0.17,0.171,0.172,0.173,0.174,0.175,0.176,0.177,0.178,0.179,0.18,0.181,0.182,0.183,0.184,0.185,0.186,0.187,0.188,0.189,0.19,0.191,0.192,0.193,0.194,0.195,0.196,0.197,0.198,0.199,0.2,0.201,0.202,0.203,0.204,0.205,0.206,0.207,0.208,0.209,0.21,0.211,0.212,0.213,0.214,0.215,0.216,0.217,0.218,0.219,0.22,0.221,0.222,0.223,0.224,0.225,0.226,0.227,0.228,0.229,0.23,0.231,0.232,0.233,0.234,0.235,0.236,0.237,0.238,0.239,0.24,0.241,0.242,0.243,0.244,0.245,0.246,0.247,0.248,0.249,0.25,0.251,0.252,0.253,0.254,0.255,0.256,0.257,0.258,0.259,0.26,0.261,0.262,0.263,0.264,0.265,0.266,0.267,0.268,0.269,0.27,0.271,0.272,0.273,0.274,0.275,0.276,0.277,0.278,0.279,0.28,0.281,0.282,0.283,0.284,0.285,0.286,0.287,0.288,0.289,0.29,0.291,0.292,0.293,0.294,0.295,0.296,0.297,0.298,0.299,0.3,0.301,0.302,0.303,0.304,0.305,0.306,0.307,0.308,0.309,0.31,0.311,0.312,0.313,0.314,0.315,0.316,0.317,0.318,0.319,0.32,0.321,0.322,0.323,0.324,0.325,0.326,0.327,0.328,0.329,0.33,0.331,0.332,0.333,0.334,0.335,0.336,0.337,0.338,0.339,0.34,0.341,0.342,0.343,0.344,0.345,0.346,0.347,0.348,0.349,0.35,0.351,0.352,0.353,0.354,0.355,0.356,0.357,0.358,0.359,0.36,0.361,0.362,0.363,0.364,0.365,0.366,0.367,0.368,0.369,0.37,0.371,0.372,0.373,0.374,0.375,0.376,0.377,0.378,0.379,0.38,0.381,0.382,0.383,0.384,0.385,0.386,0.387,0.388,0.389,0.39,0.391,0.392,0.393,0.394,0.395,0.396,0.397,0.398,0.399,0.4,0.401,0.402,0.403,0.404,0.405,0.406,0.407,0.408,0.409,0.41,0.411,0.412,0.413,0.414,0.415,0.416,0.417,0.418,0.419,0.42,0.421,0.422,0.423,0.424,0.425,0.426,0.427,0.428,0.429,0.43,0.431,0.432,0.433,0.434,0.435,0.436,0.437,0.438,0.439,0.44,0.441,0.442,0.443,0.444,0.445,0.446,0.447,0.448,0.449,0.45,0.451,0.452,0.453,0.454,0.455,0.456,0.457,0.458,0.459,0.46,0.461,0.462,0.463,0.464,0.465,0.466,0.467,0.468,0.469,0.47,0.471,0.472,0.473,0.474,0.475,0.476,0.477,0.478,0.479,0.48,0.481,0.482,0.483,0.484,0.485,0.486,0.487,0.488,0.489,0.49,0.491,0.492,0.493,0.494,0.495,0.496,0.497,0.498,0.499,0.5] # 501 elements generated from R with  cat(seq(0,0.5,0.001), sep=',')
    sigma_list = [0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.024,0.026,0.028,0.03,0.032,0.034,0.036,0.038,0.04,0.042,0.044,0.046,0.048,0.05,0.052,0.054,0.056,0.058,0.06,0.062,0.064,0.066,0.068,0.07,0.072,0.074,0.076,0.078,0.08,0.082,0.084,0.086,0.088,0.09,0.092,0.094,0.096,0.098,0.1,0.102,0.104,0.106,0.108,0.11,0.112,0.114,0.116,0.118,0.12,0.122,0.124,0.126,0.128,0.13,0.132,0.134,0.136,0.138,0.14,0.142,0.144,0.146,0.148,0.15,0.152,0.154,0.156,0.158,0.16,0.162,0.164,0.166,0.168,0.17,0.172,0.174,0.176,0.178,0.18,0.182,0.184,0.186,0.188,0.19,0.192,0.194,0.196,0.198,0.2,0.202,0.204,0.206,0.208,0.21,0.212,0.214,0.216,0.218,0.22,0.222,0.224,0.226,0.228,0.23,0.232,0.234,0.236,0.238,0.24,0.242,0.244,0.246,0.248,0.25,0.252,0.254,0.256,0.258,0.26,0.262,0.264,0.266,0.268,0.27,0.272,0.274,0.276,0.278,0.28,0.282,0.284,0.286,0.288,0.29,0.292,0.294,0.296,0.298,0.3,0.302,0.304,0.306,0.308,0.31,0.312,0.314,0.316,0.318,0.32,0.322,0.324,0.326,0.328,0.33,0.332,0.334,0.336,0.338,0.34,0.342,0.344,0.346,0.348,0.35,0.352,0.354,0.356,0.358,0.36,0.362,0.364,0.366,0.368,0.37,0.372,0.374,0.376,0.378,0.38,0.382,0.384,0.386,0.388,0.39,0.392,0.394,0.396,0.398,0.4,0.402,0.404,0.406,0.408,0.41,0.412,0.414,0.416,0.418,0.42,0.422,0.424,0.426,0.428,0.43,0.432,0.434,0.436,0.438,0.44,0.442,0.444,0.446,0.448,0.45,0.452,0.454,0.456,0.458,0.46,0.462,0.464,0.466,0.468,0.47,0.472,0.474,0.476,0.478,0.48,0.482,0.484,0.486,0.488,0.49,0.492,0.494,0.496,0.498,0.5,0.502,0.504,0.506,0.508,0.51,0.512,0.514,0.516,0.518,0.52,0.522,0.524,0.526,0.528,0.53,0.532,0.534,0.536,0.538,0.54,0.542,0.544,0.546,0.548,0.55,0.552,0.554,0.556,0.558,0.56,0.562,0.564,0.566,0.568,0.57,0.572,0.574,0.576,0.578,0.58,0.582,0.584,0.586,0.588,0.59,0.592,0.594,0.596,0.598,0.6,0.602,0.604,0.606,0.608,0.61,0.612,0.614,0.616,0.618,0.62,0.622,0.624,0.626,0.628,0.63,0.632,0.634,0.636,0.638,0.64,0.642,0.644,0.646,0.648,0.65,0.652,0.654,0.656,0.658,0.66,0.662,0.664,0.666,0.668,0.67,0.672,0.674,0.676,0.678,0.68,0.682,0.684,0.686,0.688,0.69,0.692,0.694,0.696,0.698,0.7,0.702,0.704,0.706,0.708,0.71,0.712,0.714,0.716,0.718,0.72,0.722,0.724,0.726,0.728,0.73,0.732,0.734,0.736,0.738,0.74,0.742,0.744,0.746,0.748,0.75,0.752,0.754,0.756,0.758,0.76,0.762,0.764,0.766,0.768,0.77,0.772,0.774,0.776,0.778,0.78,0.782,0.784,0.786,0.788,0.79,0.792,0.794,0.796,0.798,0.8,0.802,0.804,0.806,0.808,0.81,0.812,0.814,0.816,0.818,0.82,0.822,0.824,0.826,0.828,0.83,0.832,0.834,0.836,0.838,0.84,0.842,0.844,0.846,0.848,0.85,0.852,0.854,0.856,0.858,0.86,0.862,0.864,0.866,0.868,0.87,0.872,0.874,0.876,0.878,0.88,0.882,0.884,0.886,0.888,0.89,0.892,0.894,0.896,0.898,0.9,0.902,0.904,0.906,0.908,0.91,0.912,0.914,0.916,0.918,0.92,0.922,0.924,0.926,0.928,0.93,0.932,0.934,0.936,0.938,0.94,0.942,0.944,0.946,0.948,0.95,0.952,0.954,0.956,0.958,0.96,0.962,0.964,0.966,0.968,0.97,0.972,0.974,0.976,0.978,0.98,0.982,0.984,0.986,0.988,0.99,0.992,0.994,0.996,0.998,1] # 501 elements generated from R with  cat(seq(0,1.0,0.002), sep=',')
                           


    
    
    cppmodel = None
    
    n_list = []
    
    end = False
    
    evolved_init_culture = False
    evolved_init_culture2 = False
    prototype_init_culture = False
    trivial_ultrametric_init_culture = False
    init_culture_csv_file = None
    initial_culture= None
    resumeFrom = None
    no_migration = False # default is to have migration
    cluster_k = None
    ef2_r = None

    for arg in sys.argv[1:]:
        if arg == 'end':
            end = True
            print "Run until convergence"
        
        elif arg == 'undirected':
            directedMigration = False   
            print 'Undirected/Random migration'

        elif arg == 'evolved_init_culture': # instead of uniform init C vectors
            evolved_init_culture = True
            print 'evolutionary initial culture vectors'

        elif (arg == 'evolved_init_culture2'  # instead of uniform init C vectors
              or (len(arg.split(':')) == 2 and
                  arg.split(':')[0] == 'evolved_init_culture2')):
            evolved_init_culture2 = True
            print 'evolutionary initial culture vectors (new version ef2)'
            if (len(arg.split(':')) == 2):
                # value of r the number of traits to change each step
                ef2_r = int(arg.split(':')[1])

        elif (arg == 'prototype_init_culture' #instead of uniform init C vectors
              or (len(arg.split(':')) == 2 and
                  arg.split(':')[0] == 'prototype_init_culture')):
            prototype_init_culture = True
            print 'reverse k-means prototype based initial culture vectors'
            if (len(arg.split(':')) == 2):
                # value of k the number of prototypes
                cluster_k = int(arg.split(':')[1])
            else:
                cluster_k = 3

        elif arg == 'trivial_ultrametric_init_culture': # instead of uniform init C vectors, trivial but perfectly ultrametric
            trivial_ultrametric_init_culture = True
            print 'trivial ultrametric initial culture vectors'
            
        elif arg == 'no_migration':
            no_migration = True
            print 'No migration'

        elif (len(arg.split(':')) == 2 and  
              arg.split(':')[0] == 'F'):
              # set value of F, dimension of culture vectors (default 5)
              F = int(arg.split(':')[1])

        elif (len(arg.split(':')) == 2 and  
              arg.split(':')[0] == 'm'):
              # set value of m, dimension of lattice (default 25)
              m = int(arg.split(':')[1])

        elif (len(arg.split(':')) == 2 and  
              arg.split(':')[0] == 'read_init_culture'):
             # read_init_culture:culture.csv
             # read initial culture from CSV file
             # in this mode we get the values of q, F, and n from the file
             # (resp. max number of different values in a column,
             # number of columns, number of rows)
             init_culture_csv_file = arg.split(':')[1]

        elif (len(arg.split(':')) == 2 and
              arg.split(':')[0] == 'resumeFrom'):
              # resumeFrom:n,q,beta_p,beta_s,sigma,init_random_prob,i
              # instead of running all parameters, start at the specified set
              # (only makes sense if all other parameters are same as run that
              # is being 'resumed' - start from the parameters MPI rank 0
              # was doing when job terminated. If this is specified then
              # results<rank>.csv  and parameter<rank>.csv files will be
              # appended not overwritten
              resargs = arg.split(':')[1].split(',')
              if (len(resargs) != 7):
                sys.stderr.write('expecting 7 values for resumeFrom:  resumeFrom:n,q,beta_p,beta_s,sigma,init_random_prob,i\n')
                sys.exit(1)
              if resargs[5] == 'None':
                resinitrandomprob = None
              else:
                resinitrandomprob = float(resargs[5])
              resumeFrom = (int(resargs[0]), int(resargs[1]), 
                            float(resargs[2]), float(resargs[3]),
                            float(resargs[4]), resinitrandomprob,
                            int(resargs[6]))

        elif arg.isdigit():
            n_list.append(int(arg))


        else:
            cppmodel = arg
            if not os.path.isfile(cppmodel):
                print 'Model executable not found.'
                sys.exit()        

    if init_culture_csv_file != None:
        if len(n_list)  > 0:
            print 'Cannot provide n values for read_init_culture'
            sys.exit(1)
        if evolved_init_culture or evolved_init_culture2 or prototype_init_culture or trivial_ultrametric_init_culture:
            print 'Cannot have both read_init_culture:csvfile and evolved_init_culture'
            sys.exit(1)
        initial_culture = list()
        F = None
        q = 0
        for row in csv.reader(open(init_culture_csv_file)):
            initial_culture.append(array([int(x) for x in row]))
            if F == None:
                F = len(initial_culture[-1])
            else:
                if len(initial_culture[-1]) != F:
                    sys.stderr.write('bad row in ' + str(init_culture_csv_file)
                                     + ' expecting ' + str(F) + ' columns ' +
                                     ' but got ' + str(F) + '\n')
                    sys.exit(1)
            q = max(q, max(initial_culture[-1]) + 1) # +1 since values in 0..q-1
        n = len(initial_culture)
        if n > m**2:
            new_m = int(math.ceil(math.sqrt(n)))
            sys.stderr.write(('WARNING: n = %d agents but lattice only  ' +
                             '(m = %d)**2, m set to %d\n') %
                             (n, m, new_m))
            m = new_m
        n_list = [n]
        q_list = [q]
        
    elif n_list == None:
        print 'Please provide n values'
        sys.exit()

    if evolved_init_culture2:
        if ef2_r == None:
            ef2_r = F / 2
        else:
            if ef2_r > F:
                sys.stderr.write('value of r for evolved_init_culture2:r must be <= F (r = %d, F = %d)\n' % (ef2_r, F))
                sys.exit(1)
        
    if cppmodel == None:
        print 'Model executable not found.'
        sys.exit()          
   

    scenario(scriptpath, runs, 'results/', tmax, F, m, False, True,
                                      beta_p_list, beta_s_list, 
                                      n_list, q_list, time_list, False, 
                                      cppmodel, end, evolved_init_culture,
                                      evolved_init_culture2,
                                      prototype_init_culture,
                                      trivial_ultrametric_init_culture,
                                      sigma_list, initial_culture,
                                      init_culture_csv_file,
                                      init_random_prob_list,cluster_k, ef2_r,
                                      moore_radius,
                                      resumeFrom)

