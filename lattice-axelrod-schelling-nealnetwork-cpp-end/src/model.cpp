/*  Copyright (C) 2011 Jens Pfau <jpfau@unimelb.edu.au>
 *  Copyright (C) 2015 Alex Stivala <stivalaa@unimelb.edu.au>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <limits>
#include <cassert>
#include <unistd.h> // for getpid()


#include "model.hpp"

#define SVN_VERSION_MODEL_CPP "Moore neighbourhood Axelrod-Schelling model mean overlap tolerance tau with immutable features and interaction phase rev 3"


// modified by ADS to seed srand with time+pid and to also 
// ensure unique tmp files so safe for parallel execution with MPI Python
// NB this involved extra command line pameter to model to not compatible
// with unmodified versions
// Also added tau parameter as threshold for mean overlap tolerance

// This version is similar to the model in Gracia-Lazaro et al 2009
// "Residential segregation and cultural dissemination: An Axelrod-Schelling
// model" Phys. Rev. E 80: 046123

static char *tempfileprefix = NULL;


// TODO implement accelaration of convergence by maintaingin "active agents"
// list as per Barbosa & Fontanari (2009). This is basically the list of
// agents that can possibly interact (as per the test in stop2()), so only
// search this instead of all agents on each step.


// Read the time_list---the iterations at which the stats of the simulation are
// to be printed out---from a temporary file.
int read_time_list(int** time_list, int* n) {
        std::ifstream inFile ((toString(tempfileprefix) + ".T").c_str());

	if (inFile) {
		std::string line;

		// Get the first line and read out the number of time steps in the list
		if (!getline(inFile, line))
			return -1;

		*n = convert<int>(line);
		int* tmplist = new int[*n];

		// Get the list itself
		if (!getline(inFile, line))
			return -1;

		// Read every item of the list
		std::istringstream linestream(line);
		std::string item;
		int i = 0;
		while (getline (linestream, item, ',') and i < *n) {
			tmplist[i] = convert<int>(item);
			i++;
		}

		if (i != *n)
			return -1;

		*time_list = tmplist;
	} else {
		*n = 0;
	}

	return 0;
}



// Calculate Manhattan distance on the lattice between locatino (x1,y1) 
// and (x2,y2)
inline int manhattan_distance(int x1, int y1, int x2, int y2) {
  return abs(x2 - x1) + abs(y2 - y1);
}

//  Calculate the Chebyshev distance (max or L_inf metric) aka
//  chessboard distance.  Used for the Moore neighbourhood (like the
//  Manhattan distance is used for the von Neumann neigbhbourhood) on
//  a lattice.
// Calculate chessboard distance on the lattice between location (x1,y1) 
// and (x2,y2)
inline int chebyshev_distance(int x1, int y1, int x2, int y2) {
  return std::max(abs(x2 - x1), abs(y2 - y1));
}


// Calculate the Euclidean distance on the lattice between 
// location (x1,y1) and (x2,y2)
inline double distance(int x1, int y1, int x2, int y2) {
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}




// Init the lattice, the social network, and the culture of individuals.
void init(Grid<int>& L, Grid<int>& O, Grid<int>& C, int m, int n , int F, int q) {

	std::cout << "Initializing agent locations:" << std::endl;
	std::cout << "n = " << n << ", m = " << m << std::endl;

	// Randomly assign agents to positions on the m x m size lattice defined
	// by L.
	int tmpCell[2];
	for (int i = 0; i < n; i++) {
		do {
			tmpCell[0] = rand() % m;
			tmpCell[1] = rand() % m;
		} while (O(tmpCell[0], tmpCell[1]) == 1);
		L(i,0) = tmpCell[0];
		L(i,1) = tmpCell[1];
		O(tmpCell[0], tmpCell[1]) = 1;
	}

	std::cout << "Agents located on the grid." << std::endl;

	// Randomly assign a culture to each agent, stored in the n x F array C.
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < F; j++) {
			C(i,j) = rand() % q;
		}
	}

	std::cout << "Agents assigned cultures" << std::endl;

}


// Calculate the cultural similarity between agents a and b as the proportion of
// features they have in common.
inline double similarity(Grid<int>& C, const int F, const int a, const int b) {
	double same = 0.0;
	for (int i = 0; i < F; i++)
		if (C(a,i) == C(b,i))
			same += 1.0;
	return same/F;
}

// Calculate the cultural similarity between agents a and b as the proportion of
// mutable features they have in common.
// This version only uses the mutable features (the frist n_immutable are
// immutable)
inline double similarity_mutable(Grid<int>& C, const int F, const int n_immutable, const int a, const int b) {
	int same = 0;
	for (int i = n_immutable; i < F; i++)
		same += (C(a,i) == C(b,i));
	return (double)same/(F-n_immutable);
}



// Count the number of mutable features between a and b that are different
// Doing just the raw count not diff/sim score (div by F-n_immutable)
// pervents problems with not exactly 1.0
// This version only uses the mutable features (the frist n_immutable are
// immutable)
inline int diff_mutable(Grid<int>& C, const int F, const int n_immutable, const int a, const int b) {
	int diff = 0;
	for (int i = n_immutable; i < F; i++)
		diff += (C(a,i) != C(b,i));
	return diff;
}

// Calculate the cultural similarity between agents a and b as the proportion of
// immutable features they have in common.
// This version only uses the immutable features (the frist n_immutable are
// immutable)
inline double similarity_immutable(Grid<int>& C, const int F, const int n_immutable, const int a, const int b) {
	int same = 0;
	for (int i = 0; i < n_immutable; i++)
		same += (C(a,i) == C(b,i));
	return (double)same/(n_immutable);
}



// Count the number of immutable features between a and b that are different
// Doing just the raw count not diff/sim score (div by n_immutable)
// pervents problems with not exactly 1.0
// This version only uses the immutable features (the frist n_immutable are
// immutable)
inline int diff_immutable(Grid<int>& C, const int F, const int n_immutable, const int a, const int b) {
	int diff = 0;
	for (int i = 0; i < n_immutable; i++)
		diff += (C(a,i) != C(b,i));
	return diff;
}


// Count the number of features between a and b that are different
// Doing just the raw count not dividing by F
inline int diff(Grid<int>& C, const int F, const int a, const int b) {
	int diff = 0;
	for (int i = 0; i < F; i++)
		diff += (C(a,i) != C(b,i));
	return diff;
}


// Compute mean cultural overlaps of focal agent a with
// agents in its Moore neighbourhood
double mean_overlap_neighbours(Grid<int>& L, Grid<int>& O,
                               Grid<int>& C, int m, int F, int a,
                               int moore_radius) {
    int ax = L(a,0);
    int ay = L(a,1);
    unsigned int total_neighbours = 0;
    double total_overlap = 0;
    for (int xi = -1*moore_radius; xi <= moore_radius; xi++) {
        for (int yi = -1*moore_radius; yi <= moore_radius; yi++) {
            if ((xi != 0 || yi != 0) &&  // don't include focal agent itself
                std::max(abs(xi), abs(yi)) <= moore_radius) {
                int bx = ax + xi;
                int by = ay + yi;
                // handle edge of lattice, not toroidal
                if (bx >= 0 && bx < m && by >= 0 && by < m) {
                    int b = O(bx, by);
                    if (b > -1) {
                        // position is occupied by an agent
                        total_neighbours++;
                        total_overlap += similarity(C, F, a, b);
                    }
                }
            }
        }
    }
    return total_overlap / total_neighbours;
}



void save(Grid<int>& lastC, Grid<double>& lastG, Grid<int>& C, Grid<double>& G, int n, int F) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			lastG(i,j) = G(i,j);
		}

		for (int j = 0; j < F; j++) {
			lastC(i,j) = C(i,j);
		}
	}
}


// test for equilbrium, return true if no more change is possible
bool stop2(Grid<int>& C, Grid<int>& L, int n, int F, int n_immutable,
           double tau,  int moore_radius) {
    // at equilibrium, all agents in a neighbourhood must have either
    // identical mutable culture, or completely distinct
    // culture (no traits in common, similarity = 0)

    const int num_mutable = F - n_immutable;
    for (int i = 0; i < n; i++)  {
        for (int j = i+1; j < n; j++) { // symmetric, only need i,j where j > i
            if (chebyshev_distance(L(i,0), L(i,1), L(j, 0), L(j,1)) > moore_radius) {
              // TODO this could be made more efficient by looping over only
              // agents in neighbourhood instead of this way of looping over
              // all and just doing next if not in neighbourhood
              continue;
            }
            double sim = similarity(C, F, i, j);
            assert(sim >= 0.0 && sim <= 1.0);
            // NB using >= and <= not == to for floating point comparison
            // with 0 or 1 since floating point == is dangerous, but
            // values cannot be < 0 or > 1, as just asserted so equivalent
            // to equality
            // Also use number of differet features (int) rather than 
            // similarity (double) for mutable to avoid problems with
            // floating point comparison entirely (so instead of sim>=1.0 have 
            // diff==0 and instead of sim<=0.0 have diff==F-n_immutable)
            int num_diff_mutable = diff_mutable(C, F, n_immutable, i, j);
            int num_diff_total   = diff(C, F, i, j);
            assert(num_diff_mutable >= 0 || num_diff_mutable <= n_immutable);
            assert(num_diff_total >=0 || num_diff_total <= F);
            double immutable_sim = similarity_immutable(C, F, n_immutable, i ,j);
            if ( !((num_diff_mutable == 0) ||
                   (num_diff_total == F)) ) {
                return false;
            }
        }
    }
    return true;
}


unsigned long long model(Grid<int>& L, Grid<int>& O, Grid<int>& C, int tmax, int n, int m, int F,
		int q, double r, double s,
		bool toroidal, bool network, double tolerance, bool directed_migration,
		double phy_mob_a, double phy_mob_b, double soc_mob_a, double soc_mob_b,
		double r_2, double s_2, double phy_mob_a_2, double phy_mob_b_2, double soc_mob_a_2, double soc_mob_b_2,
		double k, int timesteps[], int time_list_length, std::ofstream& log, 
                         double tau, int n_immutable, int moore_radius, 
                         std::vector<std::pair<int, int>>& empty_coords) {
       srand(time(NULL)+(unsigned int)getpid()); // so processes started at same time have different seeds (time is only second resolution)



	std::cout << "tmax: " << tmax << std::endl;
	std::cout << "n: " << n << std::endl;
	std::cout << "m: " << m << std::endl;
	std::cout << "F: " << F << std::endl;
	std::cout << "q: " << q << std::endl;
        std::cout << "tau: " << tau << std::endl;
        std::cout << "n_immutable: " << n_immutable << std::endl;

    double w[n];

    double sumw = 0.0;


	double nw[n];
	double cw[m*m];

    int a, b, idx;

const unsigned long long extra_iters = 1000000;
bool would_have_stopped = false;
unsigned long long lastiter = 0;

	// run model
    for (unsigned long long t = 0; true; t++) {
    	// If this iteration is in the time list, write out the current state
    	// of the simulation.
    	if (t == 50000 || t == 100000 || t == 500000 || t == 1000000 || (t > 0 && t % 10000000 == 0)) {
    		std::cout << "Reaching " << t << " iterations." << std::endl;
//			save(lastC, lastG, C, G, n, F);
                if (stop2(C, L, n, F, n_immutable, tau, moore_radius)) {
  		  		std::cout << "Stopping after " << t << " iterations." << std::endl;
            return t;
  			}
      }


    	// Draw one agent randomly.
    	a = rand() % n;

        //fprintf(stderr, "t = %d a = %d \n", t, a);//XXX

        // Draw one other agent randomly, from those in the Moore
        // neighbourhood of the focal agent.
        std::vector<int>neighbours;
        // TODO this is inefficient, having to loop over all agents,
        // but lattice implemted in L has no way to lookup what agent is
        // at position (x,y), just the (x,y) for a given agent.
        for (int x = 0; x < n; x++) {
          if (x != a && chebyshev_distance(L(a,0), L(a,1), L(x,0), L(x,1)) <= moore_radius) {
            neighbours.push_back(x);
          }
        }

        if (neighbours.size() == 0) {
          continue;
        }

        b = neighbours[rand() % neighbours.size()];
      
        // With the probability of their attraction,
        // a and b interact successfully.
        double sim = similarity(C, F, a, b);
        if (rand()/(float)RAND_MAX < sim) {
          // Randomly decide on one mutable feature that a and b do not have in common yet.
          int num_diff_mutable = diff_mutable(C, F, n_immutable, a, b);
          if (num_diff_mutable > 0) {
            do {
              idx = (rand() % (F - n_immutable)) + n_immutable; // only mutable features
            } while (C(a,idx) == C(b, idx));
            
            // Let a copy this feature from b.
            C(a,idx) = C(b,idx);
          }
        }
        else {
          // unsuccessful interaction, 
          // move to random free lattice location if mean overlap with
          // neighbours is less than tolerance threshold tau
          if (mean_overlap_neighbours(L, O, C, m, F, a, moore_radius) < tau) {
            // move to a random free lattice position
            assert(empty_coords.size() > 0);
            int i = rand() % empty_coords.size() ;
            std::pair<int, int> freepos = empty_coords[i];
            assert(O(freepos.first, freepos.second) == -1);
            std::pair<int, int> old_coords = std::pair<int, int>(L(a, 0), L(a, 1));
            O(L(a, 0), L(a, 1)) = -1;
            L(a, 0) = freepos.first;
            L(a, 1) = freepos.second;
            O(freepos.first, freepos.second) = a;
            empty_coords.erase(empty_coords.begin() + i);
            empty_coords.push_back(old_coords);
          }
        }
    }
    return tmax;
}


//
// agent interactin phase from Neal & Neal (2014). Make social tie (or not)
// between each pair of agents with probability from logistic selection
// function based on proximity and homophily
// Note now not just delta for same (1) different (0) type, but 
// normalized hamming distance so varies between completely same (1)
// and completely different (0) according to number of features
// that differ (like in Axelrod model)
//
void agent_interaction(Grid<int>& L,
                       Grid<int>& C, Grid<int>& G, int n, int F, 
                       double beta_H, double beta_p) {
  double beta_0 = -1.0*(beta_H + beta_p); // sets max prob to 0.5
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dist = distance(L(i, 0), L(i, 1), L(j, 0), L(j, 1));
      double distprime = 1.0 / (1.0 + exp((dist - 5) / 0.5)); // as in paper, but why dist-5?
      //double delta = diff(C, F, i, j) > 0 ? 0 : 1;
      double x = exp(beta_0 + beta_H*similarity(C, F, i, j) + beta_p*distprime);
      double p = x / (1.0 + x);
      assert(p >= 0 && p <= 0.5);
      if (rand() / (float)RAND_MAX < p) {
        G(i, j) = 1;
        G(j, i) = 1;
      }
    }
  }
}


int main(int argc, char* argv[]) {
	std::ofstream log("log.txt");

	// If the binary file is called with the argument -v, only the svn version
	// this binary was compiled from is printed.
	if (argc == 2 and argv[1][0] == '-' and argv[1][1] == 'v') {
		std::cout << "model.hpp: " << SVN_VERSION_MODEL_HPP << ", model.cpp: " << SVN_VERSION_MODEL_CPP << std::endl;
		return 0;
	}


	// Otherwise set default model arguments.
	int n = 625, m = 25, F = 5, q = 100;
        int n_immutable = 0;

	int tmax = 100000;
	double r = 1;
	double s = 1;

	bool toroidal = false;
	bool network = true;

    double tolerance = -1;
    bool directed_migration = true;

    double phy_mob_a = 1;
    double phy_mob_b = 10;
    double soc_mob_a = 1;
    double soc_mob_b = 10;

	double r_2 = r;
	double s_2 = s;

    double phy_mob_a_2 = phy_mob_a;
    double phy_mob_b_2 = phy_mob_b;
    double soc_mob_a_2 = soc_mob_a;
    double soc_mob_b_2 = soc_mob_b;

    double k = 0.01;

   double tau = 0.0;

    // size of the Moore neighbourhood for actors to interact, i.e.
    // the maximum Chebysev distance between interacting actors
    int moore_radius = 1;


    // social tie logistic function praameters
    double beta_H = 2.5; // homophily 
    double beta_p = 2.5; // proximity 

    // If there are arguments, assume they hold model arguments in the following
    // order.
	if (argc > 1) {
		int index = 1;
		tmax = atoi(argv[index++]);
		n = atoi(argv[index++]);
		m = atoi(argv[index++]);
		F = atoi(argv[index++]);
		q = atoi(argv[index++]);
		r = atof(argv[index++]);                     // not used
		s = atof(argv[index++]);                     // not used
		toroidal = atoi(argv[index++]);              // not used
		network = atoi(argv[index++]);               // not used
		tolerance = atof(argv[index++]);             // not used
		directed_migration = atoi(argv[index++]);    // not used
		phy_mob_a = atof(argv[index++]);             // not used
		phy_mob_b = atof(argv[index++]);             // not used
		soc_mob_a = atof(argv[index++]);             // not used
		soc_mob_b = atof(argv[index++]);             // not used

		r_2 = atof(argv[index++]);                   // not used
		s_2 = atof(argv[index++]);                   // not used
		phy_mob_a_2 = atof(argv[index++]);           // not used
		phy_mob_b_2 = atof(argv[index++]);           // not used
		soc_mob_a_2 = atof(argv[index++]);           // not used
		soc_mob_b_2 = atof(argv[index++]);           // not used

		k = atof(argv[index++]);                     // not used

		tempfileprefix = argv[index++];
                tau = atof(argv[index++]); // reuse theta parameter as tau here
                n_immutable = atoi(argv[index++]);
	}

    if (toroidal) {
    	std::cout << "alarm, toroidal not supported at the moment" << std::endl;
    	return -1;
    }

 	// Try to read the list of iterations that determine when statistics are to
	// be created from a temporary file.
	int* time_list = NULL;
	int time_list_length = 0;
	int res = read_time_list(&time_list, &time_list_length);
	if (res == -1) {
		std::cout << "The time list file could not be read or there was a problem with its format." << std::endl;
		return -1;
	}

	Grid<int> G(n,n,0);    // social network
	Grid<int> L(n,2,-1);
	Grid<int> C(n,F,0);
	Grid<int> O(m,m,-1); // in this version O(x,y) contains agent id at that location or -1 if unoccupied
        std::vector<std::pair<int,int>> empty_coords; // list of empty lattice (x,y) positions

	if (argc == 1) {
		init(L, O, C, m, n, F, q);
	} else {
		// load data from file
          G.read((toString(tempfileprefix) + ".adj").c_str(), ' ');
 	        L.read((toString(tempfileprefix) + ".L").c_str(), ',');
	        C.read((toString(tempfileprefix) + ".C").c_str(), ',');
	}

	for (int i = 0; i < n; i++)
		O(L(i,0), L(i,1)) = i; // O(x,y) is agent at lattice location (x,y)

        for (int x = 0; x < m; x++) {
          for (int y = 0; y < m; y++) {
            if (O(x, y) == -1) {
              empty_coords.push_back(std::pair<int, int>(x, y));
            }
          }
        }


	// Call the model
    unsigned long long tend = model(L, O, C, tmax, n, m, F, q, r, s, toroidal, network, tolerance, directed_migration,
    		phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b,
    		r_2, s_2, phy_mob_a_2, phy_mob_b_2, soc_mob_a_2, soc_mob_b_2, k,
                                    time_list, time_list_length, log, tau, n_immutable, moore_radius, empty_coords);

        std::cout << "Last iteration: " << tend << std::endl;

        // do the second ("agent-interaction") phase to form social network ties
        agent_interaction(L, C, G, n, F, beta_H, beta_p);
    
        // Write out the state of the simulation
        G.write((toString(tempfileprefix) +  ".adj").c_str(), ' ');
        L.write((toString(tempfileprefix) + ".L").c_str(), ',');
        C.write((toString(tempfileprefix) + ".C").c_str(), ',');

	std::ofstream outFile((toString(tempfileprefix) + ".Tend").c_str());
	outFile << tend;


	delete[] time_list;
	time_list = NULL;

	std::cout << "Fin" << std::endl;
	return 0;
}
