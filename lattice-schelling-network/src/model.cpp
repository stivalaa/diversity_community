/*  Copyright (C) 2011 Jens Pfau <jpfau@unimelb.edu.au>
 *  Copyright (C) 2014 Alex Stivala <stivalaa@unimelb.edu.au>
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

#define SVN_VERSION_MODEL_CPP "Schelling segregation plus network (Neal&Neal) model rev 3"


static char *tempfileprefix = NULL;




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





// Calculate the cultural similarity between agents a and b as the proportion of
// features they have in common.
inline double similarity(Grid<int>& C, const int F, const int a, const int b) {
	int same = 0;
	for (int i = 0; i < F; i++)
			same += (C(a,i) == C(b,i));
	return (double)same/F;
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


// Count fraction of
// agents in Moore neighbourhood of the focal agent a that have
// the same culture
double similar_neighbours(Grid<int>& L, Grid<int>& O,
                                   Grid<int>& C, int m, int F, int a,
                                   int moore_radius) {
    int ax = L(a,0);
    int ay = L(a,1);
    unsigned int total_neighbours = 0;
    unsigned int count_similar_neighbours = 0;
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
                        if (diff(C, F, a, b) == 0) {
                          count_similar_neighbours++;
                        }
                    }
                }
            }
        }
    }
    return (double)count_similar_neighbours / total_neighbours;
}

// stopping criterion for Schelling segregation model: all agents
// have at least the desired fractino of similar neighbours
bool all_happy(Grid<int >&L, Grid<int>& O, Grid<int>& C, int n, int m, int F, int moore_radius, double similar_wanted) {
  for (int i = 0; i < n; i++){
    bool happy = (similar_neighbours(L, O, C, m, F, i, moore_radius) >= similar_wanted);
    if (!happy) {
      return false;
    }
  }
  return true;
}

//
// Schelling segregation model 
//   given the descired frcation of similar neighbours similar_wanted,
//   randomly move agents to vacant point until each agent is 'happy'
//   because the fraction of similar neighbours in its Moore neighbourodo
//   is at least as much as similar_wanted
// 
unsigned long long schelling(Grid<int>& L, Grid<int>& O, Grid<int>& C,
                             double similar_wanted,
                             int tmax, int n, int m, int F,
		                         int q,
		                         int timesteps[], int time_list_length, std::ofstream& log, 
                             int moore_radius,
                             std::vector<std::pair<int, int>>& empty_coords) {

  srand(time(NULL)+(unsigned int)getpid()); // so processes started at same time have different seeds (time is only second resolution)

	std::cout << "tmax: " << tmax << std::endl;
	std::cout << "n: " << n << std::endl;
	std::cout << "m: " << m << std::endl;
	std::cout << "F: " << F << std::endl;
	std::cout << "q: " << q << std::endl;

  int a, b, idx;
  int nextstep = 0;

	// run model
    for (unsigned long long t = 0; t < (unsigned long long) tmax; t++) {
    	// If this iteration is in the time list, write out the current state
    	// of the simulation.
    	if (nextstep < time_list_length and (unsigned long long)timesteps[nextstep] == t) {
	  L.write((toString(tempfileprefix) + "-" + toString(timesteps[nextstep]) + ".L").c_str(), ',');
	  C.write((toString(tempfileprefix) + "-" + toString(timesteps[nextstep]) + ".C").c_str(), ',');
    		nextstep++;
    	}
      

    	if (t == 50000 || t == 100000 || t == 500000 || t == 1000000 || (t > 0 && t % 10000000 == 0)) {
    		std::cout << "Reaching " << t << " iterations." << std::endl;

			  if (all_happy(L, O, C, n, m, F, moore_radius, similar_wanted)) {
  		  		std::cout << "Stopping after " << t << " iterations." << std::endl;
            return t;
  			}
      }


    	// Draw one agent randomly.
    	a = rand() % n;

      // Count agents in Moore neighbourhood of the focal agent that have
      // the same culture
      
      bool happy = (similar_neighbours(L, O, C, m, F, a, moore_radius) >= similar_wanted);
      if (!happy) {
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
    std::cout << "Stopping after tmax = " << tmax << " iterations." << std::endl;
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
	int n = 100, m = 10, F = 5, q = 15;

	int tmax = 100000;

  double similar_wanted = 0.5;


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
		tempfileprefix = argv[index++];
    moore_radius = atoi(argv[index++]);   // Moore neighbourhood radius
    similar_wanted = atof(argv[index++]); // fraction similar neighbours wanted
	}

   if (n > m*m) {
    std::cerr << "not enough lattice points for agents: m = " << m << " and n = " << n <<std::endl;
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
	Grid<int> L(n,2,-1);   // lattice co-ordinates of each agent
	Grid<int> C(n,F,0);    // culture vector of each agent
	Grid<int> O(m,m,-1); // in this version O(x,y) contains agent id at that location or -1 if unoccupied
  std::vector<std::pair<int,int>> empty_coords; // list of empty lattice (x,y) positions


	if (argc == 1) {
		std::cerr << "must provide initialization parameters" << std::endl;
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

  assert(m*m >= n);
  assert(empty_coords.size() == (unsigned)(m*m - n));

	// run the Schelling segregation model
  unsigned long long tend = schelling(L, O, C, similar_wanted, 
                                      tmax, n, m, F, q, 
                                      time_list, time_list_length, log,
                                      moore_radius, empty_coords);

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
