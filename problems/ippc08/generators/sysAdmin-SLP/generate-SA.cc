#include <math.h>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstdio>
#if HAVE_GETOPT_LONG
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <getopt.h>
#else
#include "getopt.h"
#endif

#include <vector>

#ifndef WHERE
#define WHERE { std::cerr<<__FUNCTION__<<" in "<<__FILE__<<":"<<__LINE__<<std::endl; }
#endif

using namespace std;

int nodes,links,seed;
bool* graph;

/* Program options. */
static struct option long_options[] = {
    { "nodes", required_argument, 0, 'n' },
    { "links", required_argument, 0, 'l' },
    { "seed", required_argument, 0, 's' },
    { "help", no_argument, 0, '?' },
    { 0, 0, 0, 0 }
};
static const char OPTION_STRING[] = "n:l:s:?";

/* Displays help. */
static void display_help()
{
  std::cout << "usage: generate-SA [options]" << std::endl
            << "options:" << std::endl
	    << "  -n n, --nodes=n\t"
	    << "number of nodes in the loop" << std::endl
	    << "  -l l, --links=l\t"
	    << "number of additionnal links" << std::endl
	    << "  -s s, --seed=s\t"
	    << "random number generator seed (if not using time)" << std::endl
	    << "  -?     --help\t\t"
            << "display this help and exit" << std::endl
            << endl
            << endl
            << "Report bugs to < ... >." << endl;
}

int main(int argc, char **argv)
{

    nodes = 5;
    links = 3;
    seed = time(NULL);
    
    try {
	/*
	 * Get command line options.
	 */
	while (1) {
	    int option_index = 0;
	    int c = getopt_long(argc, argv, OPTION_STRING,
				long_options, &option_index);
	    if (c == -1) {
		break;
	    }
	    switch (c) {
	    case 'n':
		nodes = atoi(optarg);
		break;
	    case 'l':
		links = atoi(optarg);
		break;
	    case 's':
		seed = atoi(optarg);
		break;
	    case '?':
		display_help();
		return 0;
	    case ':':
	    default:
		cerr << "Try `generate-SA --help' for more information."
			  << endl;
		return -1;
	    }
	}
	srandom(seed);
	
	if ( (nodes <= 0) || (links < 0) || (links > nodes*(nodes-2)) ) {
	    cerr << "You need to specify n (>=1) and l (>=0 & <= n^2-2n)." << endl;
	    return -1;
	}

	seed += nodes+links;

	if (optind < argc) {
	    cerr << "There seems to be too many \"free\" arguments..." << endl;
	    return -1;
	}
	

	/**
	 * Lets create a table of links (i.e. the graph)
	 */
	
	graph=(bool*) malloc (nodes*nodes*sizeof(bool));
	// Default: no link between two nodes
	for(int i=0; i<nodes; i++)
	    for(int j=0; j<nodes; j++)
		graph[i+j*nodes]=false;

	// Create oriented ring: link i with i+1
	for(int i=0; i<(nodes-1); i++)
	    graph[i+(i+1)*nodes]=true;
	graph[(nodes-1)+0*nodes]=true;

	// Add random links everywhere else (not on the diagonal though)
	for(int k=0; k<links; k++) {
	    bool found=false;
	    do {
		int i=random()%nodes;
		int j=random()%(nodes-2);
		j=(i+j+2)%nodes;
		if (graph[i+j*nodes]==false) {
		    graph[i+j*nodes]=true;
		    found = true;
		}
	    } while (found == false);
	}

	/**
	 * Start outputing problem
	 */

	cout << "(define (problem sysadmin"
	     <<"-"<<nodes
	     <<"-"<<links
	     <<"-"<<seed-(nodes+links)
	     <<")" << endl
	     << "  (:domain sysadmin-slp)" << endl;

	/**
	 * create "numbers" from 0 to max(x,y)-1
	 */
	cout << "  (:objects ";
	for (int i=0; i < nodes; i++)
	    cout << "comp" << i << " ";
	cout << "- comp)" << endl;


	/**
	 * Inits
	 */
	cout << "  (:init" << endl;

	/**
	 * Output the graph
	 */
	for (int i=0; i < nodes; i++)
	    for (int j=0; j < nodes; j++)
		if (graph[i+j*nodes])
		    cout << "	 (conn comp" << i << " comp" << j << ")" << endl;

	/**
	 * end inits
	 */
	cout << "  )" << endl;

	/**
	 * define goal, goal-reward and metric
	 */
	cout << "  (:goal (forall (?c - comp)" << endl
	     << "                 (up ?c)))" << endl
	     << "  (:goal-reward 500)" << endl
	     << " (:metric maximize (reward))" << endl
	     << ")" << endl;

    } catch (const exception& e) {
	cerr << endl << "generate-SA: " << e.what() << endl;
	return 1;
    } catch (...) {
	cerr << "generate-SA: fatal error" << endl;
	return -1;
    }
    
    
    return 0;
}
