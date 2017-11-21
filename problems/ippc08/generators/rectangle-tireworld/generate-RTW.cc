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

int x,y,h,v,u,seed;

/* Program options. */
static struct option long_options[] = {
    { "xsize", required_argument, 0, 'x' },
    { "ysize", required_argument, 0, 'y' },
    { "horizontal", required_argument, 0, 'h' },
    { "vertical", required_argument, 0, 'v' },
    { "unsafe", required_argument, 0, 'u' },
    { "seed", required_argument, 0, 's' },
    { "help", no_argument, 0, '?' },
    { 0, 0, 0, 0 }
};
static const char OPTION_STRING[] = "x:y:h:v:u:s:?";

/* Displays help. */
static void display_help()
{
  std::cout << "usage: generate-RTW [options]" << std::endl
            << "options:" << std::endl
	    << "  -x x, --xsize=x\t"
	    << "width of the rectangle world" << std::endl
	    << "  -y y, --ysize=y\t"
	    << "height of the rectangle world" << std::endl
	    << "  -h h, --horizontal=h\t"
	    << "number of horizontal lines in the rectangle world" << std::endl
	    << "  -v v, --vertical=h\t"
	    << "number of vertical lines in the rectangle world" << std::endl
	    << "  -u u, --unsafe=u\t"
	    << "number of unsafe points in the rectangle world" << std::endl
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

    x = y = h = v = 0;
    u = -1;
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
	    case 'x':
		x = atoi(optarg);
		break;
	    case 'y':
		y = atoi(optarg);
		break;
	    case 'h':
		h = atoi(optarg);
		break;
	    case 'v':
		v = atoi(optarg);
		break;
	    case 'u':
		u = atoi(optarg);
		break;
	    case 's':
		seed = atoi(optarg);
		break;
	    case '?':
		display_help();
		return 0;
	    case ':':
	    default:
		cerr << "Try `generate-RTW --help' for more information."
			  << endl;
		return -1;
	    }
	}
	srandom(seed);
	
	if ( (x < 1) || (y < 1) ||
	     (h < 1) || (v < 1) ||
	     (u < 0) ) {
	    cerr << "You need to specify x, y, h, v (all greater or equal to 1) and u (greater than 0)." << endl;
	    return -1;
	}

	seed += x+y+h+v+u;

	if (optind < argc) {
	    cerr << "There seems to be too many \"free\" arguments..." << endl;
	    return -1;
	}
	

	cout << "(define (problem rect"
	     <<"-"<<x
	     <<"-"<<y
	     <<"-"<<h
	     <<"-"<<v
	     <<"-"<<seed-(x+y+h+v+u)
	     <<")" << endl
	     << "  (:domain rectangle-world)" << endl;

	/**
	 * create "numbers" from 0 to max(x,y)-1
	 */
	cout << "  (:objects ";
	for (int i=0; i < max(x,y); i++)
	    cout << "n" << i << " ";
	cout << "- int)" << endl;

	/**
	 * start inits with
	 *initial position: (n0,n0)
	 */
	cout << "  (:init (xpos n0)" << endl
	     << "	 (ypos n0)" << endl;

	/**
	 * define the "next" property on numbers
	 */
	for (int i=0; i < max(x,y)-1; i++)
	    cout << "         (next n"<<i<<" n"<<i+1<<")" << endl;


	bool* safeTab = (bool*) malloc( x*y * sizeof(bool) );

	/**
	 * add safe rows and columns
	 */

	for (int i=0 ; i < x ; i++) safeTab[i] = false;
	for (int i=0 ; i < h ; i++) safeTab[random() % x] = true;
	for (int i=0 ; i < x ; i++)
	    if (safeTab[i])
		cout << "	 (safeX n"<<i<<")" << endl;

	for (int i=0 ; i < y ; i++) safeTab[i] = false;
	for (int i=0 ; i < v ; i++) safeTab[random() % y] = true;
	for (int i=0 ; i < y ; i++)
	    if (safeTab[i])
		cout << "	 (safeY n"<<i<<")" << endl;

	/**
	 * add unsafe points
	 */
	
	for (int i=0 ; i < x ; i++)
	    for (int j=0 ; j < y ; j++)
		safeTab[i+j*x] = false;
	
	for (int i=0 ; i < u ; i++) 
	    safeTab[(random() % x) + (random() % y) * x] = true;

	for (int i=0 ; i < x ; i++)
	    for (int j=0 ; j < y ; j++)
		if (safeTab[i + j * x])
		    cout << "	 (unsafe n"<<i<<" n"<<j<<")"<<endl;

	free(safeTab);

	/**
	 * end inits
	 */
	cout << "  )" << endl;

	/**
	 * define goal, goal-reward and metric
	 */
	cout << "  (:goal (and (xpos n"<<x-1<<") (ypos n"<<y-1<<")))" << endl
	     << "  (:goal-reward 1000)" << endl
	     << "  (:metric maximize (reward))" << endl
	     << ")" << endl;


    } catch (const exception& e) {
	cerr << endl << "generate-RTW: " << e.what() << endl;
	return 1;
    } catch (...) {
	cerr << "generate-RTW: fatal error" << endl;
	return -1;
    }
    
    
    return 0;
}
