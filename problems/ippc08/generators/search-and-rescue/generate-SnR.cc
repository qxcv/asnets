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

int z;

/* Program options. */
static struct option long_options[] = {
    { "zones", required_argument, 0, 'z' },
    { "help", no_argument, 0, '?' },
    { 0, 0, 0, 0 }
};
static const char OPTION_STRING[] = "z:?";

/* Displays help. */
static void display_help()
{
  std::cout << "usage: generate-SnR [options]" << std::endl
            << "options:" << std::endl
	    << "  -z z, --zones=z\t"
	    << "number of zones" << std::endl
	    << "  -?     --help\t\t"
            << "display this help and exit"
            << endl
            << endl
            << "Report bugs to < ... >." << endl;
}

int main(int argc, char **argv)
{

    z = 0;
    
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
	    case 'z':
		z = atoi(optarg);
		break;
	    case '?':
		display_help();
		return 0;
	    case ':':
	    default:
		cerr << "Try `generate-SnR --help' for more information."
			  << endl;
		return -1;
	    }
	}
	
	if (z < 1) {
	    cerr << "You need to specify z (greater or equal to 1)." << endl;
	    return -1;
	}

	if (optind < argc) {
	    cerr << "There seems to be too many \"free\" arguments..." << endl;
	    return -1;
	}
	
	cout << ";; -*-lisp-*-" << endl
	     << ";; Search & Rescue domain:" << endl
	     << ";;" << endl
	     << ";;   Florent Teichteil, 2008" << endl
	     << ";;" << endl
	     << ";; small modifications and problem generator:" << endl
	     << ";;   Olivier Buffet, 2008" << endl
	     << "" << endl
	     << "(define (problem search-and-rescue-"<<z<<")" << endl
	     << "" << endl
	     << "  (:domain search-and-rescue)" << endl
	     << "" << endl;

	/**
	 * create z zones
	 */
	cout <<"  (:objects ";	    
	for (int i=1; i < z+1; i++)
	    cout << "z" << i << " ";
	cout << "- zone)" << endl
	     << "" << endl;
	
	/**
	 * init state
	 */
	cout << "  (:init (at base) (on-ground) (human-alive))" << endl
	     << "" << endl;
  
	/**
	 * define goal, goal-reward and metric
	 */
	cout << "  (:goal (and (mission-ended)))" << endl
	     << "  (:goal-reward 1000)" << endl
	     << "  (:metric maximize (reward))" << endl
	     << ")" << endl;


    } catch (const exception& e) {
	cerr << endl << "generate-SnR: " << e.what() << endl;
	return 1;
    } catch (...) {
	cerr << "generate-SnR: fatal error" << endl;
	return -1;
    }
    
    
    return 0;
}
