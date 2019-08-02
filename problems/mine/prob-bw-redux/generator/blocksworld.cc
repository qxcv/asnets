#include <iostream>
#include <vector>
#include <set>
#include <assert.h>

#if HAVE_GETOPT_LONG
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <getopt.h>
#else
#include "getopt.h"
#endif

#ifndef WHERE
#define WHERE { std::cerr<<__FUNCTION__<<" in "<<__FILE__<<":"<<__LINE__<<std::endl; }
#endif

using namespace std;

int costBlock, costTower;

void
print_domain( std::ostream &os )
{
  os << "(define (domain blocks-domain)" << std::endl
     << "  (:requirements :probabilistic-effects :conditional-effects :equality :typing";
  if ((costBlock>0) || (costTower>0))
      os << " :rewards";
  os << ")" << std::endl
     << "  (:types block)" << std::endl
     << "  (:predicates (holding ?b - block) (emptyhand) (on-table ?b - block) (on ?b1 ?b2 - block) (clear ?b - block))" << std::endl;

  os << "  (:action pick-up" << std::endl
     << "    :parameters (?b1 ?b2 - block)" << std::endl
     << "    :precondition (and (emptyhand) (clear ?b1) (on ?b1 ?b2))" << std::endl
     << "    :effect" << std::endl;
  if (costBlock>0)
      os << "     (and" << std::endl
	 << "      (decrease reward " << costBlock << ")" << std::endl;
  os << "      (probabilistic" << std::endl
     << "        3/4 (and (holding ?b1) (clear ?b2) (not (emptyhand)) (not (on ?b1 ?b2)))" << std::endl
     << "        1/4 (and (clear ?b2) (on-table ?b1) (not (on ?b1 ?b2))))" << std::endl;
  if (costBlock>0)
      os << "     )" << std::endl;
  os << "  )" << std::endl
     << "  (:action pick-up-from-table" << std::endl
     << "    :parameters (?b - block)" << std::endl
     << "    :precondition (and (emptyhand) (clear ?b) (on-table ?b))" << std::endl
     << "    :effect" << std::endl;
  if (costTower>0)
      os << "     (and" << std::endl
	 << "      (decrease reward " << costTower << ")" << std::endl;
  os << "      (probabilistic 3/4 (and (holding ?b) (not (emptyhand)) (not (on-table ?b))))" << std::endl;
  if (costTower>0)
      os << "     )" << std::endl;
  os << "  )" << std::endl
     << "  (:action put-on-block" << std::endl
     << "    :parameters (?b1 ?b2 - block)" << std::endl
     << "    :precondition (and (holding ?b1) (clear ?b1) (clear ?b2) (not (= ?b1 ?b2)))" << std::endl
     << "    :effect (probabilistic 3/4 (and (on ?b1 ?b2) (emptyhand) (clear ?b1) (not (holding ?b1)) (not (clear ?b2)))" << std::endl
     << "                           1/4 (and (on-table ?b1) (emptyhand) (clear ?b1) (not (holding ?b1))))" << std::endl
     << "  )" << std::endl
     << "  (:action put-down" << std::endl
     << "    :parameters (?b - block)" << std::endl
     << "    :precondition (and (holding ?b) (clear ?b))" << std::endl
     << "    :effect (and (on-table ?b) (emptyhand) (clear ?b) (not (holding ?b)))" << std::endl
     << "  )" << std::endl
     << "  (:action pick-tower" << std::endl
     << "    :parameters (?b1 ?b2 ?b3 - block)" << std::endl
     << "    :precondition (and (emptyhand) (clear ?b1) (on ?b1 ?b2) (on ?b2 ?b3))" << std::endl
     << "    :effect" << std::endl
     << "      (probabilistic 1/10 (and (holding ?b2) (clear ?b3) (not (emptyhand)) (not (on ?b2 ?b3))))" << std::endl
     << "  )" << std::endl
     << "  (:action put-tower-on-block" << std::endl
     << "    :parameters (?b1 ?b2 ?b3 - block)" << std::endl
     << "    :precondition (and (holding ?b2) (on ?b1 ?b2) (clear ?b3) (not (= ?b1 ?b3)))" << std::endl
     << "    :effect (probabilistic 1/10 (and (on ?b2 ?b3) (emptyhand) (not (holding ?b2)) (not (clear ?b3)))" << std::endl
     << "                           9/10 (and (on-table ?b2) (emptyhand) (not (holding ?b2))))" << std::endl
     << "  )" << std::endl
     << "  (:action put-tower-down" << std::endl
     << "    :parameters (?b1 ?b2 - block)" << std::endl
     << "    :precondition (and (holding ?b2) (on ?b1 ?b2))" << std::endl
     << "    :effect (and (on-table ?b2) (emptyhand) (not (holding ?b2)))" << std::endl
     << "  )" << std::endl;

  os << ")" << std::endl;
}

void
print_state( std::ostream &os, std::vector<int> &state )
{
  std::set<int> nclear;

  os << "(emptyhand)";
  for( unsigned int i = 0; i < state.size(); ++i ) {
    if( state[i] > 0 )  {
      os << " (on b" << 1+i << " b" << state[i] << ")";
      nclear.insert(state[i]);
    }
    else
      os << " (on-table b" << 1+i << ")";
  }
  for( unsigned int i = 1; i <= state.size(); ++i ) {
    if( nclear.find(i) == nclear.end() )
      os << " (clear b" << i << ")";
  }
}

void
print_problem( std::ostream &os, const char *name, int size, std::vector<int> &init, std::vector<int> &goal )
{
  os << "(define (problem prob_bw_" << size << "_" << name << ")" << std::endl
     << "  (:domain prob_bw)" << std::endl
     << "  (:objects";
  for( int i = 0; i < size; ++i ) os << " b" << 1+i;
  os << " - block)" << std::endl;

  os << "  (:init ";
  print_state( os, init );
  os << ")" << std::endl
     << "  (:goal (and ";
  print_state( os, goal );
  os << "))" << std::endl;
  os << ")" << std::endl;
}

/* Program options. */
static struct option long_options[] = {
    { "blockPickUpCost", required_argument, 0, 'c' },
    { "towerPickUpCost", required_argument, 0, 'C' },
    { "help", no_argument, 0, '?' },
    { 0, 0, 0, 0 }
};
static const char OPTION_STRING[] = "c:C:g:?";

/* Displays help. */
static void display_help()
{
  std::cout << "typical usage: blocksworld [options] <name-suffix> `./bwstates -s 2 -n <#-blocks> -r <seed>`" << std::endl
            << "options:" << std::endl
	    << "  -c c, --blockPickUpCost=c\t"
	    << "cost of pick up action (for a block)" << std::endl
	    << "  -C C, --towerPickUpCost=C\t"
	    << "cost of pick up action (for a tower)" << std::endl
	    << "  -?     --help\t\t"
            << "display this help and exit" << std::endl
            << endl
            << endl
            << "Report bugs to < ... >." << endl;
}

int
main( int argc, char **argv )
{
    costBlock = 0;
    costTower = 0;

  if( argc == 1 ) {
      display_help();
      exit(-1);
  }

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
	  case 'c':
	      costBlock = atoi(optarg);
	      break;
	  case 'C':
	      costTower = atoi(optarg);
	      break;
	  case '?':
	      display_help();
	      return 0;
	  case ':':
	  default:
	      cerr << "Try `blocksworld --help' for more information."
		   << endl;
	      return -1;
	    }
      }

      const char *name = argv[optind];
      argv += optind+1;
      argc -= optind+1;
      
      int i, size;
      std::vector<std::vector<int> > states;
      while( argc > 0 ) {
	  std::vector<int> state;
	  size = atoi(*argv);
	  for( i = 0, --argc, ++argv; i < size; ++i, --argc, ++argv )
	      state.push_back(atoi(*argv));
	  states.push_back(state);
      }
      assert( states.size() >= 2 );
      
      print_domain( std::cout );
      std::cout << std::endl;
      print_problem( std::cout, name, size, states[0], states[1] );
      
  } catch (const exception& e) {
      cerr << endl << "blocksworld: " << e.what() << endl;
      return 1;
  } catch (...) {
      cerr << "blocksworld: fatal error" << endl;
      return -1;
  }
  

  return(0);
}

