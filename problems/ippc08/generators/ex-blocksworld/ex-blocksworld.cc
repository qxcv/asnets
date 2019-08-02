#include <iostream>
#include <vector>
#include <set>
#include <assert.h>

void
print_state( std::ostream &os, std::vector<int> &state, int blocks, bool init )
{
  std::set<int> nclear, blks;

  while( blks.size() < blocks ) {
    blks.insert( lrand48()%state.size() );
  }

  if( init ) os << " (emptyhand)";
  for( std::set<int>::const_iterator bi = blks.begin(); bi != blks.end(); ++bi ) {
    if( state[*bi] > 0 )  {
      os << " (on b" << 1+(*bi) << " b" << state[*bi] << ")";
      nclear.insert(state[*bi]);
    }
    else
      os << " (on-table b" << 1+(*bi) << ")";
  }

  if( init ) {
    for( int i = 1; i <= state.size(); ++i ) {
      if( nclear.find(i) == nclear.end() )
        os << " (clear b" << i << ")";
    }
    for( int i = 1; i <= state.size(); ++i ) {
      os << " (no-detonated b" << i << ")";
      os << " (no-destroyed b" << i << ")";
    }
    os << " (no-destroyed-table)";
  }
}

void
print_problem( std::ostream &os, const char *name, int size, int gblocks, std::vector<int> &init, std::vector<int> &goal )
{
  os << "(define (problem ex_bw_" << size << "_" << name << ")" << std::endl
     << "  (:domain exploding-blocksworld)" << std::endl
     << "  (:objects";
  for( int i = 0; i < size; ++i ) os << " b" << 1+i;
  os << " - block)" << std::endl;

  os << "  (:init";
  print_state( os, init, size, true );
  os << ")" << std::endl
     << "  (:goal (and";
  print_state( os, goal, gblocks, false );
  os << "  )" << std::endl << ")" << std::endl
     << "  (:goal-reward 1)" << std::endl
     << "  (:metric maximize (reward))" << std::endl
     << ")" << std::endl;

}

int
main( int argc, char **argv )
{
  if( argc == 1 ) {
    std::cout << "typical usage: ./ex-blocksworld <name-suffix> <blocks-in-goal> `./bwstates -s 2 -n <#-blocks> -r <seed>`" << std::endl;
    exit(-1);
  }

  const char *name = argv[1];
  --argc;
  ++argv;
  int gblocks = atoi(argv[1]);
  argv += 2;
  argc -= 2;

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
  print_problem( std::cout, name, size, gblocks, states[0], states[1] );
  return(0);
}

