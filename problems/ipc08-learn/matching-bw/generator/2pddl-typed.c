


/*********************************************************************
 * (C) Copyright 2001 Albert Ludwigs University Freiburg
 *     Institute of Computer Science
 *
 * All rights reserved. Use of this software is permitted for
 * non-commercial research purposes, and it may be copied only
 * for that use.  All copies must include this copyright message.
 * This software is made available AS IS, and neither the authors
 * nor the  Albert Ludwigs University Freiburg make any warranty
 * about the software or its performance.
 *********************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/timeb.h>
#include <math.h>


#define MAX_LENGTH 256
#define MAX_SAMPLE 50

/* data structures ... (ha ha)
 */
typedef unsigned char Bool;
#define TRUE 1
#define FALSE 0



/* commands
 */
char *gdata;
char *pname = NULL;
int gn;  // number of blocks
int fs;  // full states (1) or not (0)?



/* command line
 */
void usage( void );
Bool process_command_line( int argc, char *argv[] );



int main( int argc, char *argv[] )

{

  FILE *data;

  int i, j;
  char c;

  int *initial, *goal, *goal_blocked_above;

  gdata = NULL;
  gn = -1;
  fs = 0;

  if ( !process_command_line( argc, argv ) ) {
    usage();
    exit( 1 );
  }

  initial = ( int * ) calloc( gn + 1, sizeof( int ) );
  goal = ( int * ) calloc( gn + 1, sizeof( int ) );
  goal_blocked_above = ( int * ) calloc( gn + 1, sizeof( int ) );

  if ( (data = fopen( gdata, "r")) == NULL ) {
    printf("\n\nCannot open file %s.\n\n", gdata);
    exit( 1 );
  }
  while ( (c = fgetc( data )) != '\n' );
  for ( i = 0; i < gn; i++ ) {
    fscanf( data, " %d", &(initial[i + 1]) );
  }
  while ( (c = fgetc( data )) != '\n' );
  while ( (c = fgetc( data )) != '\n' );
  while ( (c = fgetc( data )) != '\n' );
  for ( i = 0; i < gn; i++ ) {
    fscanf( data, " %d", &(goal[i + 1]) );
  }

  if ( 0 ) {
    for ( i = 1; i < gn + 1; i++ ) {
      printf("\nBlock %d: %d -> %d", i, initial[i], goal[i]);
    }
    printf("\n\n");
  }

  if (pname != NULL) {
    printf("\n\n(define (problem %s)", pname);
  } else {
    printf("\n\n(define (problem matching-bw-typed-n%d-f%d)", gn, fs);
  }
  printf("\n(:domain matching-bw-typed)");
  printf("\n(:requirements :typing)");
  printf("\n(:objects h1 h2 - hand ");
  for ( i = 0; i < gn; i++ ) printf("b%d ", i+1);
  printf(" - block)");
  printf("\n(:init");
  printf("\n (empty h1)");
  printf("\n (empty h2)");
  printf("\n (hand-positive h1)");
  printf("\n (hand-negative h2)");
  for ( i = 1; i < gn + 1; i++ ) {
          printf("\n (solid b%d)",i);
    if (i <= gn/2)
                printf("\n (block-positive b%d)",i);
        else
                printf("\n (block-negative b%d)",i);
    if ( initial[i] == 0 ) {
      printf("\n (on-table b%d)", i);
    } else {
      printf("\n (on b%d b%d)", i, initial[i]);
    }
  }
  for ( i = 1; i < gn + 1; i++ ) {
    for ( j = 1; j < gn + 1; j++ ) {
      if ( j == i ) continue;
      if ( initial[j] == i ) break;
    }
    if ( j < gn + 1 ) continue;
    printf("\n (clear b%d)", i);
  }
  printf("\n)");
  printf("\n(:goal");
  printf("\n(and");
  for ( i = 1; i < gn + 1; i++ ) {
    // goal[i] has something above it (namely, block i)
    goal_blocked_above[goal[i]] = i;
  }
  for ( i = 1; i < gn + 1; i++ ) {
    if ( goal[i] == 0 ) {
      if (fs) {
        // full states, so stipulate that this block must be on table
        printf("\n (on-table b%d)", i);
      }
    } else {
      printf("\n (on b%d b%d)", i, goal[i]);
    }
    if ( fs && !goal_blocked_above[i] ) {
      // again, full states need to tell us which blocks are clear
      printf("\n (clear b%d)", i);
    }
  }
  if ( fs ) {
    // finally full states tell us where gripper is
    printf("\n (empty h1) (empty h2)");
  }
  printf(")\n)\n)\n\n\n");



  exit( 0 );

}



void usage( void )

{

  printf("\nusage:\n");

  printf("\nOPTIONS   DESCRIPTIONS\n\n");
  printf("-d <str>    name of data file\n\n");
  printf("-n <num>    number of blocks\n\n");
  printf("-f <0/1>    whether to use full states\n\n");
  printf("-p <name>   problem name\n\n");

}



Bool process_command_line( int argc, char *argv[] )

{

  char option;

  while ( --argc && ++argv ) {
    if ( *argv[0] != '-' || strlen(*argv) != 2 ) {
      return FALSE;
    }
    option = *++argv[0];
    switch ( option ) {
    default:
      if ( --argc && ++argv ) {
        switch ( option ) {
        case 'd':
          gdata = ( char * ) calloc( MAX_LENGTH, sizeof( char ) );
          strncpy( gdata, *argv, MAX_LENGTH );
          break;
        case 'n':
          sscanf( *argv, "%d", &gn );
          break;
        case 'f':
          sscanf( *argv, "%d", &fs );
          fs = fs != 0;
          break;
        case 'p':
          // assume argv will never die (I hope...)
          pname = calloc(strlen(*argv) + 1, sizeof(char));
          strcpy(pname, *argv);
          break;
        default:
          printf( "\n\nunknown option: %c entered\n\n", option );
          return FALSE;
        }
      } else {
        return FALSE;
      }
    }
  }

  if ( !gdata ) {
    return FALSE;
  }

  return TRUE;

}
