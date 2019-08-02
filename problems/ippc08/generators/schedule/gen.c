
/* Created by Robert Givan and Sungwook Yoon */
/* Modified by Olivier Buffet for the purpose of IPC-2008 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_PREDICATES 10
#define MAX_ARITY 5


int gnum_classes = 2;
int gnum_units = 2;
int gnum_plan_length = 1000;
int gnum_prob_c0 = 3;

int gnum_packets;


void print_domain() 
{
  int i, j;
  int cur_prob;
  
  printf("(define (domain schedule)\n");
  printf(" (:requirements :rewards :adl :probabilistic-effects)\n");
  printf(" (:types packet class timestep phase packetstatus)\n");

  /* constants printing */
  printf(" (:constants ");
  for(i=0; i< gnum_classes; i++) {
    printf("%c%d ", 'C', i);
  }
  printf("- class ");
  for(i=0; i< gnum_units+1; i++) {
    printf("%c%d ", 'U',i);
  }
  printf("- timestep\n");
  printf(" Arrivals-and-updating Cleanup-and-service - phase\n");
  printf(" Available Queued - packetstatus)\n");
  
  
  /* predciates definition printing */
  printf(" (:predicates\n");
  printf("   (current-phase ?s - phase)\n");
  printf("   (packetclass ?p - packet ?c - class)\n");
  printf("   (timetolive ?p - packet ?t - timestep)\n");
  printf("   (status ?p - packet ?i - packetstatus)\n");  
  printf("   (processed-arrivals ?c - class)\n");
  printf("   (need-to-process-arrivals ?c - class)\n");
  printf("   (dropped ?p - packet)\n");
  printf("   (not-dropped ?p - packet)\n");
  printf("   (served ?c - class)\n");
  printf("   (alive)\n");
  printf(" )\n");

  /* action definition printing, changes with parameters */
  printf(" (:action process-arrivals\n");
  printf("  :parameters (?arriving-packet - packet ?c - class)\n");
  printf("  :precondition (and\n");
  printf("                    (current-phase Arrivals-and-updating)\n");
  printf("                    (status ?arriving-packet Available)\n");
  printf("                    (need-to-process-arrivals ?c))\n");
  printf("  :effect (and\n");
  printf("             (processed-arrivals ?c)\n");
  printf("             (not (need-to-process-arrivals ?c))\n");
  printf("             (probabilistic %d/1000\n", gnum_prob_c0);
  printf("                            (when (= ?c C0)\n");
  printf("                              (and (status ?arriving-packet Queued)\n");
  printf("                                   (not (status ?arriving-packet Available))\n");
  printf("                                   (packetclass ?arriving-packet ?c)\n");
  printf("                                   (timetolive ?arriving-packet U%d))))\n",
	 gnum_units);
  if(gnum_classes > 2) {
    printf("             (probabilistic 110/%d\n", 100*(gnum_classes-1));
    printf("                            (when (not (= ?c C0))\n");
    printf("                              (and (status ?arriving-packet Queued)\n");
    printf("                                   (not (status ?arriving-packet Available))\n");
    printf("                                   (packetclass ?arriving-packet ?c)\n");
    printf("                                   (timetolive ?arriving-packet U%d))))))\n",
	   gnum_units);
  } else if (gnum_classes == 2) {
    printf("             (probabilistic 100/100\n");
    printf("                            (when (not (= ?c C0))\n");
    printf("                              (and (status ?arriving-packet Queued)\n");
    printf("                                   (not (status ?arriving-packet Available))\n");
    printf("                                   (packetclass ?arriving-packet ?c)\n");
    printf("                                   (timetolive ?arriving-packet U%d))))))\n",
	   gnum_units);
    
  } else {
    printf("                                   ))\n");
  }
  

  printf("(:action time-update\n");
  printf(" :precondition (and\n");
  printf("                   (current-phase Arrivals-and-updating)\n");
  printf("                   (forall (?c - class) (processed-arrivals ?c)))\n");
  printf(" :effect (and \n");
  printf("            (not (current-phase Arrivals-and-updating))\n");
  printf("            (current-phase Cleanup-and-service)\n");
  printf("            (forall (?p - packet)\n");
  printf("                    (when (timetolive ?p U0)\n");
  printf("                          (and (dropped ?p) (not (not-dropped ?p)))))\n");
  for(i=1; i < gnum_units+1; i++) {
    printf("            (forall (?p - packet)\n");
    printf("                    (when (timetolive ?p U%d)\n",i);
    printf("                          (timetolive ?p U%d)))\n", i-1);
  }
  printf("                          \n");
  printf(" ))\n");

  
  printf("(:action reclaim-packet\n");
  printf("  :parameters (?p - packet ?c - class)\n");
  printf("  :precondition (and\n");
  printf("                    (current-phase Cleanup-and-service)\n");
  printf("                    (packetclass ?p ?c))\n");
  printf("  :effect (and\n");
  printf("              (not (dropped ?p))\n");
  printf("              (not-dropped ?p)\n");
  printf("              (forall (?u - timestep) (not (timetolive ?p ?u)))\n");
  printf("              (not (packetclass ?p ?c))\n");
  printf("              (status ?p Available)\n");
  printf("              (not (status ?p Queued))\n");
  cur_prob = 10;
  for(i=gnum_classes-1; i >= 0; i--) {
    if( i==0 ) {
      printf("               (probabilistic %d/100 (when (= ?c C%d) (not (alive))))\n",
	     70, i);
    } else {
      printf("               (probabilistic %d/2000 (when (= ?c C%d) (not (alive))))\n",
	     (cur_prob > 100)? 100:cur_prob, i);
      cur_prob = cur_prob *2;
    }
    
  }
  printf("))\n");
  
  
  printf("(:action packet-serve\n");
  printf("  :parameters (?p - packet ?c - class)\n");
  printf("  :precondition (and\n");
  printf("                    (current-phase Cleanup-and-service)\n");
  printf("                    (forall (?p1 - packet) (not-dropped ?p1))\n");
  printf("                    (packetclass ?p ?c))\n");
  printf("  :effect (and\n");
  printf("            (forall (?c1 - class) (not (processed-arrivals ?c1)))\n");
  printf("            (forall (?c1 - class) (need-to-process-arrivals ?c1))\n");
  printf("            (not (current-phase Cleanup-and-service))\n");
  printf("            (current-phase Arrivals-and-updating)\n");
  printf("            (served ?c)\n");
  printf("            (not (packetclass ?p ?c))\n");
  printf("            (forall (?u - timestep) (not (timetolive ?p ?u)))\n");
  printf("            (not (status ?p Queued))\n");
  printf("            (status ?p Available)))\n");
  
  
  printf("(:action serve-nothing\n");
  printf("  :precondition (and\n");
  printf("                    (current-phase Cleanup-and-service)\n");
  printf("                    (forall (?p1 - packet) (not-dropped ?p1)))\n");
  printf("  :effect (and    \n");
  printf("              (forall (?c1 - class) (not (processed-arrivals ?c1)))\n");
  printf("              (forall (?c1 - class) (need-to-process-arrivals ?c1))\n");
  printf("              (not (current-phase Cleanup-and-service))\n");
  printf("              (current-phase Arrivals-and-updating))))\n");
}




void print_problem() 
{
  int i, j;
  
  printf("  (define (problem a-schedule-problem%d)\n", rand()%1000);
  printf("  (:domain schedule)\n");
  printf("  (:objects ");
  for(i=0; i< gnum_packets; i++) 
    printf("P%d ", i);
  printf(" - packet)\n");
  printf("  (:init \n");
  printf("         (alive)\n");
  printf("         (current-phase Arrivals-and-updating)\n");
  for(i=0; i< gnum_classes; i++)
    printf("         (need-to-process-arrivals C%d)\n",i);
  for(i=0; i< gnum_packets; i++)
    printf("         (status P%d Available)\n",i);
  for(i=0; i< gnum_packets; i++)
    printf("         (not-dropped P%d)\n",i);
  printf("  )\n");
  
  //printf("  (:goal (and (alive) (forall (?c - class) (served ?c)))))\n");
  printf("  (:goal (and (alive) (forall (?c - class) (served ?c))))\n");
  printf("  (:goal-reward 1)\n");
  printf("  (:metric maximize (reward))\n");
  printf("  )\n");
}


int process_command_line( int argc, char *argv[] )

{

  char option;
  
  while ( --argc && ++argv ) {
    if ( *argv[0] != '-' || strlen(*argv) != 2 ) {
      return 0;
    }
    option = *++argv[0];
    switch ( option ) {
    default:
      if ( --argc && ++argv ) {
	switch ( option ) {

	case 'c':
	  sscanf( *argv, "%d", &gnum_classes );
	  break;
	case 'u':
	  sscanf( *argv, "%d", &gnum_units);
	  break;
	case 'l':
	  sscanf( *argv, "%d", &gnum_plan_length);
	  break;
	  
	  
	default:
	  printf( "\nschedule: unknown option: %c entered\n\n", option );
	  return 0;
	}
      } else {
	return 0;
      }
    }
  }

  return 1;

}

int main(int argc, char *argv[]) 
{

 
 
  
  /* randomize */
  srand( (unsigned int) getpid() );

   /* read in command arguments */
  if(!process_command_line(argc, argv)) {
    printf("example: schedule -c 2 -u 2 -l 1000\n");
    exit(1);
  }

  if( gnum_plan_length < 15 ) {
    printf("increase plan length to more than 15");
    exit(2);
  }

  gnum_packets = gnum_classes + gnum_classes * gnum_units;

  gnum_prob_c0 = (int) ((float) 1000 * ((float) 1 /
					( ((float) (gnum_plan_length - 5*(gnum_classes -1) ))
					  / (float) (gnum_classes + 2)) ) );
    
  print_domain();
  /*create_a_problem();*/
  print_problem();

}
