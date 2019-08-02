/*
 * Copyright (C) 2004 Rutgers University
 *
 * Permission is hereby granted to distribute this software for
 * non-commercial research purposes, provided that this copyright
 * notice is included with any such distribution.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE
 * SOFTWARE IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU
 * ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.
 *
 * Author: John Asmuth jasmuth@cs.rutgers.edu
 * Modified by: Olivier Buffet firstname.lastname@loria.fr (2008)
 */

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <math.h>

using namespace std;

class City {
private:
  string name;
  int x;
  int y;
  bool airport;
  vector<City*> driveDsts;
  vector<City*> flyDsts;
public:
  City(string name, int x, int y);
  string getName() {return name;}
  int getX() {return x;}
  int getY() {return y;}
  vector<City*>& getDriveDsts() {return driveDsts;}
  vector<City*>& getFlyDsts() {return flyDsts;}
  bool hasDriveDst(City* c) {
    for (int i=0; i<driveDsts.size(); i++)
      if (driveDsts[i] == c)
	return true;
    return false;
  }
  void addDriveDst(City* c);
  void addFlyDst(City* c) {flyDsts.push_back(c);}
  bool hasAirport() {return airport;}
  void setAirport(bool a) {airport=a;}
  void clearDsts();
  float distance(City* c);
  bool operator==(City& c) {return name==c.getName();}
};

class CitySet {
private:
  vector<City*> cities;
  int indexOf(City* c);
public:
  CitySet() {}
  CitySet(vector<City*>& v);
  int size() {return cities.size();}
  void push(City* c)  {cities.push_back(c);}
  City* pop();
  bool contains(City* c) {return indexOf(c)!=-1;}
  void remove(City* c);
};

struct Box {
  string name;
  City *from;
  City *to;
};

class LogisticsDomain {
private:
  string domain_name;
  vector<City*> cities;
  int drive_cost;
  int fly_cost;
  int reward;
  int truck_load_cost;
  int truck_unload_cost;
  int plane_load_cost;
  int plane_unload_cost;
  bool do_rewards;
  bool preconditions;
public:
  LogisticsDomain(string n) {domain_name=n;}
  string getName() {return domain_name;}
  void setDriveCost(int r) {drive_cost=r;}
  void setFlyCost(int r) {fly_cost=r;}
  void setReward(int r) {reward=r;}
  void setTruckLoadCost(int r) {truck_load_cost=r;}
  void setTruckUnloadCost(int r) {truck_unload_cost=r;}
  void setPlaneLoadCost(int r) {plane_load_cost=r;}
  void setPlaneUnloadCost(int r) {plane_unload_cost=r;}
  void setDoRewards(int d) {do_rewards = d;}
  void setPreconditions(bool p) {preconditions = p;}

  void addCity(City* c) {cities.push_back(c);}
  vector<City*>& getCities() {return cities;}
  void crunchDrive();
  void findAirports();

  void printPPDDL(ostream& os);
  void printCities(ostream& os);
  void printLoadUnload(ostream& os);
  void printDriveAction(ostream& os);
  void printFlyAction(ostream& os);
};

class LogisticsProblem {
private:
  string problem_name;
  vector<Box> boxes;
  vector<City*> airports;
  LogisticsDomain *domain;
  int goal_reward;
public:
  LogisticsProblem(string n, LogisticsDomain *domain);
  void addBox(Box b) {boxes.push_back(b);}
  void setGoalReward(int gr) {goal_reward = gr;}
  void printPPDDL(ostream& os);

};

////

bool vector_contains(vector<City*>& v, City* c) {
  for (int i=0; i<v.size(); i++)
    if (v[i] == c)
      return true;
  return false;
}

City::City(string name, int x, int y) {
  this->name = name;
  this->x = x;
  this->y = y;
  this->airport = false;
}

void City::addDriveDst(City* c) {
  if (!vector_contains(driveDsts, c))
    driveDsts.push_back(c);
}

float City::distance(City* c) {
  int dx = c->getX()-x;
  int dy = c->getY()-y;
  return sqrt(1.0f*(dx*dx+dy*dy));
}

void City::clearDsts() {
  driveDsts.clear();
  flyDsts.clear();
}

CitySet::CitySet(vector<City*>& v) {
  cities = v;
}

City* CitySet::pop() {
  City* c = cities[0];
  cities.erase(cities.begin());
  return c;
}

int CitySet::indexOf(City* c) {
  for (int i=0; i<cities.size(); i++)
    if (c == cities[i])
      return i;
  return -1;
}

void CitySet::remove(City* c) {
  int index = indexOf(c);
  if (index != -1)
    cities.erase(cities.begin()+index);
}

void LogisticsDomain::crunchDrive() {
  for (int i=0; i<cities.size(); i++) {
    cities[i]->clearDsts();
  }
  for (int i=0; i<cities.size(); i++) {
    vector<City*> dsts;
    for (int j=0; j<cities.size(); j++) {
      if (i == j)
        continue;

      if (dsts.size() < 4)
        dsts.push_back(cities[j]);
      else {
        dsts[3] = cities[j];
      }

      for (int k=0; k<dsts.size(); k++) {
        for (int l=k+1; l<dsts.size(); l++) {
          float dk = cities[i]->distance(dsts[k]);
          float dl = cities[i]->distance(dsts[l]);
          if (dl < dk) {
            City *tmp = dsts[k];
            dsts[k] = dsts[l];
            dsts[l] = tmp;
          }
        }
      }
    }
    //add the three closest cities
    for (int k=0; k<3; k++) {
      cities[i]->addDriveDst(dsts[k]);
      //now make sure its symetric
      dsts[k]->addDriveDst(cities[i]);
    }
  }
}

void LogisticsDomain::findAirports() {
  CitySet unvisited(cities);
  CitySet airported;
  while (unvisited.size()) {
    CitySet pass;
    City* firstCity = unvisited.pop();
    airported.push(firstCity);
    pass.push(firstCity);
    while (pass.size()) {
      City *c = pass.pop();
      vector<City*> dsts = c->getDriveDsts();
      for (int i=0; i<dsts.size(); i++) {
        if (unvisited.contains(dsts[i])) {
          pass.push(dsts[i]);
          unvisited.remove(dsts[i]);
        }
      }
    }
  }

  int airport_count = 0;

  for (int i=0; i<cities.size(); i++)
    if (airported.contains(cities[i])) {
      cities[i]->setAirport(true);
      airport_count++;
    }

  //always at least two airports
  //cities[0] is automatically an airport, due to the nature of the algorithm
  if (airport_count==1) {
    cities[1]->setAirport(true);
  }

  for (int i=0; i<cities.size(); i++) {
    if (!cities[i]->hasAirport())
      continue;
    for (int j=0; j<cities.size(); j++) {
      if (i == j)
        continue;
      if (cities[j]->hasAirport())
        cities[i]->addFlyDst(cities[j]);
    }
  }
}

void LogisticsDomain::printPPDDL(ostream& os) {
  cout << "(define (domain " << domain_name << ")" << endl
       << " (:requirements :typing :equality :disjunctive-preconditions" << endl
       << "                :probabilistic-effects :existential-preconditions" << endl
       << "                :conditional-effects :negative-preconditions" << endl
       << "                :universal-preconditions";
  if (do_rewards)
    cout << " :rewards";
  cout << ")" << endl
       << " (:types city box truck plane)" << endl;
  //printCities(os);
  cout << " (:predicates (box-at-city ?b - box ?c - city)" << endl
       << "              (truck-at-city ?t - truck ?c - city)" << endl
       << "              (box-on-truck ?b - box ?t - truck)" << endl
       << "              (plane-at-city ?p - plane ?c - city)" << endl
       << "              (box-on-plane ?b - box ?p - plane)" << endl
       << "              (destination ?b - box ?dst - city)" << endl
       << "              (can-drive ?src - city ?dst - city)" << endl
       << "              (wrong-drive1 ?src - city ?wrongdst - city)" << endl
       << "              (wrong-drive2 ?src - city ?wrongdst - city)" << endl
       << "              (wrong-drive3 ?src - city ?wrongdst - city)" << endl
       << "              (can-fly ?src - city ?dst - city))" << endl;
  printLoadUnload(os);
  printDriveAction(os);
  printFlyAction(os);
  cout << ")" << endl;
}

void LogisticsDomain::printCities(ostream& os) {

  cout << " (:constants";
  for (int i=0; i<cities.size(); i++) {
    if (i)
      cout << "            ";
    cout << " " << cities[i]->getName() << " - city";
    if (i == cities.size()-1)
      cout << ")";
    cout << endl;
  }
}

void LogisticsDomain::printLoadUnload(ostream& os) {

  cout << " (:action load-box-on-truck-in-city" << endl
       << "  :parameters (?b - box ?t - truck ?c - city)" << endl;
  if (preconditions) {
    cout << "  :precondition (and (box-at-city ?b ?c)" << endl
         << "                     (not (destination ?b ?c))" << endl
         << "                     (truck-at-city ?t ?c)" << endl
         << "                )" << endl;
		 }
  cout << "  :effect ";
  if (!preconditions) {
    cout << "(when (and (box-at-city ?b ?c)" << endl
         << "                               (not (destination ?b ?c))" << endl
         << "                               (truck-at-city ?t ?c)" << endl
         << "                          )" << endl;
	}
  cout << "(and ";
  if (truck_load_cost) {
    cout << "(decrease (reward) " << truck_load_cost << ")" << endl
         << "               ";
  }
  cout << "(box-on-truck ?b ?t)" << endl
       << "               (not (box-at-city ?b ?c))" << endl;
  if (!preconditions)
  	cout << "                )" << endl;
  cout << "          )" << endl
       << " )" << endl

       << " (:action unload-box-from-truck-in-city" << endl
       << "  :parameters (?b - box ?t - truck ?c - city)" << endl;
  if (preconditions) {
    cout << "  :precondition (and (box-on-truck ?b ?t)" << endl
         << "                     (truck-at-city ?t ?c)" << endl
         << "                )" << endl;
  }
  cout << "  :effect ";
  if (!preconditions) {
		cout << "(when (and (box-on-truck ?b ?t)" << endl
         << "                               (truck-at-city ?t ?c)" << endl
         << "                          )" << endl
         << "                ";
	}
  cout << "(and ";
  if (truck_unload_cost) {
    cout << "(decrease (reward) " << truck_unload_cost << ")" << endl
         << "               ";
  }
  cout << "(box-at-city ?b ?c)" << endl
       << "               (not (box-on-truck ?b ?t))" << endl;
  if (reward)
    cout << "               (when (destination ?b ?c)" << endl
         << "                     (increase (reward) " << reward << ")" << endl
         << "               )" << endl;
  if (!preconditions)
  	cout << "                )" << endl;
  cout << "          )" << endl
       << " )" << endl;

  cout << " (:action load-box-on-plane-in-city" << endl
       << "  :parameters (?b - box ?p - plane ?c - city)" << endl;
  if (preconditions) {
    cout << "  :precondition (and (box-at-city ?b ?c)" << endl
         << "                     (not (destination ?b ?c))" << endl
         << "                     (plane-at-city ?p ?c)" << endl
         << "                )" << endl;
	}
  cout << "  :effect ";
  if (!preconditions) {
    cout << "(when (and (box-at-city ?b ?c)" << endl
         << "                     (not (destination ?b ?c))" << endl
         << "                     (plane-at-city ?p ?c)" << endl
         << "                )" << endl
         << "                ";
	}
  cout << "(and ";
  if (plane_load_cost) {
    cout << "(decrease (reward) " << plane_load_cost << ")" << endl
         << "               ";
  }
  cout << "(box-on-plane ?b ?p)" << endl
       << "               (not (box-at-city ?b ?c))" << endl;
  if (!preconditions) {
		cout << "                )" << endl;
	}
  cout << "          )" << endl
       << " )" << endl
       << " (:action unload-box-from-plane-in-city" << endl
       << "  :parameters (?b - box ?p - plane ?c - city)" << endl;
  if (preconditions) {
		cout << "  :precondition (and (box-on-plane ?b ?p)" << endl
				 << "                     (plane-at-city ?p ?c)" << endl
				 << "                )" << endl;
	}
  cout << "  :effect ";
  if (!preconditions) {
		cout << "(when (and (box-on-plane ?b ?p)" << endl
				 << "                     (plane-at-city ?p ?c)" << endl
				 << "                )" << endl
         << "                ";
	}
  cout << "(and ";
  if (plane_unload_cost) {
    cout << "(decrease (reward) " << plane_unload_cost << ")" << endl
         << "               ";
  }
  cout << "(box-at-city ?b ?c)" << endl
       << "               (not (box-on-plane ?b ?p))" << endl;
  if (reward)
    cout << "               (when (destination ?b ?c)" << endl
	 << "                     (increase (reward) " << reward << ")" << endl
	 << "               )" << endl;
  if (!preconditions) {
		cout << "                )" << endl;
	}
  cout << "          )" << endl
       << " )" << endl;
}

void LogisticsDomain::printDriveAction(ostream& os) {
  cout << " (:action drive-truck" << endl
       << "  :parameters (?t - truck ?src - city ?dst - city)" << endl;
  if (preconditions) {
		cout << "  :precondition (and (truck-at-city ?t ?src)" << endl
				 << "                     (can-drive ?src ?dst)" << endl
				 << "                )" << endl;
	}
  cout << "  :effect ";
  if (!preconditions) {
		cout << "(when (and (truck-at-city ?t ?src)" << endl
				 << "                     (can-drive ?src ?dst)" << endl
				 << "                )" << endl
				 << "          ";
	}
  cout << "(and ";
  if (drive_cost) {
    cout << "(decrease (reward) " << drive_cost << ")" << endl
         << "               ";
  }
  cout << "(not (truck-at-city ?t ?src))" << endl
       << "               (probabilistic" << endl
       << "                0.2 (forall (?wrongdst1 - city)" << endl
       << "                    (when (wrong-drive1 ?src ?wrongdst1)" << endl
       << "                    (forall (?wrongdst2 - city)" << endl
       << "                    (when (wrong-drive2 ?src ?wrongdst2)" << endl
       << "                    (forall (?wrongdst3 - city)" << endl
       << "                    (when (wrong-drive3 ?src ?wrongdst3)" << endl
       << "                     (probabilistic" << endl
       << "                      1/3 (truck-at-city ?t ?wrongdst1)" << endl
       << "                      1/3 (truck-at-city ?t ?wrongdst2)" << endl
       << "                      1/3 (truck-at-city ?t ?wrongdst3)" << endl
       << "                     )" << endl
       << "                    ))))))" << endl
       << "                0.8 (truck-at-city ?t ?dst)" << endl
       << "               )" << endl;
  if (!preconditions) {
		cout << "          )" << endl;
	}
  cout << "          )" << endl
       << " )" << endl;
}

void LogisticsDomain::printFlyAction(ostream& os) {
  cout << " (:action fly-plane" << endl
       << "  :parameters (?p - plane ?src - city ?dst - city)" << endl;
  if (preconditions) {
		cout << "  :precondition (and (plane-at-city ?p ?src)" << endl
				 << "                     (can-fly ?src ?dst)" << endl
				 << "                )" << endl;
	}
  cout << "  :effect ";
  if (!preconditions) {
		cout << "(when (and (plane-at-city ?p ?src)" << endl
				 << "                     (can-fly ?src ?dst)" << endl
				 << "                )" << endl;
	}
  cout << "(and ";
  if (fly_cost) {
    cout << "(decrease (reward) " << fly_cost << ")" << endl
	 << "              ";
  }
  cout << "(not (plane-at-city ?p ?src))" << endl
       << "               (plane-at-city ?p ?dst)" << endl
       << "          )" << endl;
  if (!preconditions)
    cout << " )" << endl;
  cout << " )" << endl;
}

LogisticsProblem::LogisticsProblem(string n, LogisticsDomain *domain) {
  problem_name = n;
  this->domain = domain;
  vector<City*> cities = domain->getCities();
  for (int i=0; i<cities.size(); i++) {
    if (cities[i]->hasAirport())
      airports.push_back(cities[i]);
  }
}

void LogisticsProblem::printPPDDL(ostream& os) {
  os << "(define" << endl
     << " (problem " << problem_name << ")" << endl
     << "  (:domain " << domain->getName() << ")" << endl;
  os << "  (:objects";
  for (int i=0; i<boxes.size(); i++) {
    if (i)
      os << "           ";
    os << " " << boxes[i].name << " - box" << endl;
  }
  for (int i=0; i<airports.size(); i++) {
    char buf[16];
    sprintf(buf, "truck%d", i*2);
    os << "            " << buf << " - truck" << endl;
    sprintf(buf, "truck%d", i*2+1);
    os << "            " << buf << " - truck" << endl;
    sprintf(buf, "plane%d", i);
    os << "            " << buf << " - plane" << endl;
  }

  vector<City*>& cities = domain->getCities();
  for (int i=0; i<cities.size(); i++) {
    os << "            " << cities[i]->getName() << " - city" << endl;
  }

  os << "  )" << endl
     << "  (:init";
  for (int i=0; i<boxes.size(); i++) {
    if (i)
      os << "        ";
    os << " (box-at-city " << boxes[i].name << " "
       << boxes[i].from->getName() << ")" << endl
       << "         (destination " << boxes[i].name << " "
       << boxes[i].to->getName() << ")" << endl;
  }

  for (int i=0; i<airports.size(); i++) {
    char buf[16];
    sprintf(buf, "truck%d", i*2);
    os << "         (truck-at-city " << buf << " "
       << airports[i]->getName() << ")" << endl;
    sprintf(buf, "truck%d", i*2+1);
    os << "         (truck-at-city " << buf << " "
       << airports[i]->getName() << ")" << endl;
    sprintf(buf, "plane%d", i);
    os << "         (plane-at-city " << buf << " "
       << airports[i]->getName() << ")" << endl;
  }

  for (int i=0; i<cities.size(); i++) {
    City* src = cities[i];
    vector<City*>& driveDsts = src->getDriveDsts();
    for (int j=0; j<driveDsts.size(); j++) {
      City* dst = driveDsts[j];
      os << "         (can-drive " << src->getName() << " "
	 << dst->getName() << ")" << endl;
    }
    for (int j=0; j<3; j++) {
      os << "         (wrong-drive" << j+1 << " " << src->getName() << " "
	 << driveDsts[j]->getName() << ")" << endl;
    }
    vector<City*>& flyDsts = src->getFlyDsts();
    for (int j=0; j<flyDsts.size(); j++) {
      City* dst = flyDsts[j];
      os << "         (can-fly " << src->getName() << " "
	 << dst->getName() << ")" << endl;
    }
  }

  os << "  )" << endl
     << "  (:goal (forall (?b - box)" << endl
     << "                 (exists (?c - city)" << endl
     << "                         (and (destination ?b ?c)" << endl
     << "                              (box-at-city ?b ?c)" << endl
     << "                         )" << endl
     << "                 )" << endl
     << "         )" << endl
     << "  )" << endl;
  if (goal_reward)
    cout << "  (:goal-reward " << goal_reward << ")" << endl;
  cout << "  (:metric maximize (reward))" << endl
       << ")" << endl;
}

int random_int(int range)
{
  return rand()%range;
}


void usage(char *name)
{
  int blank_len = strlen(name)+8;
  char *blank = new char[blank_len];
  for (int i=0; i<blank_len; i++)
    blank[i] = ' ';
  blank[blank_len] = 0;
  cerr << "usage: " << name << " [-h]" << endl
       << blank << "[-b box-count]" << endl
       << blank << "[-c city-count]" << endl
       << blank << "[-dc drive-cost]" << endl
       << blank << "[-fc fly-cost]" << endl
       << blank << "[-dr delivery-reward]" << endl
       << blank << "[-tlc truck-load-cost]" << endl
       << blank << "[-tuc truck-unload-cost]" << endl
       << blank << "[-plc plane-load-cost]" << endl
       << blank << "[-puc plane-unload-cost]" << endl
       << blank << "[-gr goal-reward]" << endl
       << blank << "[-dn domain name]" << endl
       << blank << "[-pn problem name]" << endl;
}

int main(int argc, char **argv) {
  srand48(time(0));
  srand(time(0));

  int city_count = 5;
  int box_count = 10;
  int drive_cost = 0;
  int fly_cost = 0;
  int reward = 0;
  int truck_load_cost = 0;
  int truck_unload_cost = 0;
  int plane_load_cost = 0;
  int plane_unload_cost = 0;
  int goal_reward = 0;
  bool do_rewards = false;
  bool preconditions = false;
  char* domain_name = "boxworld";
  char* problem_name = "boxworld";

  for (int i=1; i<argc; i++) {
		if (!strcmp(argv[i], "-h")) {
		    usage(*argv);
		    return 1;
		}
		if (!strcmp(argv[i], "-pc")) {
			preconditions = true;
		}
    if (!strcmp(argv[i], "-pn")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      problem_name = argv[++i];
    }

    if (!strcmp(argv[i], "-dn")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      domain_name = argv[++i];
    }

    if (!strcmp(argv[i], "-c")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      city_count = atoi(argv[++i]);
      if (city_count < 4) {
	cerr << "city count must be at least 4" << endl;
	return 1;
      }
    }
    if (!strcmp(argv[i], "-b")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      box_count = atoi(argv[++i]);
    }
    if (!strcmp(argv[i], "-dc")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      drive_cost = atoi(argv[++i]);
      if (drive_cost)
	do_rewards = true;
    }
    if (!strcmp(argv[i], "-fc")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      fly_cost = atoi(argv[++i]);
      if (fly_cost)
	do_rewards = true;
    }
    if (!strcmp(argv[i], "-dr")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      reward = atoi(argv[++i]);
      if (reward)
	do_rewards = true;
    }
    if (!strcmp(argv[i], "-tlc")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      truck_load_cost = atoi(argv[++i]);
      if (truck_load_cost)
	do_rewards = true;
    }
    if (!strcmp(argv[i], "-tuc")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      truck_unload_cost = atoi(argv[++i]);
      if (truck_unload_cost)
	do_rewards = true;
    }
    if (!strcmp(argv[i], "-plc")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      plane_load_cost = atoi(argv[++i]);
      if (plane_load_cost)
	do_rewards = true;
    }
    if (!strcmp(argv[i], "-puc")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      plane_unload_cost = atoi(argv[++i]);
      if (plane_unload_cost)
	do_rewards = true;
    }
    if (!strcmp(argv[i], "-gr")) {
      if (i==argc-1) {
        usage(*argv);
        return 1;
      }
      goal_reward = atoi(argv[++i]);
      if (goal_reward)
	do_rewards = true;
    }
    if (!strcmp(argv[i], "-?")) {
      usage(*argv);
      return 0;
    }
  }

  LogisticsDomain ld(domain_name);

  ld.setDriveCost(drive_cost);
  ld.setFlyCost(fly_cost);
  ld.setReward(reward);
  ld.setTruckLoadCost(truck_load_cost);
  ld.setTruckUnloadCost(truck_unload_cost);
  ld.setPlaneLoadCost(plane_load_cost);
  ld.setPlaneUnloadCost(plane_unload_cost);
  ld.setDoRewards(do_rewards);
  ld.setPreconditions(preconditions);

  City **c = new City*[city_count];
  for (int i=0; i<city_count; i++) {
    char buf[16];
    sprintf(buf, "city%d", i);
    string n = buf;
    c[i] = new City(n, random_int(5*city_count), random_int(5*city_count));
    ld.addCity(c[i]);
  }

  ld.crunchDrive();
  ld.findAirports();

  LogisticsProblem lp(problem_name, &ld);
  for (int i=0; i<box_count; i++) {
    Box b;
    char buf[16];
    sprintf(buf, "box%d", i);
    b.name = buf;
    b.from = c[random_int(city_count)];
    b.to = b.from;
    while (b.to == b.from)
      b.to = c[random_int(city_count)];
    lp.addBox(b);
  }

  lp.setGoalReward(goal_reward);

  cout << ";; Generated by boxworld generator" << endl
       << ";; http://www.cs.rutgers.edu/~jasmuth/boxworld.tar.gz" << endl
       << ";; by John Asmuth (jasmuth@cs.rutgers.edu)" << endl << endl;

  ld.printPPDDL(cout);
  lp.printPPDDL(cout);

  for (int i=0; i<city_count; i++)
    delete c[i];
  delete c;

  return 0;
}
