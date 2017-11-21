#!/usr/local/bin/perl -w
use warnings;

if ($#ARGV + 1 != 2 && $#ARGV + 1  != 4) {
  die usage();
}
elsif($#ARGV + 1 == 2 && ($ARGV[0] ne "-c")){
  die usage();
}
elsif( $#ARGV + 1 == 4 && ( !($ARGV[0] eq "-v" && $ARGV[2] eq "-c") && !($ARGV[0] eq "-c" && $ARGV[2] eq "-v")) ) {
  die usage();
}

my $num_curbs;
my $num_cars;

if($ARGV[0] eq "-c"){
  $num_curbs = $ARGV[1];                        # number of curbs
  $num_cars = $ARGV[3];                         # number of cars
}
else{
  $num_curbs = $ARGV[3];                        # number of curbs
  $num_cars = $ARGV[1];                         # number of cars
}

my $max_cars = 2 * $num_curbs - 2;
$num_cars = $max_cars unless $num_cars;  # default to all but 2 cars
local (@curbs);
local (@cars);

die "The maximum number of cars for $num_curbs curbs is $max_cars" 
  if $num_cars > $max_cars;

for my $i (0..$num_curbs - 1) {
  $curbs[$i] = "curb_".pad($i, length($num_curbs-1));
}

for my $i (0..$num_cars - 1) {
  $cars[$i] = "car_".pad($i, length($num_cars-1));
}

local %init_cfg = init_config();
local %goal_cfg = goal_config();
print define();

sub usage{
	return "usage: $0 -c number_of_curbs [-v number_of_vehicles]\n";
}

sub define {
	my $s = "";
	$s .= "(define \n"; 
  $s .= "  (problem parking1)\n";
	$s .= "  (:domain parking-lot)\n";
	$s .= "  (:requirements :strips)\n";
  $s .= objects();
  $s .= init();
  $s .= goal();
	$s .= ")\n";
	$s .= "; =========== INIT =========== \n";
  $s .= visual_layout(%init_cfg);
  $s .= "; ========== /INIT =========== \n";
  $s .= "\n";
  $s .= "; =========== GOAL =========== \n";
  $s .= visual_layout(%goal_cfg);
  $s .= "; =========== /GOAL =========== \n";
  return $s;
}

sub objects {
  my $s = "";
  $s .= "  (:objects\n";
	for my $car (@cars) {
	  $s .= "    $car\n";
	}
	for my $curb (@curbs) {
	  $s .= "    $curb\n";
	}
  $s .= "  )\n";
	return $s;
}
sub init {
  my $s = "  (:init\n";
	for my $car (@cars) {
	  $s .= "    (car $car)\n";
	}
	for my $curb (@curbs) {
	  $s .= "    (curb $curb)\n";
	}

  for $curb (@curbs) {
		if (defined($init_cfg{$curb})) {
		  my $at_curb = $init_cfg{$curb};
			$s .= "    (at-curb $at_curb)\n";
			$s .= "    (at-curb-num $at_curb $curb)\n";
			if (defined($init_cfg{$at_curb})) {
			  $car_behind = $init_cfg{$at_curb};
				$s .= "    (behind-car $car_behind $at_curb)\n";
				$s .= "    (clear $car_behind)\n";
			}
			else {
				$s .= "    (clear $at_curb)\n";
			}
		}
		else { # it's clear
		  $s .= "    (clear $curb)\n";
		}
	}

	for my $key (keys(%cfg)) {
	  $s .= "$key ===> " . $cfg{$key} ."\n";
	}

	$s .= "  )\n";
	return $s;
}
sub goal {
  my $s;
	##my $num_cars_against_curb = $num
	$s .= "  (:goal\n";
	$s .= "    (and\n";
  for $curb (@curbs) {
		if (defined($goal_cfg{$curb})) {
		  my $at_curb = $goal_cfg{$curb};
			$s .= "      (at-curb-num $at_curb $curb)\n";
			if (defined($goal_cfg{$at_curb})) {
			  $car_behind = $goal_cfg{$at_curb};
				$s .= "      (behind-car $car_behind $at_curb)\n";
			}
		}
	}

	$s .= "    )\n";
	$s .= "  )\n";
	return $s;
}

sub pad {           # add leading zeros
  my ($x, $len) = @_;
  while (length($x) < $len) {
	  $x = "0$x";
	}
	return $x;
}

# any valid config will have at least as many cars against curbs
# first, pick how many go against curb
#   assign those
#   assign the rest against those

sub goal_config {
  my $num_against_curb = ($#curbs > $#cars) ? $#cars+1 : $#curbs+1; # min
  my %against;
	for my $i (0..$num_against_curb - 1) {
	  $against{$curbs[$i]} = $cars[$i];
	}
	for $i ($num_against_curb .. $#cars) {
	  $against{$cars[$i - $num_against_curb]} = $cars[$i];
	}
	return %against;

}

sub init_config {
	use POSIX qw(ceil floor);
  my $min_against_curb = ceil(($#cars+1)/2); #random between min and number of curbs
  my $max_against_curb = ($#curbs > $#cars) ? $#cars+1 : $#curbs+1; # min

	my $num_against_curb = rand_between($min_against_curb, $max_against_curb);
	#print "$min_against_curb <= $num_against_curb <= $max_against_curb\n";

	my @shuffled_cars = shuffle(@cars);
  my %against;
	for my $i (0..$num_against_curb - 1) {
	  $against{$curbs[$i]} = $shuffled_cars[$i];
	}
	for $i ($num_against_curb .. $#cars) {
	  $against{$shuffled_cars[$i - $num_against_curb]} = $shuffled_cars[$i];
	}
	return %against;
}

sub shuffle {
  my (@a) = @_;
	srand;
	@new = ();
	while (@a) {
		push(@new, splice(@a, rand @a, 1));
	}
  return @new;
}

sub rand_between {  # inclusive
  my ($min, $max) = @_;
	return $min if $min == $max;
 ($min, $max) = ($max, $min)  if  $min > $max;
 return $min + int rand(1 + $max - $min);
}

sub visual_layout {
  my (%against) = @_;
	my $s;
  for $curb (@curbs) {
	  $s .= ";  $curb: ";
		if (defined($against{$curb})) {
		  $s .= $against{$curb}." ";
			$s .= (defined($against{$against{$curb}}) ? $against{$against{$curb}}." " : "");
		}
		$s .= "\n";
	}
	return $s;
}

