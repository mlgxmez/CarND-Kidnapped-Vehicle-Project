/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if(!is_initialized){
		
		num_particles = 100;

		default_random_engine gen;

		double std_x = std[0];
		double std_y = std[1];
		double std_theta = std[2];

		normal_distribution<double> dist_x(x, std_x);
		normal_distribution<double> dist_y(y, std_y);
		normal_distribution<double> dist_theta(theta, std_theta);

		particles.resize(num_particles);

		for(int i = 0; i < num_particles; i++){
			double sample_x, sample_y, sample_theta;

			sample_x = dist_x(gen);
			sample_y = dist_y(gen);
			sample_theta = dist_theta(gen);

		// Assign coordiantes to each particle
			particles[i].x = sample_x;
			particles[i].y = sample_y;
			particles[i].theta = sample_theta;
			particles[i].weight = 1.0;
		}
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	
	double pred_x, pred_y, pred_theta;
	double sample_x, sample_y, sample_theta;
	
	for(int j=0; j < num_particles; j++){
		normal_distribution<double> dist_x(0, std_x);
		normal_distribution<double> dist_y(0, std_y);
		normal_distribution<double> dist_theta(0, std_theta);

		if(fabs(yaw_rate)>0.00001){
			particles[j].x +=  velocity/yaw_rate*(sin(particles[j].theta + yaw_rate*delta_t) - sin(particles[j].theta));
			particles[j].y +=  velocity/yaw_rate*(cos(particles[j].theta) - cos(particles[j].theta + yaw_rate*delta_t));
			particles[j].theta += yaw_rate*delta_t;
		} else {
			particles[j].x += velocity*delta_t*cos(particles[j].theta);
			particles[j].y += velocity*delta_t*sin(particles[j].theta);
		}

		// Add noise
		particles[j].x += dist_x(gen);
		particles[j].y += dist_y(gen);
		particles[j].theta += dist_theta(gen);		
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	for(int i=0; i < num_particles; i++){
		std::vector<LandmarkObs> observations_transf;
		particles[i].weight = 1.0f;
		
		for(LandmarkObs o: observations){
			// Transform each observation
			LandmarkObs obs_t;
			obs_t.id = o.id;
			obs_t.x = particles[i].x + cos(particles[i].theta)*o.x - sin(particles[i].theta)*o.y;
			obs_t.y = particles[i].y + sin(particles[i].theta)*o.x + cos(particles[i].theta)*o.y;
			observations_transf.push_back(obs_t);
		}
		// Discard all landmarks out of sensor range
		std::vector<LandmarkObs> landmarks_within_range;
		for(Map::single_landmark_s landm: map_landmarks.landmark_list){
			if(dist(landm.x_f, landm.y_f, particles[i].x, particles[i].y) < sensor_range)
				landmarks_within_range.push_back(LandmarkObs{landm.id_i, landm.x_f, landm.y_f});
		}


		for(LandmarkObs obs_t: observations_transf){
			std::vector<double> dists;
			// For each transformed observation, calculate the distance to all landmarks
			for(LandmarkObs landm: landmarks_within_range)
				dists.push_back(dist(landm.x, landm.y, obs_t.x, obs_t.y));

			// Choose the minimum distance and associate the id of the landmark for this distance to the observation
			std::vector<double>::iterator shortest_dist;
			shortest_dist = std::min_element(std::begin(dists),std::end(dists));
			int closest_landm = std::distance(std::begin(dists), shortest_dist);
			// The closest landmark to observation obs_t is:
			LandmarkObs lm = landmarks_within_range[closest_landm];
			lm.id = obs_t.id;

			// Update weights
			particles[i].weight *= observation_pdf(obs_t.x, obs_t.y, lm.x, lm.y, std_x, std_y);
		}

	}
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
	std::mt19937 gen(rd());

	// Collect the current weights of all the particles
	std::vector<double> particle_weights;
	for(Particle p: particles)
		particle_weights.push_back(p.weight);

	std::discrete_distribution<> d(particle_weights.begin(), particle_weights.end());

	std::vector<Particle> particles_new;
	for(int i=0; i < num_particles; ++i){
		particles_new.push_back(particles[d(gen)]);
	}

	particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
