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

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  if(!is_initialized)
  {
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    // number of particles
    num_particles = 100;
    
    // resize the number of particles
    particles.resize(num_particles);
    
    // resize the weights
    weights.resize(num_particles);
    
    // initialize all particles to the first position
    for(int i = 0; i < num_particles; i++) {
      particles[i].id = i;
      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
      particles[i].theta = dist_theta(gen);
      particles[i].weight = 1.0;
    }
    
    // intialize all weights of the particles to 1
    std::fill(weights.begin(), weights.end(), 1.0);
    
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  // for each particle predict its position after delta_t based on its position
  // , velocity and yaw_rate. Also, add random Gaussian noise from normal distribution
  for(auto &particle : particles) {
    double x = particle.x;
    double y = particle.y;
    double theta = particle.theta;
    
    if(fabs(yaw_rate) < 0.00001) {
      double theta_f = theta;
      double x_f = x + velocity * delta_t * cos(theta);
      double y_f = y + velocity * delta_t * sin(theta);
      
      particle.x = x_f;
      particle.y = y_f;
      particle.theta = theta_f;

    } else {
      double theta_f = theta + yaw_rate * delta_t;
      double x_f = x + (velocity/yaw_rate) * (sin(theta_f) - sin(theta));
      double y_f = y + (velocity/yaw_rate) * (cos(theta) - cos(theta_f));
      
      particle.x = x_f;
      particle.y = y_f;
      particle.theta = theta_f;
    }
    
    // add noise
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign
  // the observed measurement to this particular landmark.
  for(int i = 0; i < observations.size(); i++) {
    
    // minimum distance between the predicted and observed measurement
    double min_dist = numeric_limits<double>::max();

    // ID of the predicted landmark observation which is closest to the observation
    int min_id = -1;
    
    for(int j = 0; j < predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      
      if(distance < min_dist) {
        min_dist = distance;
        min_id = predicted[j].id;
      }
    }
    
    // set the id to the closest predicted landmark ID
    observations[i].id = min_id;
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
  
  for(int i = 0; i < num_particles; i++)
  {
    double x_p = particles[i].x;
    double y_p = particles[i].y;
    double theta_p = particles[i].theta;
    
    // transform each observation to the particle's perspective
    std::vector<LandmarkObs> transformed_obs;
    transformed_obs.resize(observations.size());
    
    for(int j = 0; j < observations.size(); j++)
    {
      double x_c = observations[j].x;
      double y_c = observations[j].y;
      
      double x_m = x_p + (cos(theta_p) * x_c) - (sin(theta_p) * y_c);
      double y_m = y_p + (sin(theta_p) * x_c) + (cos(theta_p) * y_c);
      
      transformed_obs[j].x = x_m;
      transformed_obs[j].y = y_m;
      transformed_obs[j].id = observations[j].id;
    }

    // create a vector of predicted landmarks which are within sensor range
    std::vector<LandmarkObs> predicted_landmarks;
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double x_l = map_landmarks.landmark_list[j].x_f;
      double y_l = map_landmarks.landmark_list[j].y_f;
      int id_l = map_landmarks.landmark_list[j].id_i;
      
      if((fabs(x_l - x_p) <= sensor_range) && (fabs(y_l - y_p) <= sensor_range))
      {
        predicted_landmarks.push_back(LandmarkObs{id_l, x_l, y_l});
      }
    }
    
    // nearest neighbor data association for the landmark predictions and transformed observations
    dataAssociation(predicted_landmarks, transformed_obs);
    
    // reset weight to 1.0
    particles[i].weight = 1.0;
    
    for(int j = 0; j < transformed_obs.size(); j++)
    {
      // the observation position
      double obs_x = transformed_obs[j].x;
      double obs_y = transformed_obs[j].y;
      
      // find the associated landmark position
      int associated_landmark_id = transformed_obs[j].id;
      double x_l = 0, y_l = 0;
      
      for(int k = 0; k < predicted_landmarks.size(); k++)
      {
        if(predicted_landmarks[k].id == associated_landmark_id)
        {
          x_l = predicted_landmarks[k].x;
          y_l = predicted_landmarks[k].y;
          break;
        }
      }
      
      // compute the weight by using the multivariate gaussian probability distribution
      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];
      double sigma_x_2 = pow(sigma_x, 2);
      double sigma_y_2 = pow(sigma_y, 2);
      
      double normalizer = 1/(2 * M_PI * sigma_x * sigma_y);
      double weight = normalizer * exp(-(((pow(obs_x-x_l,2))/2*sigma_x_2) + ((pow(obs_y - y_l, 2))/2*sigma_y_2)));
      
      particles[i].weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  random_device rd;
  default_random_engine gen(rd());
  vector<Particle> resampled_particles(num_particles);
  
  // get weights vector
  vector<double> particle_weights(num_particles);
  for(int i = 0; i < num_particles; i++) {
    particle_weights[i] = particles[i].weight;
  }
    
  for(int i = 0; i < num_particles; i++) {
    discrete_distribution<int> index(particle_weights.begin(), particle_weights.end());
    resampled_particles[i] = particles[index(gen)];
  }
  
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
