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

#include "particle_filter.h"
#include "helper_functions.h"
#include "map.h"
#define _USE_MATH_DEFINES
#include <cmath>
using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 500;	
	// normal (Gaussian) distributions.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		Particle pf = Particle();
		pf.id = i;
		/*pf.x = x;
		pf.y = y;
		pf.theta = theta;*/
		pf.weight = 1.0;		
		pf.x = dist_x(gen);
		pf.y = dist_y(gen);
		pf.theta = dist_theta(gen);
		weights.push_back(1.0);
		particles.push_back(pf);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// std_pos = GPS measurement uncertainty
	// eq when yaw_rate is 0 https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/2c318113-724b-4f9f-860c-cb334e6e4ad7/lessons/5d3e95df-f402-4b22-b8ba-ec0d9257666a/concepts/ff7658c9-6edd-498b-b066-1578ec3f97aa
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++)
	{
		Particle p = particles[i];
		if (yaw_rate < 0.001) {
			p.x += velocity*delta_t*cos(p.theta);
			p.y += velocity*delta_t*sin(p.theta);			
		}
		else
		{
			p.x += (velocity / yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
			p.y += (velocity / yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
			p.theta += yaw_rate*delta_t;
		}
		
		p.x += noise_x(gen);
		p.y += noise_y(gen);
		p.theta += noise_theta(gen);
		particles[i] = p;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	int mapId = 0;
	double nearest_ngbr_dist = numeric_limits<double>::max();

	for (int i = 0; i < observations.size(); i++)
	{
		LandmarkObs obs = observations[i];

		for (int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];

			double dist_op = dist(obs.x, obs.y, pred.x, pred.y); //dist between o(bservation) and p(redicted)

			if (dist_op < nearest_ngbr_dist)
			{
				nearest_ngbr_dist = dist_op;
				mapId = pred.id;
			}
		}

		observations[i].id = mapId;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++)
	{
		Particle p = particles[i];
		double wt = 1.0;

		vector<LandmarkObs> sensedLandmarks; //sensed by sensors

		for (int l = 0; l < map_landmarks.landmark_list.size(); l++)
		{		
			Map::single_landmark_s sl = map_landmarks.landmark_list[l];
			
			//if (fabs(sl.x_f - p.x) <= sensor_range && fabs(sl.y_f - p.y) <= sensor_range) {
			if (dist(sl.x_f, sl.y_f, p.x, p.y) <= sensor_range) {
				sensedLandmarks.push_back(LandmarkObs{ sl.id_i, sl.x_f, sl.y_f }); //using vehicle co-ordinate system
			}
		}

		vector<LandmarkObs> transformedLandmarks;
		LandmarkObs obs;
		for (int o = 0; o < observations.size(); o++)
		{
			LandmarkObs trans_obs;
			obs = observations[o];
			//converting from vehicle to map co-ordinates
			/*trans_obs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
			trans_obs.y = obs.x * sin(p.theta) - obs.y * cos(p.theta) + p.y;
			trans_obs.x = p.x + (obs.x * cos(p.theta) - obs.y * sin(p.theta));
			trans_obs.y = p.y + (obs.x * sin(p.theta) - obs.y * cos(p.theta));*/

			trans_obs.x = p.x + (obs.x * cos(p.theta)) - (obs.y * sin(p.theta));
			trans_obs.y = p.y + (obs.x * sin(p.theta)) + (obs.y * cos(p.theta));
			transformedLandmarks.push_back(trans_obs);
		
			particles[i].weight = 1.0;
			dataAssociation(sensedLandmarks, transformedLandmarks);
			double p_x=0.0, p_y=0.0, m_x=0.0, m_y=0.0;

			for (int t = 0; t < transformedLandmarks.size(); t++)
			{
				m_x = transformedLandmarks[t].x; //measured
				m_y = transformedLandmarks[t].y;
				for (int s = 0; s < sensedLandmarks.size(); s++)
				{
					if (sensedLandmarks[s].id == transformedLandmarks[t].id)
					{
						p_x = sensedLandmarks[s].x; //predicted
						p_y = sensedLandmarks[s].y;
						break;
					}
				}
			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];

			//long double mwg_w = (1 / (2 * M_PI*std_x*std_y)) * exp(-(pow(p_x - m_x, 2) / (2 * pow(std_x, 2)) + (pow(p_y - m_y, 2) / (2 * pow(std_y, 2)))));
			//cout << "mwg_w: " << mwg_w << endl;

			double num = exp(-0.5 * (pow((p_x - m_x), 2) / pow(std_x, 2) + pow((p_y - m_y), 2) / pow(std_y, 2)));
			double denom = 2 * M_PI * std_x * std_y;
			wt *= num/denom;
			cout << "weight: " << wt << endl;
		}
			//long double mwg_w = (1 / (2 * M_PI*std_x*std_y)) * exp(-(pow(m_x - p_x, 2) / (2 * pow(std_x, 2)) + (pow(m_y - p_y, 2) / (2 * pow(std_y, 2)))));
			//if (fabs(mwg_w) > 0.001)
			//{
			p.weight = particles[i].weight *= wt;
			//}

			weights[i] = p.weight;
		
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resample_particles;
	for (int i = 0; i < num_particles; i++)
	{
		resample_particles.push_back(particles[distribution(gen)]);
	}

	particles = resample_particles;
}

double * ParticleFilter::weighted_mean_error(double gt_x, double gt_y, double gt_theta) {
	// TODO: Calculate the weighted mean error of the particle filter.
	static double error[3];
	//Particle max = *max_element(particles.begin(), particles.end()); // there is * coz it return an iterator
	double max_weight = 0.0;
	Particle best_particle = Particle();
	for (int i = 0; i < num_particles; i++) {
		if (particles[i].weight > max_weight) {
			best_particle = particles[i];
			max_weight = best_particle.weight;
		}
	}
		
	error[0] = (best_particle.x - gt_x);
	error[1] = floor(best_particle.y - gt_y);
	error[2] = (best_particle.theta - gt_theta);
	/*error[2] = fmod(error[2], 2.0 * M_PI);
	if (error[2] > M_PI) {
		error[2] = 2.0 * M_PI - error[2];
	}*/
	return error;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
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