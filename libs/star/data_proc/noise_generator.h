/*
* Created by Haonan Chang, 10/09/2021
*/
#pragma once
#include <opencv2/opencv.hpp>
#include <random>

namespace star {
	/*
	* \brief: Apply gaussian noise to depth image
	* \param: noise_level: (meter)
	*/
	void apply_gaussian_noise(
		const cv::Mat& depth_image,
		cv::Mat& noise_depth_image,
		float noise_level
		);  //TODO: change to cuda version


	/*
	* \brief: Randomly corp a part away from a depth image
	*/
	void apply_random_corp(
		const cv::Mat& depth_image,
		cv::Mat& droped_depth_image,
		float corp_ratio
	);
}