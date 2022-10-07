//  ================================================================
//  Created by Gregory Kramida on 9/23/19.
//  Copyright (c) 2019 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================

#include "GenericFileFetch.h"
#include <star/common/OpenCV_CrossPlatform.h>
#include <opencv2/core.hpp>
#include <boost/filesystem/operations.hpp>
#include <exception>
#include <regex>
#include <iterator>
#include <star/common/logging.h>
#include <array>

namespace fs = boost::filesystem;

template <unsigned max_cam>
star::GenericFileFetch<max_cam>::GenericFileFetch(const fs::path &data_path, std::string extension, bool force_no_masks) : m_mask_buffer_ix(SIZE_MAX), m_use_masks(false)
{
	std::vector<path> sorted_paths;
	if (extension.length() > 0 && extension[0] != '.')
	{
		extension = "." + extension;
	}
	std::copy(fs::recursive_directory_iterator(data_path), fs::recursive_directory_iterator(), std::back_inserter(sorted_paths));
	std::sort(sorted_paths.begin(), sorted_paths.end());
	bool unexpected_mask_frame_number = false;
	for (auto &path : sorted_paths)
	{
		if (FilenameIndicatesDepthImage(path.filename().string(), extension))
		{
			int frame_number = GetFrameNumber(path);
			int cam_number = GetCamNumber(path);
			if (frame_number != m_depth_image_paths[cam_number].size())
			{
				throw std::runtime_error("Unexpected depth frame number encountered");
			}
			m_depth_image_paths[cam_number].push_back(path);
		}
		else if (FilenameIndicatesRGBImage(path.filename().string(), extension))
		{
			int frame_number = GetFrameNumber(path);
			int cam_number = GetCamNumber(path);
			if (frame_number != m_rgb_image_paths[cam_number].size())
			{
				throw std::runtime_error("Unexpected RGB frame number encountered");
			}
			m_rgb_image_paths[cam_number].push_back(path);
		}
		else if (!force_no_masks && FilenameIndicatesMaskImage(path.filename().string(), extension))
		{
			int frame_number = GetFrameNumber(path);
			int cam_number = GetCamNumber(path);
			if (frame_number != m_mask_image_paths[cam_number].size())
			{
				unexpected_mask_frame_number = true;
			}
			m_mask_image_paths[cam_number].push_back(path);
		}
	}

	unsigned total_frame_num = m_depth_image_paths[0].size();
	for (unsigned cam_idx = 0; cam_idx < max_cam; ++cam_idx)
	{ // Check frame number equivilence
		if (m_depth_image_paths[cam_idx].size() != total_frame_num)
		{
			LOG(FATAL) << "Depth image cound doesn't equal to presumed count: " << total_frame_num;
		}
		else if (m_rgb_image_paths[cam_idx].size() != total_frame_num)
		{
			LOG(FATAL) << "Rgb image cound doesn't equal to presumed count: " << total_frame_num;
		}
	}

	if (!force_no_masks)
	{
		if (unexpected_mask_frame_number)
		{
			LOG(WARNING)
				<< "Warning: inconsistent mask frame numbers encountered in the filenames. Proceeding without masks.";
		}
		else
		{
			unsigned mask_idx_fit = true;
			for (unsigned cam_idx = 0; cam_idx < max_cam; ++cam_idx)
			{
				if (!m_mask_image_paths[cam_idx].empty())
				{
					if (m_depth_image_paths[cam_idx].size() != total_frame_num && !total_frame_num)
					{
						LOG(WARNING)
							<< "Warning: camera-" << cam_idx << " seems like there were some mask image files, but their number doesn't match the "
																"number of depth frames. Proceeding without masks.";
					}
					else
					{
						mask_idx_fit = false;
					}
				}
			}
			if (mask_idx_fit)
				m_use_masks = true;
		}
	}
}

template <unsigned max_cam>
bool star::GenericFileFetch<max_cam>::HasSubstringFromSet(const std::string &string, const std::string *set, int set_size)
{
	bool found_indicator = false;
	for (int i_target_string = 0; i_target_string < set_size; i_target_string++)
	{
		if (string.find(set[i_target_string]) != std::string::npos)
		{
			return true;
		}
	}
	return false;
}

template <unsigned max_cam>
bool star::GenericFileFetch<max_cam>::FilenameIndicatesDepthImage(const path &filename, const std::string &valid_extension)
{
	if (filename.extension() != valid_extension)
		return false;
	const std::array<std::string, 3> possible_depth_indicators = {"depth", "DEPTH", "Depth"};
	return HasSubstringFromSet(filename.string(), possible_depth_indicators.data(), possible_depth_indicators.size());
}

template <unsigned max_cam>
bool star::GenericFileFetch<max_cam>::FilenameIndicatesRGBImage(const path &filename, const std::string &valid_extension)
{
	if (filename.extension() != valid_extension)
		return false;
	const std::array<std::string, 5> possible_depth_indicators = {"color", "COLOR", "Color", "rgb", "RGB"};
	return HasSubstringFromSet(filename.string(), possible_depth_indicators.data(), possible_depth_indicators.size());
}

template <unsigned max_cam>
bool star::GenericFileFetch<max_cam>::FilenameIndicatesMaskImage(const star::GenericFileFetch<max_cam>::path &filename,
																 const std::string &valid_extension)
{
	if (filename.extension() != valid_extension)
		return false;
	const std::array<std::string, 3> possible_depth_indicators = {"mask", "Mask", "MASK"};
	return HasSubstringFromSet(filename.string(), possible_depth_indicators.data(), possible_depth_indicators.size());
}

template <unsigned max_cam>
int star::GenericFileFetch<max_cam>::GetFrameNumber(const star::GenericFileFetch<max_cam>::path &filename)
{
	const std::regex digits_regex("\\d+");
	std::smatch match_result;
	const std::string filename_stem = filename.stem().string();
	if (!std::regex_search(filename_stem, match_result, digits_regex))
	{ // Number not sufficient
		throw std::runtime_error("Could not find frame number in filename.");
	};
	return std::stoi(match_result.str(0)); // The final number in filename is frame number
}

template <unsigned max_cam>
int star::GenericFileFetch<max_cam>::GetCamNumber(const star::GenericFileFetch<max_cam>::path &filename)
{
	const std::regex digits_regex("\\d+");
	std::smatch match_result;
	const std::string parent_dirname = filename.parent_path().filename().string();
	if (!std::regex_search(parent_dirname, match_result, digits_regex))
	{
		throw std::runtime_error("Could not find camera number in dirname.");
	};
	return std::stoi(match_result.str(0)); // The first number in full path name is cam number
}

template <unsigned max_cam>
bool star::GenericFileFetch<max_cam>::FetchDepthImage(size_t cam_idx, size_t frame_idx, cv::Mat &depth_img)
{
	path file_path = this->m_depth_image_paths[cam_idx][frame_idx];
	// Read the image
	depth_img = cv::imread(file_path.string(), CV_ANYCOLOR | CV_ANYDEPTH); // NOLINT(hicpp-signed-bitwise)
	if (this->m_use_masks)
	{
		mask_mutex.lock();
		// Apply mask to image
		if (this->m_mask_buffer_ix != frame_idx)
		{
			m_mask_image_buffer = cv::imread(this->m_mask_image_paths[cam_idx][frame_idx].string(),
											 CV_ANYCOLOR | CV_ANYDEPTH); // NOLINT(hicpp-signed-bitwise)
			m_mask_buffer_ix = frame_idx;
		}
		cv::Mat masked;
		depth_img.copyTo(masked, m_mask_image_buffer);
		mask_mutex.unlock();
		depth_img = masked;
	}
	return true;
}

template <unsigned max_cam>
bool star::GenericFileFetch<max_cam>::FetchDepthImage(size_t cam_idx, size_t frame_idx, void *depth_img)
{
	return false;
}

template <unsigned max_cam>
bool star::GenericFileFetch<max_cam>::FetchRGBImage(size_t cam_idx, size_t frame_idx, cv::Mat &rgb_img)
{
	path file_path = this->m_rgb_image_paths[cam_idx][frame_idx];
	// Read the image
	rgb_img = cv::imread(file_path.string(), CV_ANYCOLOR | CV_ANYDEPTH); // NOLINT(hicpp-signed-bitwise)
	if (this->m_use_masks)
	{
		mask_mutex.lock();
		// Apply mask to rgb image
		if (this->m_mask_buffer_ix != frame_idx)
		{
			m_mask_image_buffer = cv::imread(this->m_mask_image_paths[cam_idx][frame_idx].string(),
											 CV_ANYCOLOR | CV_ANYDEPTH); // NOLINT(hicpp-signed-bitwise)
			m_mask_buffer_ix = frame_idx;
		}
		cv::Mat masked;
		rgb_img.copyTo(masked, m_mask_image_buffer);
		mask_mutex.unlock();
		rgb_img = masked;
	}
	return true;
}

template <unsigned max_cam>
bool star::GenericFileFetch<max_cam>::FetchRGBImage(size_t cam_idx, size_t frame_idx, void *rgb_img)
{
	return false;
}