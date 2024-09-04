/**
 * File: StereoParameters.h
 * Author: Nicolas Soncini 
 * Date: Aug 23, 2024
 * Description: Data structures and functionality to work with stereo
 * License: See LICENSE.txt file at the top project folder
 *
**/

#ifndef __STEREO_PARAMETERS__
#define __STEREO_PARAMETERS__

#include <opencv2/core.hpp>
#include "yaml-cpp/yaml.h"

// ---------------------------------------------------------------------------

struct StereoParameters
{
    cv::Mat left_camera_matrix;   // for PnP
    cv::Mat left_dist_coeffs;     // for PnP
    cv::Mat left_rectification;
    cv::Mat left_projection;      // for triangulation
    cv::Mat right_camera_matrix;
    cv::Mat right_dist_coeffs;
    cv::Mat right_rectification;
    cv::Mat right_projection;     // for triangulation
    cv::Size2i size;

    StereoParameters() = default;

    StereoParameters(std::string yaml_path)
    {
        YAML::Node yaml = YAML::LoadFile(yaml_path);

        left_camera_matrix = cv::Mat(
            yaml["left_camera_matrix"]["data"]
                .as<std::vector<float>>(), true
            ).reshape(0, 3);
        left_dist_coeffs = cv::Mat(
            yaml["left_distortion_coefficients"]["data"]
                .as<std::vector<float>>(), true
            ).reshape(0, 1);
        left_rectification = cv::Mat(
            yaml["left_rectification_matrix"]["data"]
                .as<std::vector<float>>(), true
            ).reshape(0, 3);
        left_projection = cv::Mat(
            yaml["left_projection_matrix"]["data"]
                .as<std::vector<float>>(), true
            ).reshape(0, 3);
        
        right_camera_matrix = cv::Mat(
            yaml["right_camera_matrix"]["data"]
                .as<std::vector<float>>(), true
            ).reshape(0, 3);
        right_dist_coeffs = cv::Mat(
            yaml["right_distortion_coefficients"]["data"]
                .as<std::vector<float>>(), true
            ).reshape(0, 1);
        right_rectification = cv::Mat(
            yaml["right_rectification_matrix"]["data"]
                .as<std::vector<float>>(), true
            ).reshape(0, 3);
        right_projection = cv::Mat(
            yaml["right_projection_matrix"]["data"]
                .as<std::vector<float>>(), true
            ).reshape(0, 3);

        size = cv::Size(
            yaml["image_width"].as<int>(), yaml["image_height"].as<int>()
            );
    }
};

// ---------------------------------------------------------------------------

#endif //__STEREO_PARAMETERS__