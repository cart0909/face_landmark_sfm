#pragma once
#include <memory>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "camera.h"

class FaceSfm {
public:
    FaceSfm(int width, int height, double fx, double fy, double cx, double cy,
            double k1, double k2, double p1, double p2);

    struct Frame {
        int id;
        cv::Mat img;
        Sophus::SE3d Twc;
        std::vector<int> lm_id;
        std::vector<Eigen::Vector2d> lm_pt;
        std::vector<Eigen::Vector3d> lm_pt_n;
    };

    struct Landmark {
        int id;
        Eigen::Vector3d x3Dw;
        std::vector<Eigen::Vector3d> pt_n_per_frame;
    };

    void Process(const std::vector<cv::Mat>& v_img, const std::vector<std::vector<Eigen::Vector2d>>& v_landmarks);
    void InitGraph();
    void GlobalBA();

    std::vector<Frame> frames;
    std::vector<Landmark> landmarks;
    std::shared_ptr<CameraBase> camera;
};
