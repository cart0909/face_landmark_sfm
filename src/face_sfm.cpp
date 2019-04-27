#include "face_sfm.h"
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include "local_parameterization.h"
#include "factor.h"

FaceSfm::FaceSfm(int width, int height, double fx, double fy, double cx, double cy,
                 double k1, double k2, double p1, double p2)
{
    camera = std::make_shared<Pinhole>(width, height, fx, fy, cx, cy, k1, k2, p1, p2);
}

void FaceSfm::Process(const std::vector<cv::Mat>& v_img, const std::vector<std::vector<Eigen::Vector2d>>& v_landmarks)
{
    ROS_ASSERT(v_img.size() == v_landmarks.size() && v_img.size() >= 2);
    int num_frames = v_img.size();
    int num_landmarks = v_landmarks[0].size();

    for(int i = 0; i < num_landmarks; ++i) {
        Landmark lm;
        lm.id = i;
        landmarks.emplace_back(lm);
    }

    for(int i = 0; i < num_frames; ++i) {
        Frame frame;
        frame.id = i;
        frame.img = v_img[i];

        for(int j = 0; j < num_landmarks; ++j) {
            frame.lm_id.emplace_back(j);
            frame.lm_pt.emplace_back(v_landmarks[i][j]);
            Eigen::Vector3d pt_n = camera->BackProject(v_landmarks[i][j]);
            frame.lm_pt_n.emplace_back(pt_n);
            landmarks[j].pt_n_per_frame.emplace_back(pt_n);
//            cv::circle(frame.img, cv::Point(v_landmarks[i][j](0), v_landmarks[i][j](1)), 2, cv::Scalar(0, 255, 0), -1);
        }

//        cv::imshow("face", frame.img);
//        cv::waitKey(0);
        frames.emplace_back(frame);
    }

    InitGraph();
    GlobalBA();
}

void FaceSfm::InitGraph() {
    frames[0].Twc.so3() = Sophus::SO3d::rotX(-M_PI/2);

    // find the frame of biggest pixel movement from the first frame
    int best_frame_idx = -1;
    double best_dx_ave = -1;
    for(int i = 1; i < frames.size(); ++i) {
        double dx_ave = 0;
        for(int j = 0; j < landmarks.size(); ++j) {
            Eigen::Vector2d dx = frames[i].lm_pt_n[j].head<2>() - frames[0].lm_pt_n[j].head<2>();
            dx_ave += dx.norm();
        }
        dx_ave /= landmarks.size();

        if(dx_ave > best_dx_ave) {
            best_dx_ave = dx_ave;
            best_frame_idx = i;
        }
    }

    // essential matrix and recover pose
//    {
//        std::vector<cv::Point2d> point1, point2;
//        cv::Mat K = cv::Mat::eye(3, 3, CV_64F) * camera->f();
//        for(int i = 0; i < landmarks.size(); ++i) {
//            point1.emplace_back(landmarks[i].pt_n_per_frame[0](0) * camera->f(),
//                                landmarks[i].pt_n_per_frame[0](1) * camera->f());
//            point2.emplace_back(landmarks[i].pt_n_per_frame[best_frame_idx](0) * camera->f(),
//                                landmarks[i].pt_n_per_frame[best_frame_idx](1) * camera->f());
//        }
//        cv::Mat mask;
//        cv::Mat E = cv::findEssentialMat(point1, point2, K, cv::RANSAC, 0.999, 1.0, mask);
//        cv::Mat R, t; // R10, t10
//        cv::recoverPose(E, point1, point2, K, R, t, mask);
//        Eigen::Matrix3d R10;
//        Eigen::Vector3d t10;
//        cv::cv2eigen(R, R10);
//        cv::cv2eigen(t, t10);
//        Sophus::SE3d T10(R10, t10);
//        frames[best_frame_idx].Twc = frames[0].Twc * T10.inverse();
//    }

    // triangulation
    {
//        int iter_idx[2] = {0, best_frame_idx};
//        Sophus::SE3d Tcw[2] = {frames[0].Twc.inverse(), frames[best_frame_idx].Twc.inverse()};
//        Eigen::Matrix<double, 3, 4> P[2];
//        P[0] << Tcw[0].so3().matrix(), Tcw[0].translation();
//        P[1] << Tcw[1].so3().matrix(), Tcw[1].translation();

//        for(auto& lm : landmarks) {
//            Eigen::Matrix4d A;
//            for(int i = 0; i < 2; ++i) {
//                int frame_idx = iter_idx[i];
//                Eigen::Vector3d& pt_j = lm.pt_n_per_frame[frame_idx];
//                A.row(2 * i)     = pt_j(0) * P[i].row(2) - P[i].row(0);
//                A.row(2 * i + 1) = pt_j(1) * P[i].row(2) - P[i].row(1);
//            }

//            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
//            Eigen::Vector4d x = svd.matrixV().rightCols(1);
//            Eigen::Vector3d x3Dw = x.head<3>() / x(3);
//            lm.x3Dw = x3Dw;
//        }

        // simple set the distance of landmarks is front of first pose in 1 m
        for(auto& lm : landmarks) {
            Eigen::Vector3d x3Dc = lm.pt_n_per_frame[0] * 5;
            lm.x3Dw = frames[0].Twc * x3Dc;
        }
    }

    // PnP calc other frame poses
    {
        std::vector<cv::Point3d> object_points;
        for(auto& lm : landmarks) {
            object_points.emplace_back(lm.x3Dw(0), lm.x3Dw(1), lm.x3Dw(2));
        }

        for(int i = 1; i < frames.size(); ++i) {
//            if(i == best_frame_idx)
//                continue;

            std::vector<cv::Point2d> image_points;
            cv::Mat K = cv::Mat::eye(3, 3, CV_64F), rvec, tvec;

            for(auto& lm : landmarks) {
                image_points.emplace_back(lm.pt_n_per_frame[i](0), lm.pt_n_per_frame[i](1));
            }

            cv::solvePnPRansac(object_points, image_points, K, cv::noArray(), rvec, tvec, false,
                               100, 8.0 / camera->f());

            cv::Mat R;
            cv::Rodrigues(rvec, R);

            Eigen::Matrix3d Rcw;
            Eigen::Vector3d tcw;
            cv::cv2eigen(R, Rcw);
            cv::cv2eigen(tvec, tcw);
            Sophus::SE3d Tcw(Rcw, tcw);

            frames[i].Twc = Tcw.inverse();
        }
    }
}

void FaceSfm::GlobalBA() {
    LOG(WARNING) << "GlobalBA";
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(1);
    ceres::LocalParameterization* local_para_se3 = new LocalParameterizationSE3();

    double *para_pose = new double[frames.size() * 7];
    double *para_landmark = new double[landmarks.size() * 3];

    for(int i = 0, n = frames.size(); i < n; ++i) {
        std::memcpy(para_pose + 7 * i, frames[i].Twc.data(), sizeof(double) * 7);
        problem.AddParameterBlock(para_pose + 7 * i, 7, local_para_se3);
        if(i == 0)
            problem.SetParameterBlockConstant(para_pose);
    }

    for(int i = 0, n = landmarks.size(); i < n; ++i)
        std::memcpy(para_landmark + 3 * i, landmarks[i].x3Dw.data(), sizeof(double) * 3);

    for(auto& lm : landmarks) {
        for(int frame_idx= 0; frame_idx < lm.pt_n_per_frame.size(); ++frame_idx) {
            auto factor = ProjectionFactor::Create(lm.pt_n_per_frame[frame_idx], camera->f());
            problem.AddResidualBlock(factor, loss_function, para_pose + 7 * frame_idx, para_landmark + 3 * lm.id);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 50;
    options.num_threads = 6;
    ceres::Solver::Summary summary;
    LOG(WARNING) << "Solve...";
    ceres::Solve(options, &problem, &summary);

    LOG(INFO) << summary.FullReport() << std::endl;

    for(int i = 0, n = frames.size(); i < n; ++i)
        std::memcpy(frames[i].Twc.data(), para_pose + 7 * i, sizeof(double) * 7);

    for(int i = 0, n = landmarks.size(); i < n; ++i)
        std::memcpy(landmarks[i].x3Dw.data(), para_landmark + 3 * i, sizeof(double) * 3);

    // release
    delete [] para_pose;
    delete [] para_landmark;
}
