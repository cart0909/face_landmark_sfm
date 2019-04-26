#include "face_sfm.h"
#include <ceres/ceres.h>
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
        }

        frames.emplace_back(frame);
    }
}

void FaceSfm::InitGraph() {
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
}

void FaceSfm::GlobalBA() {
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
    options.num_threads = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOG(INFO) << summary.FullReport() << std::endl;

    for(int i = 0, n = frames.size(); i < n; ++i)
        std::memcpy(frames[i].Twc.data(), para_pose + 7 * i, sizeof(double));

    for(int i = 0, n = landmarks.size(); i < n; ++i)
        std::memcpy(landmarks[i].x3Dw.data(), para_landmark + 3 * i, sizeof(double));

    // release
    delete [] para_pose;
    delete [] para_landmark;
}
