#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <boost/foreach.hpp>
#include "face_sfm.h"

class DataMgr {
public:
    DataMgr(const ros::NodeHandle& nh) {
        ReadParam(nh);
    }

    void ReadParam(const ros::NodeHandle& nh) {
        nh.getParam("fx", fx);
        nh.getParam("fy", fy);
        nh.getParam("cx", cx);
        nh.getParam("cy", cy);
        nh.getParam("k1", k1);
        nh.getParam("k2", k2);
        nh.getParam("p1", p1);
        nh.getParam("p2", p2);
        nh.getParam("width", width);
        nh.getParam("height", height);
        face_sfm = std::make_shared<FaceSfm>(width, height, fx, fy, cx, cy, k1, k2, p1, p2);
    }

    void Callback(const sensor_msgs::ImageConstPtr& img_msg,
                  const sensor_msgs::PointCloud2ConstPtr& lm_msg)
    {
        cv::Mat img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
        std::vector<Eigen::Vector2d> landmarks(lm_msg->width);

        for(int i = 0, n = landmarks.size(); i < n; ++i) {
            Eigen::Vector2f pt;
            std::memcpy(pt.data(), lm_msg->data.data() + 8 * i, 2 * sizeof(float));
            landmarks[i] = pt.cast<double>();
        }

        v_img.emplace_back(img);
        v_landmarks.emplace_back(landmarks);
    };

    void Process() {
         face_sfm->Process(v_img, v_landmarks);
    }

    std::vector<cv::Mat> v_img;
    std::vector<std::vector<Eigen::Vector2d>> v_landmarks;
    double fx, fy, cx, cy, k1, k2, p1, p2;
    int width, height;
    std::shared_ptr<FaceSfm> face_sfm;
};

template <class T>
class BagSubscriber : public message_filters::SimpleFilter<T>
{
public:
    void NewMessage(const boost::shared_ptr<T const>& msg) {
        this->signalMessage(msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "face_landmark_sfm_node");
    ros::NodeHandle nh("~");
    DataMgr data_mgr(nh);

    rosbag::Bag bag;
    bag.open(argv[1], rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.emplace_back("/image_raw");
    topics.emplace_back("/landmark");

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    BagSubscriber<sensor_msgs::Image> sub_img;
    BagSubscriber<sensor_msgs::PointCloud2> sub_lm;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> sync(sub_img, sub_lm, 10);

    sync.registerCallback(boost::bind(&DataMgr::Callback, &data_mgr, _1, _2));

    BOOST_FOREACH(rosbag::MessageInstance const m, view) {
        sensor_msgs::ImageConstPtr img_msg = m.instantiate<sensor_msgs::Image>();
        if(img_msg) {
            sub_img.NewMessage(img_msg);
        }

        sensor_msgs::PointCloud2ConstPtr lm_msg = m.instantiate<sensor_msgs::PointCloud2>();
        if(lm_msg) {
            sub_lm.NewMessage(lm_msg);
        }
    }

    bag.close();
    data_mgr.Process();

    ros::spin();
    return 0;
}
