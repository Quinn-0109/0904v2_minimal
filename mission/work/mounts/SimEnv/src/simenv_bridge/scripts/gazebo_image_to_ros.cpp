#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include <gazebo/gazebo_client.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>

namespace {

class GazeboImageToRos {
 public:
  explicit GazeboImageToRos(ros::NodeHandle* private_node)
      : private_node_(*private_node) {
    private_node_.param<std::string>(
        "gazebo_topic", gazebo_topic_,
        "/gazebo/generated_world/a1_gazebo/front_camera/recording_camera/image");
    private_node_.param<std::string>(
        "image_topic", image_topic_, "/simenv/recording_camera/image_raw");
    private_node_.param<std::string>(
        "camera_info_topic", camera_info_topic_,
        "/simenv/recording_camera/camera_info");
    private_node_.param<std::string>(
        "ready_topic", ready_topic_, "/simenv/recording_camera_ready");
    private_node_.param<std::string>(
        "frame_id", frame_id_, "front_camera");
    private_node_.param("horizontal_fov_rad", horizontal_fov_,
                        1.0466666666666666);

    image_publisher_ = private_node_.advertise<sensor_msgs::Image>(
        image_topic_, 2);
    info_publisher_ = private_node_.advertise<sensor_msgs::CameraInfo>(
        camera_info_topic_, 2);
    ready_publisher_ = private_node_.advertise<std_msgs::Bool>(
        ready_topic_, 1, true);
    std_msgs::Bool not_ready;
    not_ready.data = false;
    ready_publisher_.publish(not_ready);
  }

  bool Start() {
    stopping_.store(false);
    gazebo_node_.reset(new gazebo::transport::Node());
    gazebo_node_->Init();
    gazebo_subscriber_ = gazebo_node_->Subscribe(
        gazebo_topic_, &GazeboImageToRos::OnImage, this);
    if (!gazebo_subscriber_) {
      ROS_ERROR("Cannot subscribe to Gazebo image topic %s",
                gazebo_topic_.c_str());
      return false;
    }
    ROS_INFO("Bridging Gazebo image %s -> ROS %s",
             gazebo_topic_.c_str(), image_topic_.c_str());
    return true;
  }

  void Stop() {
    stopping_.store(true);
    // Gazebo transport owns a callback thread.  Release its subscriber and
    // node while the client is still alive; destroying them after
    // gazebo::client::shutdown() can race a dead transport mutex.
    gazebo_subscriber_.reset();
    gazebo_node_.reset();
  }

 private:
  void OnImage(const boost::shared_ptr<const gazebo::msgs::ImageStamped>& stamped) {
    if (stopping_.load() || !stamped || !stamped->has_image()) {
      return;
    }
    const gazebo::msgs::Image& source = stamped->image();
    const uint32_t width = source.width();
    const uint32_t height = source.height();
    const uint32_t expected_step = width * 3u;
    const uint32_t step = source.has_step() && source.step() > 0u
                              ? source.step()
                              : expected_step;
    if (width == 0u || height == 0u ||
        source.data().size() < static_cast<std::size_t>(step) * height) {
      ROS_WARN_THROTTLE(5.0, "Malformed Gazebo RGB frame: %ux%u step=%u bytes=%zu",
                        width, height, step, source.data().size());
      return;
    }

    const ros::Time stamp = ros::Time::now();
    sensor_msgs::Image image;
    image.header.stamp = stamp;
    image.header.frame_id = frame_id_;
    image.width = width;
    image.height = height;
    image.encoding = "rgb8";
    image.is_bigendian = false;
    image.step = step;
    image.data.assign(source.data().begin(), source.data().end());
    image_publisher_.publish(image);

    sensor_msgs::CameraInfo info;
    info.header = image.header;
    info.width = width;
    info.height = height;
    info.distortion_model = "plumb_bob";
    info.D.assign(5, 0.0);
    const double focal = static_cast<double>(width) /
        (2.0 * std::tan(0.5 * horizontal_fov_));
    const double cx = 0.5 * (static_cast<double>(width) - 1.0);
    const double cy = 0.5 * (static_cast<double>(height) - 1.0);
    info.K = {focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0};
    info.R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    info.P = {focal, 0.0, cx, 0.0, 0.0, focal, cy, 0.0,
              0.0, 0.0, 1.0, 0.0};
    info_publisher_.publish(info);
    if (!ready_published_) {
      std_msgs::Bool ready;
      ready.data = true;
      ready_publisher_.publish(ready);
      ready_published_ = true;
      ROS_INFO("First rendered RGB frame bridged (%ux%u)", width, height);
    }
  }

  ros::NodeHandle private_node_;
  ros::Publisher image_publisher_;
  ros::Publisher info_publisher_;
  ros::Publisher ready_publisher_;
  gazebo::transport::NodePtr gazebo_node_;
  gazebo::transport::SubscriberPtr gazebo_subscriber_;
  std::string gazebo_topic_;
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string ready_topic_;
  std::string frame_id_;
  double horizontal_fov_ = 1.0466666666666666;
  std::atomic<bool> stopping_{false};
  bool ready_published_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "gazebo_recording_camera_bridge");
  ros::NodeHandle private_node("~");
  try {
    gazebo::client::setup(argc, argv);
    int exit_code = 0;
    {
      GazeboImageToRos bridge(&private_node);
      if (!bridge.Start()) {
        exit_code = 2;
      } else {
        ros::spin();
      }
      bridge.Stop();
    }
    gazebo::client::shutdown();
    return exit_code;
  } catch (const std::exception& error) {
    ROS_FATAL("Gazebo image bridge failed: %s", error.what());
    gazebo::client::shutdown();
    return 2;
  }
}
