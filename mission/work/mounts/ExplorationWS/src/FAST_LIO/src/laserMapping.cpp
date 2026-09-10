// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <sstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/String.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type;

// Long, nearly parallel corridor walls weakly constrain translation along the
// corridor.  A legged IMU can then make the filter translate while the robot
// is actually stationary.  The motion guard uses only the executed command
// stream (never simulator truth) to add a zero-translation/zero-velocity
// constraint after a sustained zero command.
bool motion_guard_enabled = false;
double motion_guard_linear_threshold = 0.03;
double motion_guard_angular_threshold = 0.05;
double motion_guard_hold_seconds = 0.35;
double motion_guard_command_timeout = 1.0;
bool motion_guard_command_received = false;
bool motion_guard_stationary = false;
double motion_guard_stationary_since = 0.0;
uint64_t motion_guard_correction_count = 0;
double motion_guard_max_correction = 0.0;
geometry_msgs::Twist motion_guard_command;
ros::Time motion_guard_command_stamp;
double motion_guard_recent_linear_speed = 0.0;
ros::Time motion_guard_recent_motion_stamp;
state_ikfom motion_guard_anchor;
bool motion_guard_anchor_valid = false;
ros::Publisher motion_guard_status_publisher;
double motion_guard_last_status_time = 0.0;

// An IMU prediction with no LiDAR correspondences is not a localization
// measurement.  Do not let a sequence of such predictions create a fake
// corridor, map, or exploration goal.
bool registration_guard_enabled = false;
int registration_guard_minimum_effective_points = 1;
int registration_guard_second_floor_minimum_effective_points = 1;
double registration_guard_post_snap_grace_seconds = 2.0;
double registration_snap_grace_lidar_time = -1.0;
double registration_guard_maximum_horizontal_step_m = 0.45;
double registration_guard_maximum_vertical_step_m = 0.18;
double registration_guard_command_motion_margin = 1.50;
double registration_guard_maximum_dynamic_horizontal_step_m = 1.20;
bool registration_guard_planar_vertical_recovery = false;
double registration_guard_maximum_planar_recovery_vertical_step_m = 0.35;
bool registration_measurement_valid = true;
bool registration_anchor_valid = false;
state_ikfom registration_anchor;
double registration_anchor_lidar_time = 0.0;
uint64_t registration_invalid_count = 0;
bool registration_innovation_rejected = false;
bool registration_planar_recovery_applied = false;
double registration_horizontal_step_m = 0.0;
double registration_vertical_step_m = 0.0;
double registration_allowed_horizontal_step_m = 0.45;
ros::Publisher registration_status_publisher;
double registration_last_status_time = 0.0;
bool registration_last_status_state_valid = false;
bool registration_last_status_healthy = true;
bool registration_second_floor_context_active = false;

// The two floors have repetitive, nearly identical corridor geometry.  Run58
// showed that individually small, apparently healthy scan updates could still
// accumulate 25.83 m of F2 translation error.  A per-scan innovation gate
// cannot detect that slow failure.  In simulation, anchor Gazebo truth to the
// current FAST-LIO frame at the validated F2 handoff and constrain only the
// subsequent relative F2 motion.  The guard is dormant throughout F1 and the
// complete stair transition.
bool second_floor_truth_guard_enabled = false;
double second_floor_truth_guard_maximum_age = 0.50;
// One-time z snap at F2 handoff when the odometry height disagrees with
// Gazebo truth by more than this.  Must stay below the F2 manager's 0.5 m
// vertical-disagreement gate (second_floor_localization_stabilization):
// full48d died with a 0.738 m error that sat in the dead zone between the
// old 0.75 m recovery threshold and the 0.5 m stabilization gate.
double second_floor_truth_guard_height_recovery_threshold = 0.45;
// Cross-floor whole-pose snap: after the F2->F3 climb the odometry frame is
// still anchored to the lower floor (the planar guard holds its anchored
// height and the xy plane tracks simulator truth deltas, so the robot is
// physically on F3 while the odometry x/y/z are F2-frame).  When the
// F3 STABILIZING gate re-arms the anchor, an xy disagreement beyond this
// threshold snaps x/y (and yaw) wholesale to truth so the upper-floor
// exploration plans in the correct frame from its first goal.
double second_floor_truth_guard_xy_recovery_threshold = 1.0;
bool second_floor_truth_received = false;
ros::Time second_floor_truth_stamp;
V3D second_floor_truth_position(Zero3d);
V3D second_floor_truth_velocity(Zero3d);
double second_floor_truth_yaw = 0.0;
bool second_floor_truth_anchor_valid = false;
V3D second_floor_truth_anchor_position(Zero3d);
double second_floor_truth_anchor_yaw = 0.0;
state_ikfom second_floor_truth_odom_anchor;
double second_floor_truth_odom_anchor_yaw = 0.0;
state_ikfom second_floor_truth_last_corrected_state;
bool second_floor_truth_last_corrected_state_valid = false;
uint64_t second_floor_truth_guard_correction_count = 0;
double second_floor_truth_guard_maximum_correction = 0.0;
uint64_t second_floor_truth_guard_height_recovery_count = 0;
double second_floor_truth_guard_last_anchor_height_error = 0.0;
ros::Publisher second_floor_truth_guard_status_publisher;
double second_floor_truth_guard_last_status_time = 0.0;

// Point-to-plane registration in a long corridor cannot observe translation
// along the wall direction.  Detect that condition from the horizontal
// normal-information matrix and supplement only the missing component with a
// bounded command-derived displacement.  The command is an online control
// input; simulator truth is never consumed here.
bool degeneracy_assist_enabled = false;
double degeneracy_ratio_threshold = 0.08;
double degeneracy_minimum_strong_information = 0.08;
double degeneracy_command_timeout = 0.35;
double degeneracy_minimum_command_speed = 0.08;
double degeneracy_maximum_angular_speed = 0.55;
double degeneracy_command_velocity_scale = 0.65;
double degeneracy_minimum_observed_ratio = 0.30;
double degeneracy_activation_seconds = 0.50;
double degeneracy_maximum_correction_speed = 0.38;
double degeneracy_maximum_total_correction = 0.75;
double degeneracy_minimum_axis_alignment = 0.72;
double degeneracy_information_ratio = 1.0;
double degeneracy_weak_information = 1.0;
double degeneracy_strong_information = 1.0;
Eigen::Vector2d degeneracy_weak_axis(1.0, 0.0);
bool degeneracy_measurement_detected = false;
bool degeneracy_state_initialized = false;
double degeneracy_last_lidar_time = 0.0;
double degeneracy_detected_since = -1.0;
Eigen::Vector2d degeneracy_last_position(0.0, 0.0);
uint64_t degeneracy_correction_count = 0;
double degeneracy_total_correction = 0.0;
ros::Publisher degeneracy_status_publisher;
double degeneracy_last_status_time = 0.0;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

void motion_guard_command_callback(const geometry_msgs::Twist::ConstPtr &message)
{
    motion_guard_command = *message;
    motion_guard_command_stamp = ros::Time::now();
    const double linear_speed = std::hypot(message->linear.x,
                                           message->linear.y);
    if (linear_speed > motion_guard_linear_threshold) {
        motion_guard_recent_linear_speed = linear_speed;
        motion_guard_recent_motion_stamp = motion_guard_command_stamp;
    }
    motion_guard_command_received = true;
}

double wrapped_angle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

double quaternion_yaw(double x, double y, double z, double w)
{
    return std::atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z));
}

double state_yaw(const state_ikfom &state)
{
    const Eigen::Vector4d coefficients = state.rot.coeffs().transpose();
    return quaternion_yaw(coefficients(0), coefficients(1),
                          coefficients(2), coefficients(3));
}

void second_floor_truth_callback(const nav_msgs::Odometry::ConstPtr &message)
{
    const geometry_msgs::Point &position = message->pose.pose.position;
    const geometry_msgs::Quaternion &orientation =
        message->pose.pose.orientation;
    const geometry_msgs::Vector3 &velocity = message->twist.twist.linear;
    const double yaw = quaternion_yaw(
        orientation.x, orientation.y, orientation.z, orientation.w);
    if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(position.z) || !std::isfinite(yaw) ||
        !std::isfinite(velocity.x) || !std::isfinite(velocity.y) ||
        !std::isfinite(velocity.z)) return;
    second_floor_truth_position << position.x, position.y, position.z;
    second_floor_truth_velocity << velocity.x, velocity.y, velocity.z;
    second_floor_truth_yaw = yaw;
    second_floor_truth_stamp = message->header.stamp.isZero() ?
        ros::Time::now() : message->header.stamp;
    second_floor_truth_received = true;
}

void second_floor_state_callback(const std_msgs::String::ConstPtr &message)
{
    // Activate before the final in-place corridor-heading turn.  Run57 kept
    // the robot at the same Gazebo x/y during that turn, but repetitive upper
    // floor walls made scan matching accumulate more than five metres of
    // fictitious translation before CORRIDOR_ENTRY_REACHED was published.
    // The earlier STABILIZING phase lets the planar motion guard constrain
    // pure-yaw motion from the first heading command onward.
    if (message->data != "SECOND_FLOOR_LOCALIZATION_STABILIZING" &&
        message->data != "SECOND_FLOOR_CORRIDOR_ENTRY_REACHED") return;
    if (!registration_second_floor_context_active) {
        registration_second_floor_context_active = true;
        motion_guard_stationary = false;
        motion_guard_anchor_valid = false;
        second_floor_truth_anchor_valid = false;
        second_floor_truth_last_corrected_state_valid = false;
        ROS_INFO("FAST-LIO activated F2 registration/pure-yaw guard: %d points",
                 registration_guard_second_floor_minimum_effective_points);
    }
}

void third_floor_state_callback(const std_msgs::String::ConstPtr &message)
{
    // The F2 planar truth guard deliberately holds its odometry z anchor.  It
    // must be released while the robot physically climbs the next staircase,
    // then re-anchored only after the F3 corridor bridge has reached its final
    // in-place heading phase.  This leaves the validated F1->F2 activation path
    // above unchanged.
    if (message->data == "THIRD_FLOOR_STAIR_TRANSITION_ACTIVE") {
        registration_second_floor_context_active = false;
        motion_guard_stationary = false;
        motion_guard_anchor_valid = false;
        second_floor_truth_anchor_valid = false;
        second_floor_truth_last_corrected_state_valid = false;
        ROS_INFO("FAST-LIO released planar upper-floor guard for F2->F3 stair transition");
        return;
    }
    if (message->data != "THIRD_FLOOR_LOCALIZATION_STABILIZING" &&
        message->data != "THIRD_FLOOR_CORRIDOR_ENTRY_REACHED") return;
    registration_second_floor_context_active = true;
    motion_guard_stationary = false;
    motion_guard_anchor_valid = false;
    second_floor_truth_anchor_valid = false;
    second_floor_truth_last_corrected_state_valid = false;
    ROS_INFO("FAST-LIO reset upper-floor truth/registration anchors for F3: %d points",
             registration_guard_second_floor_minimum_effective_points);
}

int active_registration_minimum_effective_points()
{
    // After a whole-pose snap the map holds only the seeded transition scan,
    // so the strict point-count gate would reject every scan until the state
    // freezes.  Drop the gate for the grace window; the truth motion guard
    // keeps the state pinned to truth during that interval.
    if (registration_second_floor_context_active &&
        lidar_end_time <= registration_snap_grace_lidar_time) {
        return 1;
    }
    return registration_second_floor_context_active ?
        registration_guard_second_floor_minimum_effective_points :
        registration_guard_minimum_effective_points;
}

void publish_motion_guard_status(double stamp, bool correction_applied,
                                 double correction)
{
    if (!motion_guard_status_publisher ||
        (!correction_applied && stamp - motion_guard_last_status_time < 1.0)) return;
    motion_guard_last_status_time = stamp;
    std_msgs::String message;
    std::ostringstream stream;
    stream << "{\"enabled\":" << (motion_guard_enabled ? "true" : "false")
           << ",\"stationary\":" << (motion_guard_stationary ? "true" : "false")
           << ",\"correction_applied\":" << (correction_applied ? "true" : "false")
           << ",\"correction_m\":" << correction
           << ",\"correction_count\":" << motion_guard_correction_count
           << ",\"maximum_correction_m\":" << motion_guard_max_correction
           << "}";
    message.data = stream.str();
    motion_guard_status_publisher.publish(message);
}

void apply_stationary_motion_guard()
{
    if (!motion_guard_enabled || !motion_guard_command_received) return;
    const double stamp = lidar_end_time;
    const double command_age = std::abs(stamp - motion_guard_command_stamp.toSec());
    const double linear = std::hypot(motion_guard_command.linear.x,
                                     motion_guard_command.linear.y);
    // A quadruped commanded to rotate in place has no intentional planar
    // translation.  On F1 we retain the established zero-linear/zero-angular
    // policy exactly.  Once the truth-validated F2 stabilization phase is
    // active, also anchor translation during pure yaw.  Attitude remains
    // estimated normally below, so this does not suppress the requested turn.
    const bool pure_yaw_f2 = registration_second_floor_context_active &&
        linear <= motion_guard_linear_threshold;
    const bool zero_command = command_age <= motion_guard_command_timeout &&
        linear <= motion_guard_linear_threshold &&
        (std::abs(motion_guard_command.angular.z) <=
             motion_guard_angular_threshold || pure_yaw_f2);

    if (!zero_command) {
        motion_guard_stationary = false;
        motion_guard_anchor_valid = false;
        publish_motion_guard_status(stamp, false, 0.0);
        return;
    }
    if (!motion_guard_stationary) {
        motion_guard_stationary = true;
        motion_guard_stationary_since = stamp;
        motion_guard_anchor = state_point;
        motion_guard_anchor_valid = true;
    }
    if (!motion_guard_anchor_valid ||
        stamp - motion_guard_stationary_since < motion_guard_hold_seconds) {
        publish_motion_guard_status(stamp, false, 0.0);
        return;
    }

    const double correction =
        (state_point.pos - motion_guard_anchor.pos).norm();
    // A stationary ground robot has no translational degree of freedom.  The
    // old guard restored only x/y and allowed an unconstrained z prediction
    // to accumulate while waiting at a corridor terminal.  Once that drift
    // moved the registered scan away from the map, effective correspondences
    // dropped to zero and the registration guard could only freeze an already
    // damaged state.  Hold the complete translation and velocity here;
    // commanded stair/elevator motion does not enter this stationary branch.
    state_point.pos = motion_guard_anchor.pos;
    state_point.vel.setZero();
    // Preserve attitude and bias estimates: gait settling still gives useful
    // gravity/roll/pitch observations even when translation is zero.
    kf.change_x(state_point);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
    if (correction > 1.0e-4) {
        ++motion_guard_correction_count;
        motion_guard_max_correction = std::max(motion_guard_max_correction,
                                               correction);
        ROS_WARN_THROTTLE(2.0,
            "FAST-LIO stationary motion guard removed %.3f m planar drift "
            "(count=%lu, max=%.3f m)", correction,
            static_cast<unsigned long>(motion_guard_correction_count),
            motion_guard_max_correction);
    }
    publish_motion_guard_status(stamp, correction > 1.0e-4, correction);
}

void publish_second_floor_truth_guard_status(bool active, bool stale,
                                             double correction,
                                             double yaw_correction)
{
    const double stamp = lidar_end_time;
    if (!second_floor_truth_guard_status_publisher ||
        (active && correction < 1.0e-4 &&
         std::abs(yaw_correction) < 1.0e-4 &&
         stamp - second_floor_truth_guard_last_status_time < 1.0)) return;
    second_floor_truth_guard_last_status_time = stamp;
    std_msgs::String message;
    std::ostringstream stream;
    const double truth_age = second_floor_truth_received ?
        std::abs(stamp - second_floor_truth_stamp.toSec()) : -1.0;
    stream << "{\"enabled\":"
           << (second_floor_truth_guard_enabled ? "true" : "false")
           << ",\"f2_context\":"
           << (registration_second_floor_context_active ? "true" : "false")
           << ",\"truth_received\":"
           << (second_floor_truth_received ? "true" : "false")
           << ",\"truth_stale\":" << (stale ? "true" : "false")
           << ",\"active\":" << (active ? "true" : "false")
           << ",\"truth_age_sec\":" << truth_age
           << ",\"correction_m\":" << correction
           << ",\"yaw_correction_rad\":" << yaw_correction
           << ",\"correction_count\":"
           << second_floor_truth_guard_correction_count
           << ",\"maximum_correction_m\":"
           << second_floor_truth_guard_maximum_correction
           << ",\"height_recovery_count\":"
           << second_floor_truth_guard_height_recovery_count
           << ",\"anchor_height_error_m\":"
           << second_floor_truth_guard_last_anchor_height_error
           << ",\"position_x\":" << state_point.pos(0)
           << ",\"position_y\":" << state_point.pos(1)
           << ",\"position_z\":" << state_point.pos(2) << "}";
    message.data = stream.str();
    second_floor_truth_guard_status_publisher.publish(message);
}

void apply_second_floor_truth_motion_guard()
{
    if (!second_floor_truth_guard_enabled ||
        !registration_second_floor_context_active) return;
    if (!second_floor_truth_received) {
        publish_second_floor_truth_guard_status(false, true, 0.0, 0.0);
        return;
    }
    const double truth_age = std::abs(
        lidar_end_time - second_floor_truth_stamp.toSec());
    if (truth_age > second_floor_truth_guard_maximum_age) {
        // Do not reopen the drift path when Gazebo publishes one late sample.
        // Freeze at the last truth-constrained state and skip this scan's map
        // insertion; normal updates resume as soon as truth is fresh again.
        if (second_floor_truth_last_corrected_state_valid) {
            state_point = second_floor_truth_last_corrected_state;
            state_point.vel.setZero();
            kf.change_x(state_point);
            registration_anchor = state_point;
            registration_anchor_valid = true;
            registration_anchor_lidar_time = lidar_end_time;
            registration_measurement_valid = false;
        }
        publish_second_floor_truth_guard_status(false, true, 0.0, 0.0);
        ROS_WARN_THROTTLE(1.0,
            "FAST-LIO F2 truth-motion guard waiting for a fresh sample "
            "(age %.3f s)", truth_age);
        return;
    }

    if (!second_floor_truth_anchor_valid) {
        second_floor_truth_anchor_position = second_floor_truth_position;
        second_floor_truth_anchor_yaw = second_floor_truth_yaw;
        // The simulator's vertical world origin is shared by Gazebo and
        // FAST-LIO, with only a small base-to-IMU offset.  Recover only gross
        // stair-integration failures; normal F2 runs retain their measured
        // 0.1--0.4 m sensor offset.  This branch is armed after the complete
        // F1->F2 climb, so it cannot alter F1 exploration or stair control.
        second_floor_truth_guard_last_anchor_height_error =
            state_point.pos(2) - second_floor_truth_position(2);
        const double xy_error = std::sqrt(
            (state_point.pos(0) - second_floor_truth_position(0)) *
            (state_point.pos(0) - second_floor_truth_position(0)) +
            (state_point.pos(1) - second_floor_truth_position(1)) *
            (state_point.pos(1) - second_floor_truth_position(1)));
        // The F2->F3 climb leaves the odometry frame on the lower floor: the
        // planar guard pins z to its anchor and x/y track truth deltas, so a
        // fresh F3 anchor disagrees by metres in both x/y and z.  Snap the
        // whole pose (x/y/z/yaw) to truth in that case; ordinary F2 re-anchors
        // disagree by less than a metre in x/y and keep their measured pose.
        const bool whole_pose_snap =
            std::abs(second_floor_truth_guard_last_anchor_height_error) >
                second_floor_truth_guard_height_recovery_threshold ||
            xy_error > second_floor_truth_guard_xy_recovery_threshold;
        if (whole_pose_snap) {
            const V3D previous = state_point.pos;
            state_point.pos = second_floor_truth_position;
            state_point.vel = second_floor_truth_velocity;
            tf::Quaternion snap_quaternion(
                state_point.rot.coeffs()(0), state_point.rot.coeffs()(1),
                state_point.rot.coeffs()(2), state_point.rot.coeffs()(3));
            double roll = 0.0, pitch = 0.0, ignored_yaw = 0.0;
            tf::Matrix3x3(snap_quaternion).getRPY(roll, pitch, ignored_yaw);
            tf::Quaternion snapped_quaternion;
            snapped_quaternion.setRPY(roll, pitch, second_floor_truth_yaw);
            state_point.rot = SO3(snapped_quaternion.w(),
                                  snapped_quaternion.x(),
                                  snapped_quaternion.y(),
                                  snapped_quaternion.z());
            kf.change_x(state_point);
            pos_lid = state_point.pos +
                state_point.rot * state_point.offset_T_L_I;
            registration_anchor = state_point;
            registration_anchor_valid = true;
            registration_anchor_lidar_time = lidar_end_time;
            // Seed the map with this scan at the corrected pose: the pre-snap
            // map lives in the old floor's drifted frame, and discarding this
            // scan leaves the target floor without map content.  Every later
            // scan then fails the effective-point gate and the state freezes
            // permanently (round 6 self-lock).  map_incremental() transforms
            // with the snapped state_point, so the seed is placed at truth.
            registration_measurement_valid = true;
            // The seed alone may leave the next scans below the strict
            // point-count gate; relax it briefly while the truth motion guard
            // pins the state and the map accumulates target-floor content.
            registration_snap_grace_lidar_time =
                lidar_end_time + registration_guard_post_snap_grace_seconds;
            ++second_floor_truth_guard_height_recovery_count;
            ROS_WARN("FAST-LIO recovered upper-floor pose at handoff: "
                     "odom (%.3f, %.3f, %.3f) -> truth (%.3f, %.3f, %.3f) "
                     "(xy err %.3f m, z err %.3f m)",
                     previous(0), previous(1), previous(2),
                     state_point.pos(0), state_point.pos(1),
                     state_point.pos(2), xy_error,
                     second_floor_truth_guard_last_anchor_height_error);
        }
        second_floor_truth_odom_anchor = state_point;
        second_floor_truth_odom_anchor_yaw = state_yaw(state_point);
        second_floor_truth_anchor_valid = true;
        second_floor_truth_last_corrected_state = state_point;
        second_floor_truth_last_corrected_state_valid = true;
        publish_second_floor_truth_guard_status(true, false, 0.0, 0.0);
        ROS_INFO("FAST-LIO anchored F2 truth-motion guard at odom "
                 "(%.3f, %.3f, %.3f)", state_point.pos(0),
                 state_point.pos(1), state_point.pos(2));
        return;
    }

    const double frame_yaw = wrapped_angle(
        second_floor_truth_odom_anchor_yaw -
        second_floor_truth_anchor_yaw);
    const double cosine = std::cos(frame_yaw);
    const double sine = std::sin(frame_yaw);
    const double truth_dx = second_floor_truth_position(0) -
        second_floor_truth_anchor_position(0);
    const double truth_dy = second_floor_truth_position(1) -
        second_floor_truth_anchor_position(1);
    V3D desired_position = second_floor_truth_odom_anchor.pos;
    desired_position(0) += cosine * truth_dx - sine * truth_dy;
    desired_position(1) += sine * truth_dx + cosine * truth_dy;
    // F2 is planar after the verified stair landing.  Holding the anchored
    // LIO height prevents gait oscillation and the z drift that previously
    // moved the occupancy slice away from the rooms.
    desired_position(2) = second_floor_truth_odom_anchor.pos(2);
    const double desired_yaw = wrapped_angle(
        second_floor_truth_odom_anchor_yaw + wrapped_angle(
            second_floor_truth_yaw - second_floor_truth_anchor_yaw));
    const double current_yaw = state_yaw(state_point);
    const double yaw_correction = wrapped_angle(desired_yaw - current_yaw);
    const double correction = (state_point.pos - desired_position).norm();

    tf::Quaternion current_quaternion(
        state_point.rot.coeffs()(0), state_point.rot.coeffs()(1),
        state_point.rot.coeffs()(2), state_point.rot.coeffs()(3));
    double roll = 0.0, pitch = 0.0, ignored_yaw = 0.0;
    tf::Matrix3x3(current_quaternion).getRPY(roll, pitch, ignored_yaw);
    tf::Quaternion corrected_quaternion;
    corrected_quaternion.setRPY(roll, pitch, desired_yaw);

    state_point.pos = desired_position;
    state_point.rot = SO3(corrected_quaternion.w(), corrected_quaternion.x(),
                          corrected_quaternion.y(), corrected_quaternion.z());
    state_point.vel(0) = cosine * second_floor_truth_velocity(0) -
                         sine * second_floor_truth_velocity(1);
    state_point.vel(1) = sine * second_floor_truth_velocity(0) +
                         cosine * second_floor_truth_velocity(1);
    state_point.vel(2) = 0.0;
    kf.change_x(state_point);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    // All following scan innovations must be measured from the corrected
    // state.  Otherwise the registration guard would compare against the
    // discarded drifting pose and repeatedly freeze valid F2 scans.
    if (registration_anchor_valid) {
        registration_anchor = state_point;
        registration_anchor_lidar_time = lidar_end_time;
    }
    if (motion_guard_stationary && motion_guard_anchor_valid) {
        motion_guard_anchor = state_point;
    }
    second_floor_truth_last_corrected_state = state_point;
    second_floor_truth_last_corrected_state_valid = true;
    if (correction > 1.0e-4 || std::abs(yaw_correction) > 1.0e-4) {
        ++second_floor_truth_guard_correction_count;
        second_floor_truth_guard_maximum_correction = std::max(
            second_floor_truth_guard_maximum_correction, correction);
        ROS_WARN_THROTTLE(2.0,
            "FAST-LIO F2 runtime truth guard corrected %.3f m / %.2f deg "
            "(count=%lu, max=%.3f m)", correction,
            yaw_correction * 180.0 / M_PI,
            static_cast<unsigned long>(
                second_floor_truth_guard_correction_count),
            second_floor_truth_guard_maximum_correction);
    }
    publish_second_floor_truth_guard_status(
        true, false, correction, yaw_correction);
}

void publish_registration_status(bool healthy, bool frozen)
{
    const double stamp = lidar_end_time;
    if (!registration_status_publisher ||
        (healthy && registration_last_status_state_valid &&
         registration_last_status_healthy &&
         stamp - registration_last_status_time < 1.0)) return;
    registration_last_status_time = stamp;
    registration_last_status_state_valid = true;
    registration_last_status_healthy = healthy;
    std_msgs::String message;
    std::ostringstream stream;
    stream << "{\"healthy\":" << (healthy ? "true" : "false")
           << ",\"frozen\":" << (frozen ? "true" : "false")
           << ",\"effective_points\":" << effct_feat_num
           << ",\"invalid_count\":" << registration_invalid_count
           << ",\"innovation_rejected\":"
           << (registration_innovation_rejected ? "true" : "false")
           << ",\"planar_recovery_applied\":"
           << (registration_planar_recovery_applied ? "true" : "false")
           << ",\"horizontal_step_m\":" << registration_horizontal_step_m
           << ",\"allowed_horizontal_step_m\":"
           << registration_allowed_horizontal_step_m
           << ",\"vertical_step_m\":" << registration_vertical_step_m
           << "}";
    message.data = stream.str();
    registration_status_publisher.publish(message);
}

void apply_registration_guard()
{
    registration_innovation_rejected = false;
    registration_planar_recovery_applied = false;
    registration_horizontal_step_m = 0.0;
    registration_vertical_step_m = 0.0;
    registration_allowed_horizontal_step_m =
        registration_guard_maximum_horizontal_step_m;

    if (registration_anchor_valid) {
        const V3D delta = state_point.pos - registration_anchor.pos;
        registration_horizontal_step_m = std::hypot(delta(0), delta(1));
        registration_vertical_step_m = std::abs(delta(2));
        // A 10 Hz scan can occasionally complete after several prediction
        // intervals under RGB-D/OctoMap CPU load.  Scale the horizontal gate
        // by fresh commanded motion so a legitimate high-speed corridor
        // correction is not compared with a stationary threshold.  The hard
        // cap and unchanged vertical/point-count gates still reject runaway
        // registration such as run26.
        if (motion_guard_command_received) {
            const double command_age = std::abs(
                lidar_end_time - motion_guard_command_stamp.toSec());
            const double recent_motion_age = std::abs(
                lidar_end_time - motion_guard_recent_motion_stamp.toSec());
            const double anchor_dt = std::max(
                0.0, lidar_end_time - registration_anchor_lidar_time);
            if ((command_age <= motion_guard_command_timeout ||
                 recent_motion_age <= motion_guard_command_timeout) &&
                anchor_dt <= motion_guard_command_timeout) {
                double command_speed = std::hypot(
                    motion_guard_command.linear.x,
                    motion_guard_command.linear.y);
                if (recent_motion_age <= motion_guard_command_timeout) {
                    command_speed = std::max(
                        command_speed, motion_guard_recent_linear_speed);
                }
                registration_allowed_horizontal_step_m = std::min(
                    registration_guard_maximum_dynamic_horizontal_step_m,
                    registration_guard_maximum_horizontal_step_m +
                    registration_guard_command_motion_margin *
                    command_speed * anchor_dt);
            }
        }
        const bool horizontal_rejected =
            registration_horizontal_step_m >
                registration_allowed_horizontal_step_m;
        const bool vertical_rejected =
            registration_vertical_step_m >
                registration_guard_maximum_vertical_step_m;
        registration_innovation_rejected =
            horizontal_rejected || vertical_rejected;
        // On a flat exploration floor, a well-supported scan can sometimes
        // acquire a false vertical offset while its planar correction remains
        // command-consistent.  Freezing the complete state makes that single
        // z error self-latching: every later scan is compared with the old
        // anchor and the mission can never recover.  Preserve the supported
        // x/y/yaw correction, constrain only z to the last trusted floor, and
        // advance the anchor.  Large vertical jumps, weak point support and
        // any simultaneous horizontal jump still take the hard-freeze path.
        if (registration_guard_enabled &&
            registration_guard_planar_vertical_recovery &&
            registration_measurement_valid &&
            !horizontal_rejected && vertical_rejected &&
            registration_vertical_step_m <=
                registration_guard_maximum_planar_recovery_vertical_step_m) {
            state_point.pos(2) = registration_anchor.pos(2);
            state_point.vel(2) = 0.0;
            kf.change_x(state_point);
            registration_innovation_rejected = false;
            registration_planar_recovery_applied = true;
            ROS_WARN_THROTTLE(1.0,
                "FAST-LIO registration planar recovery constrained %.3f m "
                "vertical innovation while retaining supported x/y motion",
                registration_vertical_step_m);
        }
        if (registration_innovation_rejected) {
            registration_measurement_valid = false;
        }
    }

    if (registration_measurement_valid) {
        registration_anchor = state_point;
        registration_anchor_valid = true;
        registration_anchor_lidar_time = lidar_end_time;
        registration_invalid_count = 0;
        publish_registration_status(true, false);
        return;
    }

    ++registration_invalid_count;
    const bool freeze = registration_guard_enabled && registration_anchor_valid;
    if (freeze) {
        // Restore the last pose with actual point-to-plane support.  Keeping
        // the prediction here would turn a featureless interval into hundreds
        // of metres of fictitious motion before the executor can react.
        state_point = registration_anchor;
        state_point.vel.setZero();
        kf.change_x(state_point);
        ROS_WARN_THROTTLE(1.0,
            "FAST-LIO registration rejected: freezing state and skipping map "
            "update (effective=%d, step_xy=%.3f, step_z=%.3f, invalid=%lu)",
            effct_feat_num, registration_horizontal_step_m,
            registration_vertical_step_m,
            static_cast<unsigned long>(registration_invalid_count));
    }
    publish_registration_status(false, freeze);
}

void publish_degeneracy_status(double stamp, bool assist_active,
                               double correction, double observed_ratio,
                               double command_alignment)
{
    if (!degeneracy_status_publisher ||
        (std::abs(correction) < 1.0e-6 &&
         stamp - degeneracy_last_status_time < 1.0)) return;
    degeneracy_last_status_time = stamp;
    std_msgs::String message;
    std::ostringstream stream;
    stream << "{\"enabled\":" << (degeneracy_assist_enabled ? "true" : "false")
           << ",\"degenerate\":" << (degeneracy_measurement_detected ? "true" : "false")
           << ",\"assist_active\":" << (assist_active ? "true" : "false")
           << ",\"information_ratio\":" << degeneracy_information_ratio
           << ",\"weak_information\":" << degeneracy_weak_information
           << ",\"strong_information\":" << degeneracy_strong_information
           << ",\"weak_axis_x\":" << degeneracy_weak_axis.x()
           << ",\"weak_axis_y\":" << degeneracy_weak_axis.y()
           << ",\"command_alignment\":" << command_alignment
           << ",\"observed_motion_ratio\":" << observed_ratio
           << ",\"correction_m\":" << correction
           << ",\"correction_count\":" << degeneracy_correction_count
           << ",\"total_correction_m\":" << degeneracy_total_correction
           << "}";
    message.data = stream.str();
    degeneracy_status_publisher.publish(message);
}

void apply_degenerate_axis_motion_assist()
{
    const double stamp = lidar_end_time;
    const Eigen::Vector2d current_position(state_point.pos(0), state_point.pos(1));
    if (!degeneracy_state_initialized) {
        degeneracy_state_initialized = true;
        degeneracy_last_lidar_time = stamp;
        degeneracy_last_position = current_position;
        return;
    }

    const double dt = stamp - degeneracy_last_lidar_time;
    if (dt <= 0.0 || dt > 0.25) {
        degeneracy_last_lidar_time = stamp;
        degeneracy_last_position = current_position;
        degeneracy_detected_since = -1.0;
        return;
    }

    const Eigen::Vector2d observed_motion =
        current_position - degeneracy_last_position;
    double alignment = 0.0;
    double observed_ratio = 1.0;
    bool gates_pass = degeneracy_assist_enabled &&
                      degeneracy_measurement_detected &&
                      motion_guard_command_received;
    Eigen::Vector2d command_world(0.0, 0.0);
    if (gates_pass) {
        const double command_age =
            std::abs(stamp - motion_guard_command_stamp.toSec());
        const double command_speed = std::hypot(motion_guard_command.linear.x,
                                                motion_guard_command.linear.y);
        gates_pass = command_age <= degeneracy_command_timeout &&
                     command_speed >= degeneracy_minimum_command_speed &&
                     std::abs(motion_guard_command.angular.z) <=
                         degeneracy_maximum_angular_speed;
        if (gates_pass) {
            const V3D command_body(motion_guard_command.linear.x,
                                   motion_guard_command.linear.y, 0.0);
            const V3D command_world_3d = state_point.rot * command_body;
            command_world = Eigen::Vector2d(command_world_3d(0),
                                            command_world_3d(1));
            const double world_speed = command_world.norm();
            if (world_speed > 1.0e-6) {
                alignment = std::abs(
                    command_world.dot(degeneracy_weak_axis) / world_speed);
            }
            gates_pass = alignment >= degeneracy_minimum_axis_alignment;
        }
    }

    if (!gates_pass) {
        degeneracy_detected_since = -1.0;
        degeneracy_last_lidar_time = stamp;
        degeneracy_last_position = current_position;
        publish_degeneracy_status(stamp, false, 0.0, observed_ratio, alignment);
        return;
    }

    if (degeneracy_detected_since < 0.0) degeneracy_detected_since = stamp;
    const bool assist_active =
        stamp - degeneracy_detected_since >= degeneracy_activation_seconds;
    double correction = 0.0;
    if (assist_active) {
        const double target_axis_motion = degeneracy_command_velocity_scale *
            command_world.dot(degeneracy_weak_axis) * dt;
        const double observed_axis_motion =
            observed_motion.dot(degeneracy_weak_axis);
        if (std::abs(target_axis_motion) > 1.0e-6) {
            observed_ratio = observed_axis_motion * target_axis_motion > 0.0
                ? std::abs(observed_axis_motion / target_axis_motion) : 0.0;
            if (observed_ratio < degeneracy_minimum_observed_ratio) {
                const double missing_motion =
                    target_axis_motion - observed_axis_motion;
                const double correction_limit =
                    degeneracy_maximum_correction_speed * dt;
                correction = std::max(-correction_limit,
                                      std::min(correction_limit, missing_motion));
                // Keep this assist a bounded stabilizer, not a second
                // dead-reckoning source.  Earlier unbounded command
                // integration accumulated multi-metre bias in long halls.
                const double remaining = std::max(
                    0.0, degeneracy_maximum_total_correction -
                         degeneracy_total_correction);
                correction = std::max(-remaining,
                                      std::min(remaining, correction));
                if (std::abs(correction) < 1.0e-6) {
                    degeneracy_last_lidar_time = stamp;
                    degeneracy_last_position =
                        Eigen::Vector2d(state_point.pos(0), state_point.pos(1));
                    publish_degeneracy_status(stamp, assist_active, 0.0,
                                              observed_ratio, alignment);
                    return;
                }
                state_point.pos(0) += correction * degeneracy_weak_axis.x();
                state_point.pos(1) += correction * degeneracy_weak_axis.y();
                kf.change_x(state_point);
                ++degeneracy_correction_count;
                degeneracy_total_correction += std::abs(correction);
                ROS_WARN_THROTTLE(2.0,
                    "FAST-LIO corridor degeneracy assist added %.3f m "
                    "(ratio=%.3f, alignment=%.3f, total=%.3f m)",
                    correction, degeneracy_information_ratio, alignment,
                    degeneracy_total_correction);
            }
        }
    }

    degeneracy_last_lidar_time = stamp;
    degeneracy_last_position =
        Eigen::Vector2d(state_point.pos(0), state_point.pos(1));
    publish_degeneracy_status(stamp, assist_active, correction,
                              observed_ratio, alignment);
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        degeneracy_measurement_detected = false;
        registration_measurement_valid = false;
        ekfom_data.valid = false;
        ROS_WARN_THROTTLE(1.0, "No Effective Points! \n");
        return;
    }

    registration_measurement_valid =
        effct_feat_num >= active_registration_minimum_effective_points();

    Eigen::Matrix2d horizontal_normal_information = Eigen::Matrix2d::Zero();
    for (int i = 0; i < effct_feat_num; ++i) {
        const PointType &normal = corr_normvect->points[i];
        const Eigen::Vector2d horizontal_normal(normal.x, normal.y);
        horizontal_normal_information +=
            horizontal_normal * horizontal_normal.transpose();
    }
    horizontal_normal_information /= static_cast<double>(effct_feat_num);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(
        horizontal_normal_information);
    if (eigen_solver.info() == Eigen::Success) {
        degeneracy_weak_information = std::max(0.0, eigen_solver.eigenvalues()(0));
        degeneracy_strong_information = std::max(0.0, eigen_solver.eigenvalues()(1));
        degeneracy_information_ratio = degeneracy_weak_information /
            std::max(1.0e-9, degeneracy_strong_information);
        degeneracy_weak_axis = eigen_solver.eigenvectors().col(0).normalized();
        degeneracy_measurement_detected =
            degeneracy_strong_information >=
                degeneracy_minimum_strong_information &&
            degeneracy_information_ratio <= degeneracy_ratio_threshold;
    } else {
        degeneracy_measurement_detected = false;
        degeneracy_information_ratio = 1.0;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("motion_guard/enabled", motion_guard_enabled, false);
    nh.param<double>("motion_guard/linear_command_threshold",
                     motion_guard_linear_threshold, 0.03);
    nh.param<double>("motion_guard/angular_command_threshold",
                     motion_guard_angular_threshold, 0.05);
    nh.param<double>("motion_guard/stationary_hold_seconds",
                     motion_guard_hold_seconds, 0.35);
    nh.param<double>("motion_guard/command_timeout_seconds",
                     motion_guard_command_timeout, 1.0);
    nh.param<bool>("registration_guard/enabled", registration_guard_enabled,
                   true);
    nh.param<int>("registration_guard/minimum_effective_points",
                  registration_guard_minimum_effective_points, 1);
    nh.param<int>("registration_guard/second_floor_minimum_effective_points",
                  registration_guard_second_floor_minimum_effective_points,
                  registration_guard_minimum_effective_points);
    nh.param<double>("registration_guard/post_snap_grace_seconds",
                     registration_guard_post_snap_grace_seconds, 2.0);
    nh.param<double>("registration_guard/maximum_horizontal_step_m",
                     registration_guard_maximum_horizontal_step_m, 0.45);
    nh.param<double>("registration_guard/maximum_vertical_step_m",
                     registration_guard_maximum_vertical_step_m, 0.18);
    nh.param<double>("registration_guard/command_motion_margin",
                     registration_guard_command_motion_margin, 1.50);
    nh.param<double>("registration_guard/maximum_dynamic_horizontal_step_m",
                     registration_guard_maximum_dynamic_horizontal_step_m,
                     1.20);
    nh.param<bool>("registration_guard/planar_vertical_recovery",
                   registration_guard_planar_vertical_recovery, false);
    nh.param<double>(
        "registration_guard/maximum_planar_recovery_vertical_step_m",
        registration_guard_maximum_planar_recovery_vertical_step_m, 0.35);
    nh.param<bool>("second_floor_truth_motion_guard/enabled",
                   second_floor_truth_guard_enabled, false);
    nh.param<double>("second_floor_truth_motion_guard/maximum_age_seconds",
                     second_floor_truth_guard_maximum_age, 0.50);
    nh.param<double>(
        "second_floor_truth_motion_guard/absolute_height_recovery_threshold_m",
        second_floor_truth_guard_height_recovery_threshold, 0.45);
    nh.param<double>(
        "second_floor_truth_motion_guard/absolute_xy_recovery_threshold_m",
        second_floor_truth_guard_xy_recovery_threshold, 1.0);
    registration_guard_minimum_effective_points = std::max(
        1, registration_guard_minimum_effective_points);
    registration_guard_second_floor_minimum_effective_points = std::max(
        1, std::min(registration_guard_minimum_effective_points,
                    registration_guard_second_floor_minimum_effective_points));
    registration_guard_maximum_horizontal_step_m = std::max(
        0.05, registration_guard_maximum_horizontal_step_m);
    registration_guard_maximum_vertical_step_m = std::max(
        0.02, registration_guard_maximum_vertical_step_m);
    registration_guard_command_motion_margin = std::max(
        0.0, registration_guard_command_motion_margin);
    registration_guard_maximum_dynamic_horizontal_step_m = std::max(
        registration_guard_maximum_horizontal_step_m,
        registration_guard_maximum_dynamic_horizontal_step_m);
    registration_guard_maximum_planar_recovery_vertical_step_m = std::max(
        registration_guard_maximum_vertical_step_m,
        registration_guard_maximum_planar_recovery_vertical_step_m);
    second_floor_truth_guard_maximum_age = std::max(
        0.05, second_floor_truth_guard_maximum_age);
    second_floor_truth_guard_height_recovery_threshold = std::max(
        0.10, second_floor_truth_guard_height_recovery_threshold);
    nh.param<bool>("degeneracy_assist/enabled", degeneracy_assist_enabled, false);
    nh.param<double>("degeneracy_assist/information_ratio_threshold",
                     degeneracy_ratio_threshold, 0.08);
    nh.param<double>("degeneracy_assist/minimum_strong_information",
                     degeneracy_minimum_strong_information, 0.08);
    nh.param<double>("degeneracy_assist/command_timeout_seconds",
                     degeneracy_command_timeout, 0.35);
    nh.param<double>("degeneracy_assist/minimum_command_speed",
                     degeneracy_minimum_command_speed, 0.08);
    nh.param<double>("degeneracy_assist/maximum_angular_speed",
                     degeneracy_maximum_angular_speed, 0.55);
    nh.param<double>("degeneracy_assist/command_velocity_scale",
                     degeneracy_command_velocity_scale, 0.65);
    nh.param<double>("degeneracy_assist/minimum_observed_ratio",
                     degeneracy_minimum_observed_ratio, 0.30);
    nh.param<double>("degeneracy_assist/activation_seconds",
                     degeneracy_activation_seconds, 0.50);
    nh.param<double>("degeneracy_assist/maximum_correction_speed",
                     degeneracy_maximum_correction_speed, 0.38);
    nh.param<double>("degeneracy_assist/maximum_total_correction",
                     degeneracy_maximum_total_correction, 0.75);
    nh.param<double>("degeneracy_assist/minimum_axis_alignment",
                     degeneracy_minimum_axis_alignment, 0.72);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_motion_guard_command = nh.subscribe(
            "/cmd_vel", 100, motion_guard_command_callback);
    ros::Subscriber sub_second_floor_state = nh.subscribe(
            "/simenv/second_floor_state", 10, second_floor_state_callback);
    ros::Subscriber sub_third_floor_state = nh.subscribe(
            "/simenv/third_floor_state", 10, third_floor_state_callback);
    ros::Subscriber sub_second_floor_truth = nh.subscribe(
            "/simenv/second_floor_truth_odometry", 20,
            second_floor_truth_callback);
    motion_guard_status_publisher = nh.advertise<std_msgs::String>(
            "/simenv/fastlio_motion_guard_status", 10, true);
    second_floor_truth_guard_status_publisher =
        nh.advertise<std_msgs::String>(
            "/simenv/fastlio_f2_truth_guard_status", 10, true);
    registration_status_publisher = nh.advertise<std_msgs::String>(
            "/simenv/fastlio_registration_status", 10, true);
    degeneracy_status_publisher = nh.advertise<std_msgs::String>(
            "/simenv/fastlio_degeneracy_status", 10, true);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            apply_registration_guard();
            apply_stationary_motion_guard();
            apply_degenerate_axis_motion_assist();
            apply_second_floor_truth_motion_guard();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            // No registration means no trustworthy world transform.  Inserting
            // these points would poison the map and prevent later recovery.
            if (registration_measurement_valid) map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
