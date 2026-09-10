bool fall_quarantine_command_quiet()
{
    if (!motion_guard_command_received) return false;
    const double command_age = std::abs(
        lidar_end_time - motion_guard_command_stamp.toSec());
    return command_age <= motion_guard_command_timeout &&
        std::hypot(motion_guard_command.linear.x,
                   motion_guard_command.linear.y) <=
            motion_guard_linear_threshold &&
        std::abs(motion_guard_command.angular.z) <=
            motion_guard_angular_threshold;
}

void publish_fall_quarantine_status(bool force)
{
    if (!fall_quarantine_status_publisher) return;
    const double stamp = lidar_end_time > 0.0 ?
        lidar_end_time : ros::Time::now().toSec();
    if (!force && fall_quarantine_last_status_time >= 0.0 &&
        stamp - fall_quarantine_last_status_time < 1.0) return;
    fall_quarantine_last_status_time = stamp;

    std_msgs::String message;
    std::ostringstream stream;
    stream << "{\"enabled\":"
           << (fall_quarantine_enabled ? "true" : "false")
           << ",\"phase\":\""
           << fast_lio_fall_quarantine::Policy::phase_name(
                  fall_quarantine_policy.phase())
           << "\",\"active\":"
           << (fall_quarantine_policy.active() ? "true" : "false")
           << ",\"warming_up\":"
           << (fall_quarantine_policy.warming_up() ? "true" : "false")
           << ",\"frozen_state_valid\":"
           << (fall_quarantine_frozen_state_valid ? "true" : "false")
           << ",\"trusted_state_valid\":"
           << (fall_quarantine_last_trusted_state_valid ? "true" : "false")
           << ",\"clean_scans\":"
           << fall_quarantine_policy.consecutive_clean_scans()
           << ",\"required_clean_scans\":"
           << fall_quarantine_policy.required_clean_scans()
           << ",\"last_registration_valid\":"
           << (fall_quarantine_last_registration_valid ? "true" : "false")
           << ",\"last_truth_correction_frame\":"
           << (fall_quarantine_last_truth_correction_frame ? "true" : "false")
           << ",\"active_entries\":"
           << fall_quarantine_policy.active_entries()
           << ",\"warmup_entries\":"
           << fall_quarantine_policy.warmup_entries()
           << ",\"active_packets\":"
           << fall_quarantine_policy.active_packets()
           << ",\"invalid_resets\":"
           << fall_quarantine_policy.invalid_resets()
           << ",\"correction_resets\":"
           << fall_quarantine_policy.correction_resets()
           << ",\"release_count\":"
           << fall_quarantine_policy.release_count()
           << ",\"frozen_odom_count\":"
           << fall_quarantine_frozen_odom_count
           << ",\"blocked_root_build_count\":"
           << fall_quarantine_blocked_root_build_count
           << ",\"empty_map_bootstrap_count\":"
           << fall_quarantine_empty_map_bootstrap_count
           << ",\"blocked_map_count\":"
           << fall_quarantine_blocked_map_count
           << ",\"blocked_cloud_count\":"
           << fall_quarantine_blocked_cloud_count
           << "}";
    message.data = stream.str();
    fall_quarantine_status_publisher.publish(message);
}

void fall_recovery_active_callback(const std_msgs::Bool::ConstPtr &message)
{
    fall_recovery_signal_active = message->data;
    if (!fall_quarantine_enabled) return;
    const fast_lio_fall_quarantine::SignalTransition transition =
        fall_quarantine_policy.observe_fall_signal(message->data);
    if (transition.entered_active) {
        // The compliant reset preserves the physical world x/y/yaw.  In
        // continuous planar-truth mode the synchronized pre-fall SE(2) pair
        // is therefore still the only trustworthy frame relationship.  Do
        // not discard it here: repeated corridor geometry can otherwise let
        // the first post-reset ICP frames establish a completely unrelated
        // anchor (f3anchorsync observed 13.7,-2.2,-99.7 deg), after which
        // continuous truth faithfully preserves the *wrong* relationship.
        // Keeping the proven pair lets apply_requested_truth_recovery_guard()
        // restore the live truth displacement before odometry/map output on
        // the first warm-up scan.  One-shot FAST-LIO mode retains the old
        // fail-closed behaviour and must establish a fresh quiet anchor.
        if (!truth_recovery_continuous_planar_localization)
            second_floor_truth_anchor_valid = false;
        // The continuously maintained trusted state is preferred. Before the
        // first clean registration frame, the current filter state is the
        // only available fallback and is captured exactly on the rising edge.
        fall_quarantine_frozen_state =
            fall_quarantine_last_trusted_state_valid ?
            fall_quarantine_last_trusted_state : state_point;
        fall_quarantine_frozen_state.vel.setZero();
        fall_quarantine_frozen_state_valid = true;
        truth_recovery_request_pending = false;
        truth_recovery_request_registration_fault = false;
        truth_recovery_request_anchored_se2_fault = false;
        motion_guard_stationary = false;
        motion_guard_anchor_valid = false;
        ROS_WARN("FAST-LIO fall quarantine ACTIVE: saved last trusted pose; "
                 "map and registered clouds are frozen");
        publish_fall_quarantine_status(true);
    } else if (transition.entered_warmup) {
        ROS_WARN("FAST-LIO fall quarantine WARMUP: requiring %d consecutive "
                 "clean registration scans; no timeout fail-open",
                 fall_quarantine_policy.required_clean_scans());
        publish_fall_quarantine_status(true);
    }
}

bool handle_active_fall_quarantine_packet(
    const ros::Publisher &pub_odom_after_mapped)
{
    if (!fall_quarantine_enabled || !fall_quarantine_policy.active())
        return false;
    if (!fall_quarantine_frozen_state_valid) {
        fall_quarantine_frozen_state = state_point;
        fall_quarantine_frozen_state.vel.setZero();
        fall_quarantine_frozen_state_valid = true;
    }

    // p_imu->Process has already consumed this packet and advanced all sensor
    // timestamps. Restore the saved pose before any FOV segmentation, root
    // build or ICP so a fallen-body scan can never acquire a world transform.
    state_point = fall_quarantine_frozen_state;
    state_point.vel.setZero();
    kf.change_x(state_point);
    registration_measurement_valid = false;
    truth_recovery_correction_active = false;
    pos_lid = state_point.pos +
        state_point.rot * state_point.offset_T_L_I;
    euler_cur = SO3ToEuler(state_point.rot);
    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];
    if (ikdtree.Root_Node == nullptr)
        ++fall_quarantine_blocked_root_build_count;
    fall_quarantine_policy.note_active_packet();
    ++fall_quarantine_frozen_odom_count;
    publish_odometry(pub_odom_after_mapped);
    publish_fall_quarantine_status(false);
    return true;
}

bool observe_fall_quarantine_registration_frame()
{
    fall_quarantine_last_registration_valid =
        registration_measurement_valid;
    fall_quarantine_last_truth_correction_frame =
        truth_recovery_correction_active;
    const fast_lio_fall_quarantine::FrameDecision decision =
        fall_quarantine_policy.observe_registration(
            registration_measurement_valid,
            truth_recovery_correction_active);
    if (decision.just_released) {
        ROS_WARN("FAST-LIO fall quarantine OPEN after %d consecutive clean "
                 "registration scans",
                 fall_quarantine_policy.required_clean_scans());
        publish_fall_quarantine_status(true);
    } else if (decision.invalid_reset || decision.correction_reset) {
        ROS_WARN_THROTTLE(1.0,
            "FAST-LIO fall quarantine warmup reset (%s); clean=%d/%d",
            decision.correction_reset ? "truth_correction" :
                                        "invalid_registration",
            fall_quarantine_policy.consecutive_clean_scans(),
            fall_quarantine_policy.required_clean_scans());
        publish_fall_quarantine_status(false);
    }

    if (decision.clean_scan &&
        fall_quarantine_policy.phase() ==
            fast_lio_fall_quarantine::Phase::kOpen) {
        fall_quarantine_last_trusted_state = state_point;
        fall_quarantine_last_trusted_state_valid = true;
    }
    return decision.allow_map_and_cloud;
}

bool fall_quarantine_may_initialize_empty_map()
{
    const bool allowed = fall_quarantine_policy.may_initialize_empty_map(
        fall_quarantine_command_quiet());
    if (!allowed) {
        ++fall_quarantine_blocked_root_build_count;
        publish_fall_quarantine_status(false);
    }
    return allowed;
}

void note_fall_quarantine_empty_map_bootstrap()
{
    if (!fall_quarantine_policy.warming_up()) return;
    ++fall_quarantine_empty_map_bootstrap_count;
    publish_fall_quarantine_status(true);
}
