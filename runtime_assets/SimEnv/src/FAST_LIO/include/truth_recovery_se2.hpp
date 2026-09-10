#pragma once

#include <algorithm>
#include <cmath>

namespace fast_lio_truth_recovery {

struct Pose2d {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
};

inline Pose2d extrapolate_world_pose(const Pose2d &pose, double velocity_x,
                                     double velocity_y, double yaw_rate,
                                     double signed_dt, double maximum_abs_dt)
{
    const double bound = std::max(0.0, maximum_abs_dt);
    const double dt = std::max(-bound, std::min(bound, signed_dt));
    Pose2d extrapolated;
    extrapolated.x = pose.x + velocity_x * dt;
    extrapolated.y = pose.y + velocity_y * dt;
    extrapolated.yaw = std::atan2(
        std::sin(pose.yaw + yaw_rate * dt),
        std::cos(pose.yaw + yaw_rate * dt));
    return extrapolated;
}

inline double wrap_angle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

// Map truth motion into the existing FAST-LIO frame.  The two absolute
// frames are never equated: only the transform between synchronized anchors
// and the subsequent truth displacement are used.
inline Pose2d anchored_truth_target(const Pose2d &odom_anchor,
                                    const Pose2d &truth_anchor,
                                    const Pose2d &truth_pose)
{
    const double frame_yaw = wrap_angle(
        odom_anchor.yaw - truth_anchor.yaw);
    const double cosine = std::cos(frame_yaw);
    const double sine = std::sin(frame_yaw);
    const double truth_dx = truth_pose.x - truth_anchor.x;
    const double truth_dy = truth_pose.y - truth_anchor.y;
    Pose2d target;
    target.x = odom_anchor.x + cosine * truth_dx - sine * truth_dy;
    target.y = odom_anchor.y + sine * truth_dx + cosine * truth_dy;
    target.yaw = wrap_angle(
        odom_anchor.yaw + wrap_angle(truth_pose.yaw - truth_anchor.yaw));
    return target;
}

struct Residual2d {
    double planar_m = 0.0;
    double yaw_rad = 0.0;
};

inline Residual2d residual(const Pose2d &estimate, const Pose2d &target)
{
    Residual2d value;
    value.planar_m = std::hypot(
        estimate.x - target.x, estimate.y - target.y);
    value.yaw_rad = std::abs(wrap_angle(estimate.yaw - target.yaw));
    return value;
}


// A simulator-only cumulative innovation gate.  Unlike RecoveryGate, this
// policy is evaluated while the robot may still be moving: it only rejects the
// current LiDAR registration so the existing navigation safety path can stop
// the body.  It never authorizes a pose correction; RecoveryGate below remains
// responsible for the fresh-truth and quiet-body correction checks.
struct DivergenceGate {
    bool enabled = false;
    bool context_active = false;
    bool anchor_valid = false;
    bool truth_fresh = false;
    bool quarantine_open = false;
    double planar_residual_m = 0.0;
    double yaw_residual_rad = 0.0;
    double maximum_planar_residual_m = 0.0;
    double maximum_yaw_residual_rad = 0.0;
};

inline bool reject_divergent_registration(const DivergenceGate &gate)
{
    if (!gate.enabled || !gate.context_active || !gate.anchor_valid ||
        !gate.truth_fresh || !gate.quarantine_open)
        return false;
    return gate.planar_residual_m >= gate.maximum_planar_residual_m ||
        gate.yaw_residual_rad >= gate.maximum_yaw_residual_rad;
}

// Command-derived motion must never override a registration freeze later in
// the same frame. This predicate is intentionally fail-closed for every
// registration rejection source.
inline bool registration_supports_degeneracy_assist(
    bool measurement_valid, bool innovation_rejected,
    bool anchored_truth_divergence_rejected)
{
    return measurement_valid && !innovation_rejected &&
        !anchored_truth_divergence_rejected;
}

enum class RecoveryDecision {
    kApply,
    kDisabled,
    kNoContext,
    kNoRequest,
    kRequestExpired,
    kNoAnchor,
    kStaleTruth,
    kBodyMoving,
    kNoFaultEvidence,
    kResidualBelowThreshold,
    kCorrectionTooLarge,
};

struct RecoveryGate {
    bool enabled = false;
    bool context_active = false;
    bool request_pending = false;
    bool request_fresh = false;
    bool anchor_valid = false;
    bool truth_fresh = false;
    bool command_quiet = false;
    bool truth_quiet = false;
    bool registration_fault = false;
    bool request_reports_registration_fault = false;
    bool request_reports_anchored_se2_fault = false;
    double planar_residual_m = 0.0;
    double yaw_residual_rad = 0.0;
    double minimum_planar_residual_m = 0.0;
    double minimum_yaw_residual_rad = 0.0;
    double maximum_planar_correction_m = 0.0;
    double maximum_yaw_correction_rad = 0.0;
};

inline RecoveryDecision decide(const RecoveryGate &gate)
{
    if (!gate.enabled) return RecoveryDecision::kDisabled;
    if (!gate.context_active) return RecoveryDecision::kNoContext;
    if (!gate.request_pending) return RecoveryDecision::kNoRequest;
    if (!gate.request_fresh) return RecoveryDecision::kRequestExpired;
    if (!gate.anchor_valid) return RecoveryDecision::kNoAnchor;
    if (!gate.truth_fresh) return RecoveryDecision::kStaleTruth;
    if (!gate.command_quiet || !gate.truth_quiet)
        return RecoveryDecision::kBodyMoving;

    // A relative-pose disagreement is itself localization-fault evidence,
    // but it is trusted only when the monitor explicitly says that its
    // comparison used synchronized SE(2) anchors.  A generic registration
    // request additionally requires FAST-LIO to still be unhealthy now.
    const bool fault_evidence =
        gate.request_reports_anchored_se2_fault ||
        (gate.request_reports_registration_fault && gate.registration_fault);
    if (!fault_evidence) return RecoveryDecision::kNoFaultEvidence;

    if (gate.planar_residual_m < gate.minimum_planar_residual_m &&
        gate.yaw_residual_rad < gate.minimum_yaw_residual_rad)
        return RecoveryDecision::kResidualBelowThreshold;
    if (gate.planar_residual_m > gate.maximum_planar_correction_m ||
        gate.yaw_residual_rad > gate.maximum_yaw_correction_rad)
        return RecoveryDecision::kCorrectionTooLarge;
    return RecoveryDecision::kApply;
}

inline const char *decision_name(RecoveryDecision decision)
{
    switch (decision) {
    case RecoveryDecision::kApply: return "apply";
    case RecoveryDecision::kDisabled: return "disabled";
    case RecoveryDecision::kNoContext: return "no_context";
    case RecoveryDecision::kNoRequest: return "no_request";
    case RecoveryDecision::kRequestExpired: return "request_expired";
    case RecoveryDecision::kNoAnchor: return "no_anchor";
    case RecoveryDecision::kStaleTruth: return "stale_truth";
    case RecoveryDecision::kBodyMoving: return "body_moving";
    case RecoveryDecision::kNoFaultEvidence: return "no_fault_evidence";
    case RecoveryDecision::kResidualBelowThreshold:
        return "residual_below_threshold";
    case RecoveryDecision::kCorrectionTooLarge:
        return "correction_too_large";
    }
    return "unknown";
}

}  // namespace fast_lio_truth_recovery
