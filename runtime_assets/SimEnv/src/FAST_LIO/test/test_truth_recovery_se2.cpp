#include <cassert>
#include <cmath>
#include <iostream>

#include "truth_recovery_se2.hpp"

using fast_lio_truth_recovery::DivergenceGate;
using fast_lio_truth_recovery::Pose2d;
using fast_lio_truth_recovery::RecoveryDecision;
using fast_lio_truth_recovery::RecoveryGate;

namespace {

bool near(double left, double right, double tolerance = 1.0e-9)
{
    return std::abs(left - right) <= tolerance;
}

RecoveryGate valid_gate()
{
    RecoveryGate gate;
    gate.enabled = true;
    gate.context_active = true;
    gate.request_pending = true;
    gate.request_fresh = true;
    gate.anchor_valid = true;
    gate.truth_fresh = true;
    gate.command_quiet = true;
    gate.truth_quiet = true;
    gate.registration_fault = true;
    gate.request_reports_registration_fault = true;
    gate.planar_residual_m = 0.8;
    gate.yaw_residual_rad = 0.0;
    gate.minimum_planar_residual_m = 0.2;
    gate.minimum_yaw_residual_rad = 0.15;
    gate.maximum_planar_correction_m = 30.0;
    gate.maximum_yaw_correction_rad = 3.141592653589793;
    return gate;
}

}  // namespace

int main()
{
    const Pose2d timed_pose{1.0, 2.0, 0.20};
    const Pose2d forward_extrapolated =
        fast_lio_truth_recovery::extrapolate_world_pose(
            timed_pose, 2.0, -1.0, 0.5, 0.10, 0.15);
    assert(near(forward_extrapolated.x, 1.20));
    assert(near(forward_extrapolated.y, 1.90));
    assert(near(forward_extrapolated.yaw, 0.25));
    const Pose2d backward_extrapolated =
        fast_lio_truth_recovery::extrapolate_world_pose(
            timed_pose, 2.0, -1.0, 0.5, -0.10, 0.15);
    assert(near(backward_extrapolated.x, 0.80));
    assert(near(backward_extrapolated.y, 2.10));
    assert(near(backward_extrapolated.yaw, 0.15));
    const Pose2d bounded_extrapolated =
        fast_lio_truth_recovery::extrapolate_world_pose(
            timed_pose, 2.0, -1.0, 0.5, 0.50, 0.15);
    assert(near(bounded_extrapolated.x, 1.30));
    assert(near(bounded_extrapolated.y, 1.85));
    assert(near(bounded_extrapolated.yaw, 0.275));

    // Truth and odometry may differ by any fixed SE(2) transform.  Only the
    // relative displacement after the anchors must be transferred.
    const Pose2d odom_anchor{40.0, -25.0, 1.9207963267948966};
    const Pose2d truth_anchor{4.0, -3.0, 0.35};
    const Pose2d truth_pose{7.5, 1.25, 1.10};
    const Pose2d target =
        fast_lio_truth_recovery::anchored_truth_target(
            odom_anchor, truth_anchor, truth_pose);
    assert(near(target.x, 35.75));
    assert(near(target.y, -21.5));
    assert(near(target.yaw, 2.6707963267948966));

    const auto zero_error = fast_lio_truth_recovery::residual(target, target);
    assert(near(zero_error.planar_m, 0.0));
    assert(near(zero_error.yaw_rad, 0.0));

    DivergenceGate divergence;
    divergence.enabled = true;
    divergence.context_active = true;
    divergence.anchor_valid = true;
    divergence.truth_fresh = true;
    divergence.quarantine_open = true;
    divergence.maximum_planar_residual_m = 1.25;
    divergence.maximum_yaw_residual_rad = 0.55;
    divergence.planar_residual_m = 1.249;
    divergence.yaw_residual_rad = 0.549;
    assert(!fast_lio_truth_recovery::reject_divergent_registration(
        divergence));
    divergence.planar_residual_m = 1.25;
    assert(fast_lio_truth_recovery::reject_divergent_registration(
        divergence));
    divergence.planar_residual_m = 0.20;
    divergence.yaw_residual_rad = 0.55;
    assert(fast_lio_truth_recovery::reject_divergent_registration(
        divergence));
    divergence.truth_fresh = false;
    assert(!fast_lio_truth_recovery::reject_divergent_registration(
        divergence));
    divergence.truth_fresh = true;
    divergence.quarantine_open = false;
    assert(!fast_lio_truth_recovery::reject_divergent_registration(
        divergence));

    assert(fast_lio_truth_recovery::registration_supports_degeneracy_assist(
        true, false, false));
    assert(!fast_lio_truth_recovery::registration_supports_degeneracy_assist(
        false, false, false));
    assert(!fast_lio_truth_recovery::registration_supports_degeneracy_assist(
        true, true, false));
    assert(!fast_lio_truth_recovery::registration_supports_degeneracy_assist(
        true, false, true));

    RecoveryGate gate = valid_gate();
    assert(fast_lio_truth_recovery::decide(gate) ==
           RecoveryDecision::kApply);

    gate = valid_gate();
    gate.command_quiet = false;
    assert(fast_lio_truth_recovery::decide(gate) ==
           RecoveryDecision::kBodyMoving);

    gate = valid_gate();
    gate.registration_fault = false;
    assert(fast_lio_truth_recovery::decide(gate) ==
           RecoveryDecision::kNoFaultEvidence);

    gate = valid_gate();
    gate.registration_fault = false;
    gate.request_reports_registration_fault = false;
    gate.request_reports_anchored_se2_fault = true;
    assert(fast_lio_truth_recovery::decide(gate) ==
           RecoveryDecision::kApply);

    gate = valid_gate();
    gate.planar_residual_m = 0.05;
    gate.yaw_residual_rad = 0.05;
    assert(fast_lio_truth_recovery::decide(gate) ==
           RecoveryDecision::kResidualBelowThreshold);

    gate = valid_gate();
    gate.planar_residual_m = 31.0;
    assert(fast_lio_truth_recovery::decide(gate) ==
           RecoveryDecision::kCorrectionTooLarge);

    std::cout << "truth_recovery_se2 tests passed\n";
    return 0;
}
