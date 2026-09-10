#include <cassert>
#include <iostream>

#include "fall_quarantine_policy.hpp"

using fast_lio_fall_quarantine::Phase;
using fast_lio_fall_quarantine::Policy;

int main()
{
    Policy policy(3);
    assert(policy.phase() == Phase::kOpen);
    assert(policy.observe_registration(true, false).allow_map_and_cloud);

    const auto rising = policy.observe_fall_signal(true);
    assert(rising.entered_active);
    assert(policy.active());
    assert(!policy.observe_registration(true, false).allow_map_and_cloud);
    assert(!policy.may_initialize_empty_map(true));
    policy.note_active_packet();
    assert(policy.active_packets() == 1);

    const auto falling = policy.observe_fall_signal(false);
    assert(falling.entered_warmup);
    assert(policy.warming_up());
    assert(!policy.may_initialize_empty_map(false));
    assert(policy.may_initialize_empty_map(true));

    auto frame = policy.observe_registration(true, false);
    assert(frame.clean_scan);
    assert(!frame.allow_map_and_cloud);
    assert(policy.consecutive_clean_scans() == 1);

    frame = policy.observe_registration(false, false);
    assert(frame.invalid_reset);
    assert(!frame.allow_map_and_cloud);
    assert(policy.consecutive_clean_scans() == 0);
    assert(policy.invalid_resets() == 1);

    assert(!policy.observe_registration(true, false).allow_map_and_cloud);
    frame = policy.observe_registration(true, true);
    assert(frame.correction_reset);
    assert(!frame.allow_map_and_cloud);
    assert(policy.consecutive_clean_scans() == 0);
    assert(policy.correction_resets() == 1);

    assert(!policy.observe_registration(true, false).allow_map_and_cloud);
    assert(!policy.observe_registration(true, false).allow_map_and_cloud);
    frame = policy.observe_registration(true, false);
    assert(frame.just_released);
    assert(frame.allow_map_and_cloud);
    assert(policy.phase() == Phase::kOpen);
    assert(policy.release_count() == 1);

    // Registration validity remains a permanent cloud/map gate even after
    // quarantine release.
    assert(!policy.observe_registration(false, false).allow_map_and_cloud);
    assert(!policy.observe_registration(true, true).allow_map_and_cloud);
    assert(policy.observe_registration(true, false).allow_map_and_cloud);

    std::cout << "fall_quarantine_policy tests passed\n";
    return 0;
}
