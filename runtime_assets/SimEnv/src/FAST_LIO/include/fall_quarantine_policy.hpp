#pragma once

#include <algorithm>
#include <cstdint>

namespace fast_lio_fall_quarantine {

enum class Phase {
    kOpen,
    kActive,
    kWarmup,
};

struct SignalTransition {
    bool entered_active = false;
    bool entered_warmup = false;
};

struct FrameDecision {
    bool allow_map_and_cloud = false;
    bool clean_scan = false;
    bool just_released = false;
    bool invalid_reset = false;
    bool correction_reset = false;
};

class Policy {
public:
    explicit Policy(int required_clean_scans = 10)
        : required_clean_scans_(std::max(1, required_clean_scans))
    {
    }

    void set_required_clean_scans(int value)
    {
        required_clean_scans_ = std::max(1, value);
        if (consecutive_clean_scans_ > required_clean_scans_)
            consecutive_clean_scans_ = required_clean_scans_;
    }

    SignalTransition observe_fall_signal(bool active)
    {
        SignalTransition transition;
        if (active) {
            if (phase_ != Phase::kActive) {
                phase_ = Phase::kActive;
                consecutive_clean_scans_ = 0;
                ++active_entries_;
                transition.entered_active = true;
            }
            return transition;
        }
        if (phase_ == Phase::kActive) {
            phase_ = Phase::kWarmup;
            consecutive_clean_scans_ = 0;
            ++warmup_entries_;
            transition.entered_warmup = true;
        }
        return transition;
    }

    void note_active_packet()
    {
        if (phase_ == Phase::kActive) ++active_packets_;
    }

    FrameDecision observe_registration(bool registration_valid,
                                       bool truth_correction_frame)
    {
        FrameDecision decision;
        if (phase_ == Phase::kActive) return decision;

        decision.clean_scan =
            registration_valid && !truth_correction_frame;
        if (phase_ == Phase::kOpen) {
            decision.allow_map_and_cloud = decision.clean_scan;
            return decision;
        }

        if (truth_correction_frame) {
            consecutive_clean_scans_ = 0;
            ++correction_resets_;
            decision.correction_reset = true;
            return decision;
        }
        if (!registration_valid) {
            consecutive_clean_scans_ = 0;
            ++invalid_resets_;
            decision.invalid_reset = true;
            return decision;
        }

        ++consecutive_clean_scans_;
        if (consecutive_clean_scans_ >= required_clean_scans_) {
            consecutive_clean_scans_ = required_clean_scans_;
            phase_ = Phase::kOpen;
            ++release_count_;
            decision.allow_map_and_cloud = true;
            decision.just_released = true;
        }
        return decision;
    }

    bool may_initialize_empty_map(bool command_quiet) const
    {
        if (phase_ == Phase::kActive) return false;
        if (phase_ == Phase::kWarmup) return command_quiet;
        return true;
    }

    Phase phase() const { return phase_; }
    bool active() const { return phase_ == Phase::kActive; }
    bool warming_up() const { return phase_ == Phase::kWarmup; }
    int required_clean_scans() const { return required_clean_scans_; }
    int consecutive_clean_scans() const { return consecutive_clean_scans_; }
    std::uint64_t active_entries() const { return active_entries_; }
    std::uint64_t warmup_entries() const { return warmup_entries_; }
    std::uint64_t active_packets() const { return active_packets_; }
    std::uint64_t invalid_resets() const { return invalid_resets_; }
    std::uint64_t correction_resets() const { return correction_resets_; }
    std::uint64_t release_count() const { return release_count_; }

    static const char *phase_name(Phase phase)
    {
        switch (phase) {
        case Phase::kOpen: return "open";
        case Phase::kActive: return "active";
        case Phase::kWarmup: return "warmup";
        }
        return "unknown";
    }

private:
    Phase phase_ = Phase::kOpen;
    int required_clean_scans_ = 10;
    int consecutive_clean_scans_ = 0;
    std::uint64_t active_entries_ = 0;
    std::uint64_t warmup_entries_ = 0;
    std::uint64_t active_packets_ = 0;
    std::uint64_t invalid_resets_ = 0;
    std::uint64_t correction_resets_ = 0;
    std::uint64_t release_count_ = 0;
};

}  // namespace fast_lio_fall_quarantine
