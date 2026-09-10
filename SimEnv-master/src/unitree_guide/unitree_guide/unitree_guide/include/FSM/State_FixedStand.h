/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#ifndef FIXEDSTAND_H
#define FIXEDSTAND_H

#include "FSM/FSMState.h"
#include "common/fixed_stand_defaults.h"
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

class State_FixedStand : public FSMState{
public:
    State_FixedStand(CtrlComponents *ctrlComp);
    ~State_FixedStand(){}
    void enter();
    void run();
    void exit();
    FSMStateName checkChange();

private:
    // Preserve the stance used by the last known-good Room0 RL run.  The RL
    // policy owns the transition to its nominal pose during its zero-command
    // warm-up; forcing the nominal pose here changed the learned initial state.
    // The default is defined once in common/fixed_stand_defaults.h and is
    // re-applied unconditionally at every enter() so no prior climb-time pose
    // can leak into later FixedStand entries after /stand_target_joints is cleared.
    float _targetPos[12];
    float _startPos[12];
    float _startPos_real[12];
    float real_stand_p[12] = {80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80};
    float real_stand_d[12]= {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float _duration = 3.0;   // seconds
    float _settleDuration = 0.5;
    float _minimumReadyElapsed = 5.0;
    float _stableReadyDuration = 2.5;
    // A captured-pose policy boundary has effectively zero position error.
    // Keep enough joint impedance from its first control sample to prevent a
    // single-support gait phase from tipping before the ordinary ramp rises.
    float _minimumGainRatio = 0.0;
    float _elapsed = 0;
    float _percent = 0;
    float _stableElapsed = 0;
    // Readiness tilt gate.  A stair-top captured-pose stand can freeze on
    // uneven support (one foot on the top nosing) and hold a perfectly
    // stable body lean of 0.13--0.17 rad that the legacy hard-coded 0.12
    // gate could never certify, dead-locking the policy switch.  The limits
    // stay overridable from the param server.
    float _readyMaxRollRad = 0.12f;
    float _readyMaxPitchRad = 0.12f;
    bool _ready = false;
    ros::Publisher _readyPub;
    ros::Publisher _statusPub;
    void setRampSimStanceGain(float percent);
};

#endif  // FIXEDSTAND_H
