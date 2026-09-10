/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include <iostream>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include "FSM/State_FixedStand.h"
#include "common/angle_wrap.h"

namespace {
float readStandDuration()
{
    const float defaultDuration = 3.0f;
    const char *envValue = std::getenv("UNITREE_STAND_DURATION");
    if(envValue == nullptr || envValue[0] == '\0'){
        return defaultDuration;
    }

    char *end = nullptr;
    errno = 0;
    float duration = std::strtof(envValue, &end);
    if(errno != 0 || end == envValue || *end != '\0' || !std::isfinite(duration) || duration < 0.5f || duration > 8.0f){
        std::cout << "[WARNING] Invalid UNITREE_STAND_DURATION='" << envValue
                  << "', using default " << defaultDuration << "s." << std::endl;
        return defaultDuration;
    }
    return duration;
}

float readStandSettleDuration()
{
    const float defaultDuration = 0.5f;
    const char *envValue = std::getenv("UNITREE_STAND_SETTLE_DURATION");
    if(envValue == nullptr || envValue[0] == '\0'){
        return defaultDuration;
    }

    char *end = nullptr;
    errno = 0;
    float duration = std::strtof(envValue, &end);
    if(errno != 0 || end == envValue || *end != '\0' || !std::isfinite(duration) || duration < 0.0f || duration > 3.0f){
        std::cout << "[WARNING] Invalid UNITREE_STAND_SETTLE_DURATION='" << envValue
                  << "', using default " << defaultDuration << "s." << std::endl;
        return defaultDuration;
    }
    return duration;
}

// Overrides _targetPos with a 12-value comma-separated joint pose from the
// environment (joint order: FR/FL/RR/RL x hip/thigh/calf, the junior_ctrl
// FSM/URDF order; NOT the name-sorted Gazebo joint_states order, so any
// snapshot taken from /a1_gazebo/joint_states must be re-indexed first).
// Returns true and writes the pose only when the
// whole string parses; otherwise the hardcoded default stance is kept.
bool readStandTargetJoints(float* targetPos)
{
    const char *envValue = std::getenv("UNITREE_STAND_TARGET_JOINTS");
    if(envValue == nullptr || envValue[0] == '\0'){
        return false;
    }

    std::istringstream stream(envValue);
    std::string token;
    float values[12];
    int count = 0;
    while(std::getline(stream, token, ',')){
        if(count >= 12){
            std::cout << "[WARNING] UNITREE_STAND_TARGET_JOINTS has more than "
                      << "12 values, using default stance." << std::endl;
            return false;
        }
        char *end = nullptr;
        errno = 0;
        const float value = std::strtof(token.c_str(), &end);
        if(errno != 0 || end == token.c_str() || *end != '\0' ||
           !std::isfinite(value) || value < -3.2f || value > 3.2f){
            std::cout << "[WARNING] Invalid UNITREE_STAND_TARGET_JOINTS token '"
                      << token << "', using default stance." << std::endl;
            return false;
        }
        values[count++] = value;
    }
    if(count != 12){
        std::cout << "[WARNING] UNITREE_STAND_TARGET_JOINTS has " << count
                  << " values, expected 12, using default stance." << std::endl;
        return false;
    }
    for(int i=0; i<12; ++i){
        targetPos[i] = values[i];
    }
    std::cout << "[INFO] Using UNITREE_STAND_TARGET_JOINTS custom stance." << std::endl;
    return true;
}

float smoothStep(float value)
{
    if(value <= 0.0f){
        return 0.0f;
    }
    if(value >= 1.0f){
        return 1.0f;
    }
    return value * value * (3.0f - 2.0f * value);
}

float lerp(float start, float end, float percent)
{
    return start + (end - start) * percent;
}
}

State_FixedStand::State_FixedStand(CtrlComponents *ctrlComp)
                :FSMState(ctrlComp, FSMStateName::FIXEDSTAND, "fixed stand")
{
    _readyPub = nh.advertise<std_msgs::Bool>("/fixed_stand_ready", 1, true);
    _statusPub = nh.advertise<std_msgs::String>("/fixed_stand_status", 10);
    fixedStandDefaultTargetPos(_targetPos);
}

void State_FixedStand::enter(){
    _duration = readStandDuration();
    _settleDuration = readStandSettleDuration();
    _minimumReadyElapsed = 5.0f;
    _stableReadyDuration = 2.5f;
    _minimumGainRatio = 0.0f;
    // Readiness tilt limits: overridable, clamped to a sane band.  Defaults
    // keep the historical 0.12 rad behaviour for callers that set nothing.
    double readyMaxRoll = 0.12;
    double readyMaxPitch = 0.12;
    nh.param("/simenv/fixed_stand_ready_max_roll_rad",
             readyMaxRoll, readyMaxRoll);
    nh.param("/simenv/fixed_stand_ready_max_pitch_rad",
             readyMaxPitch, readyMaxPitch);
    _readyMaxRollRad = std::max(0.02f, std::min(
        0.45f, static_cast<float>(readyMaxRoll)));
    _readyMaxPitchRad = std::max(0.02f, std::min(
        0.45f, static_cast<float>(readyMaxPitch)));
    // Stair managers may freeze an atomically captured, already-upright,
    // low-velocity gait pose before a policy swap.  In that narrow case the
    // ordinary 3 s interpolation plus 2.5 s readiness dwell changes no joint
    // target and charges about eight stationary seconds to every stair.  A
    // one-shot ROS latch lets the verified caller request a shorter gain ramp
    // and fresh stability proof.  Startup/default-stance entries retain the
    // conservative values above, and every supporting parameter is consumed
    // with the enable latch so it cannot leak into a later stand.
    bool fastReady = false;
    if(nh.getParam("/simenv/fixed_stand_fast_ready_enabled", fastReady)){
        nh.deleteParam("/simenv/fixed_stand_fast_ready_enabled");
    }
    if(fastReady){
        double duration = 1.0;
        double settle = 0.10;
        double minimumElapsed = 1.0;
        double stableDuration = 1.0;
        double minimumGainRatio = 0.0;
        nh.param("/simenv/fixed_stand_fast_duration_seconds",
                 duration, duration);
        nh.param("/simenv/fixed_stand_fast_settle_seconds", settle, settle);
        nh.param("/simenv/fixed_stand_fast_minimum_elapsed_seconds",
                 minimumElapsed, minimumElapsed);
        nh.param("/simenv/fixed_stand_fast_stable_seconds",
                 stableDuration, stableDuration);
        nh.param("/simenv/fixed_stand_fast_minimum_gain_ratio",
                 minimumGainRatio, minimumGainRatio);
        nh.deleteParam("/simenv/fixed_stand_fast_duration_seconds");
        nh.deleteParam("/simenv/fixed_stand_fast_settle_seconds");
        nh.deleteParam("/simenv/fixed_stand_fast_minimum_elapsed_seconds");
        nh.deleteParam("/simenv/fixed_stand_fast_stable_seconds");
        nh.deleteParam("/simenv/fixed_stand_fast_minimum_gain_ratio");
        if(std::isfinite(duration) && std::isfinite(settle) &&
           std::isfinite(minimumElapsed) && std::isfinite(stableDuration) &&
           std::isfinite(minimumGainRatio)){
            _duration = std::max(0.5f, std::min(
                3.0f, static_cast<float>(duration)));
            _settleDuration = std::max(0.0f, std::min(
                0.5f, static_cast<float>(settle)));
            _minimumReadyElapsed = std::max(
                _duration + _settleDuration,
                std::min(5.0f, static_cast<float>(minimumElapsed)));
            _stableReadyDuration = std::max(0.5f, std::min(
                2.5f, static_cast<float>(stableDuration)));
            _minimumGainRatio = std::max(0.0f, std::min(
                1.0f, static_cast<float>(minimumGainRatio)));
            ROS_WARN("Using one-shot captured-pose FixedStand profile: "
                     "ramp %.2f s, settle %.2f s, ready after %.2f s + "
                     "%.2f s stable, minimum gain %.2f",
                     _duration, _settleDuration, _minimumReadyElapsed,
                     _stableReadyDuration, _minimumGainRatio);
        }
    }
    // Unconditional fresh-default reset: _targetPos is mutable state, so a
    // previous climb-time pose (param or env override from an earlier entry)
    // must never leak into a later FixedStand entry after the param is cleared.
    fixedStandDefaultTargetPos(_targetPos);
    // A dynamic param wins over the env override: the F2->F3 stair manager
    // sets /stand_target_joints right before its pre-ascent stand and clears
    // it afterwards, so the startup stand (auto-hold) keeps the hardcoded
    // default stance instead of collapsing into a climb-time pose on flat
    // ground.  Env remains as a static fallback for manual runs.
    std::vector<double> paramTarget;
    if(nh.getParam("stand_target_joints", paramTarget)){
        if(paramTarget.size() == 12){
            for(int i=0; i<12; ++i){
                _targetPos[i] = static_cast<float>(paramTarget[i]);
            }
            std::cout << "[INFO] Using /stand_target_joints param stance." << std::endl;
        }else{
            std::cout << "[WARNING] /stand_target_joints has "
                      << paramTarget.size() << " values, expected 12; "
                      << "using default stance." << std::endl;
        }
    }else{
        readStandTargetJoints(_targetPos);
    }
    // In Gazebo the joint-level controller reports and clamps limited
    // revolute joints in their canonical URDF interval, so sim commands must
    // stay canonical too: wrap the target to (-pi, pi] and never emit a
    // +2*pi equivalent based on the measured q.  The real robot keeps the
    // original target semantics (custom targets such as +3.2 are legal).
    if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
        for(int i=0; i<12; ++i){
            _targetPos[i] = wrapToPi(_targetPos[i]);
        }
    }
    _elapsed = 0.0f;
    _percent = 0.0f;
    _stableElapsed = 0.0f;
    _ready = false;
    std_msgs::Bool readyMessage;
    readyMessage.data = false;
    _readyPub.publish(readyMessage);
    for(int i=0; i<4; i++){
        if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
            setRampSimStanceGain(_minimumGainRatio);
        }
        else if(_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT){
            _lowCmd->setRealStanceGain(i);
        }
        _lowCmd->setZeroDq(i);
        _lowCmd->setZeroTau(i);
    }
    for(int i=0; i<12; i++){
        _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
        _startPos[i] = _lowState->motorState[i].q;
        if(_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT && _ctrlComp->ioInterFreeDog){
            _startPos_real[i] = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[i].q;
        }else{
            _startPos_real[i] = _lowState->motorState[i].q;
        }
    }
    _ctrlComp->setAllStance();
}

void State_FixedStand::run(){
    _elapsed += static_cast<float>(_ctrlComp->dt);
    if(_elapsed < _settleDuration){
        if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
            setRampSimStanceGain(_minimumGainRatio);
        }
        for(int j=0; j<12; j++){
            _lowCmd->motorCmd[j].q = _startPos[j];
        }
        return;
    }

    _percent = (_elapsed - _settleDuration) / _duration;
    _percent = _percent > 1 ? 1 : _percent;
    const float smoothPercent = smoothStep(_percent);
    if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
        const float gainPercent = _minimumGainRatio +
            (1.0f - _minimumGainRatio) * smoothPercent;
        setRampSimStanceGain(gainPercent);
    }
    for(int j=0; j<12; j++){
        _lowCmd->motorCmd[j].q = (1 - smoothPercent)*_startPos[j] + smoothPercent*_targetPos[j];
    }

    const Vec3 rpy = rotMatToRPY(_lowState->getRotMat());
    const double baseZ = _ctrlComp->estimator->getPosition()(2);
    double velocitySquare = 0.0;
    double errorSquare = 0.0;
    for(int j=0; j<12; ++j){
        velocitySquare += _lowState->motorState[j].dq * _lowState->motorState[j].dq;
        const double error = _ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO
            ? shortestAngleDiff(_lowState->motorState[j].q, _targetPos[j])
            : static_cast<double>(_lowState->motorState[j].q - _targetPos[j]);
        errorSquare += error * error;
    }
    const double velocityRms = std::sqrt(velocitySquare / 12.0);
    const double positionError = std::sqrt(errorSquare / 12.0);
    const double gyroNorm = _lowState->getGyro().norm();
    const double accelerationNorm = _lowState->getAcc().norm();
    const bool interpolationComplete = _percent >= 0.999f;
    const bool stableNow = interpolationComplete &&
        _elapsed >= _minimumReadyElapsed &&
        std::abs(rpy(0)) < _readyMaxRollRad &&
        std::abs(rpy(1)) < _readyMaxPitchRad &&
        // base_z is kept only as a status diagnostic: the estimator value is
        // not a reliable floor-independent body height in this interface, so
        // it must not gate readiness on upper floors.
        velocityRms < 0.35 &&
        positionError < 0.12 && gyroNorm < 0.35 &&
        accelerationNorm > 7.0 && accelerationNorm < 12.5;
    _stableElapsed = stableNow ? _stableElapsed + _ctrlComp->dt : 0.0f;
    _ready = _stableElapsed >= _stableReadyDuration;
    std_msgs::Bool readyMessage;
    readyMessage.data = _ready;
    _readyPub.publish(readyMessage);
    static int publishDivider = 0;
    if(++publishDivider % 25 == 0){
        std::ostringstream payload;
        payload << "{\"timestamp\":" << ros::Time::now().toSec()
                << ",\"roll\":" << rpy(0) << ",\"pitch\":" << rpy(1)
                << ",\"base_z\":" << baseZ
                << ",\"joint_velocity_rms\":" << velocityRms
                << ",\"joint_position_error\":" << positionError
                << ",\"stable_now\":" << (stableNow ? "true" : "false")
                << ",\"stable_duration\":" << _stableElapsed
                << ",\"interpolation_progress\":" << _percent
                << ",\"fixed_stand_complete\":" << (_ready ? "true" : "false") << "}";
        std_msgs::String message;
        message.data = payload.str();
        _statusPub.publish(message);
    }

    if (real == true && _ctrlComp->ioInterFreeDog){
        for(int j=0; j<12; j++){
            std::vector<double> joint{(1 - smoothPercent)*_startPos_real[j] + \
                smoothPercent*_targetPos[j], 0, 0, real_stand_p[j], real_stand_d[j]};
            _ctrlComp->ioInterFreeDog->setCmd(j,joint);
        }
    }
}

void State_FixedStand::setRampSimStanceGain(float percent){
    const float gainPercent = smoothStep(percent);
    for(int legID=0; legID<4; legID++){
        const int hip = legID * 3;
        const int thigh = hip + 1;
        const int calf = hip + 2;

        _lowCmd->motorCmd[hip].mode = 10;
        _lowCmd->motorCmd[hip].Kp = lerp(15.0f, 95.0f, gainPercent);
        _lowCmd->motorCmd[hip].Kd = lerp(1.5f, 5.0f, gainPercent);

        _lowCmd->motorCmd[thigh].mode = 10;
        _lowCmd->motorCmd[thigh].Kp = lerp(15.0f, 95.0f, gainPercent);
        _lowCmd->motorCmd[thigh].Kd = lerp(1.5f, 5.0f, gainPercent);

        _lowCmd->motorCmd[calf].mode = 10;
        _lowCmd->motorCmd[calf].Kp = lerp(25.0f, 140.0f, gainPercent);
        _lowCmd->motorCmd[calf].Kd = lerp(2.0f, 7.0f, gainPercent);
    }
}

void State_FixedStand::exit(){
    _percent = 0;
}

FSMStateName State_FixedStand::checkChange(){
    if(_lowState->userCmd == UserCommand::L2_B){
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::L2_X){
        return FSMStateName::FREESTAND;
    }
    else if(_lowState->userCmd == UserCommand::L1_X){
        return FSMStateName::BALANCETEST;
    }
    else if(_lowState->userCmd == UserCommand::L1_A){
        return FSMStateName::SWINGTEST;
    }
    else if(_lowState->userCmd == UserCommand::L1_Y){
        return FSMStateName::STEPTEST;
    }
    else if(_lowState->userCmd == UserCommand::START){
        return FSMStateName::TROTTING;
    }
#ifdef COMPILE_WITH_MOVE_BASE
    else if(_lowState->userCmd == UserCommand::L2_Y){
        return FSMStateName::MOVE_BASE;
    }
#endif  // COMPILE_WITH_MOVE_BASE
    else if(_lowState->userCmd == UserCommand::RL ||
            _lowState->userCmd == UserCommand::RL_KEYBOARD){
        return FSMStateName::RL;
    }
    else{
        return FSMStateName::FIXEDSTAND;
    }
}
