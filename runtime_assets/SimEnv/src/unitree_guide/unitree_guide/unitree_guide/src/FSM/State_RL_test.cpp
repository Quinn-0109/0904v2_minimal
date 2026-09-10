/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include "FSM/State_RL_test.h"
#include "common/angle_wrap.h"
#include "common/policy_request.h"

namespace {
float finiteAxis(float value)
{
    if(!std::isfinite(value)){
        return 0.0f;
    }
    if(value > 1.0f){
        return 1.0f;
    }
    if(value < -1.0f){
        return -1.0f;
    }
    return value;
}
}

State_RL::State_RL(CtrlComponents *ctrlComp)
                :FSMState(ctrlComp, FSMStateName::RL, "RL")
{
    load_policy();
    // Load both mission gaits before any exploration clock can start.  Later
    // policy requests only exchange the active module pointer; no disk I/O,
    // deserialization, or CUDA initialization occurs at a stair boundary.
    preloadConfiguredPolicies();
    _activePolicyPath = model_path;
    gravity(0,0) = 0.0;
    gravity(1,0) = 0.0;
    gravity(2,0) = -0.98;
    //在构造函数中初始化，订阅
    this->Sub_=nh.subscribe<geometry_msgs::Twist>("/cmd_vel",1000,boost::bind(&FSMState::cmdVelCallback,this,_1));
    _readyPub = nh.advertise<std_msgs::Bool>("/locomotion_ready", 1, true);
    _statusPub = nh.advertise<std_msgs::String>("/rl_takeover_status", 20);
    _policyRequestSub = nh.subscribe<std_msgs::String>(
        "/simenv/rl_policy_request", 1,
        boost::bind(&State_RL::policyRequestCallback, this, _1));
    double tiltWarning = _planeTiltGuardWarningRad;
    double tiltStop = _planeTiltGuardStopRad;
    double tiltRelease = _planeTiltGuardReleaseRad;
    nh.param("/simenv/plane_tilt_guard_warning_rad", tiltWarning, tiltWarning);
    nh.param("/simenv/plane_tilt_guard_stop_rad", tiltStop, tiltStop);
    nh.param("/simenv/plane_tilt_guard_release_rad", tiltRelease, tiltRelease);
    if(std::isfinite(tiltWarning) && std::isfinite(tiltStop) &&
       std::isfinite(tiltRelease) && tiltRelease > 0.0 &&
       tiltRelease < tiltWarning && tiltWarning < tiltStop && tiltStop < 0.5){
        _planeTiltGuardWarningRad = static_cast<float>(tiltWarning);
        _planeTiltGuardStopRad = static_cast<float>(tiltStop);
        _planeTiltGuardReleaseRad = static_cast<float>(tiltRelease);
    }else{
        ROS_WARN("Invalid plane tilt guard thresholds; using %.2f/%.2f/%.2f rad",
                 _planeTiltGuardReleaseRad, _planeTiltGuardWarningRad,
                 _planeTiltGuardStopRad);
    }
    std::cout << "[INFO] RL observation source: trunk IMU and joint states (no ground truth)."
              << std::endl;

}


void State_RL::enter(){
    const bool keyboardMode = (_lowState->userCmd == UserCommand::RL_KEYBOARD);
    _keyboardMode.store(keyboardMode);
    if(keyboardMode){
        _ctrlComp->ioInter->zeroCmdPanel();
        _lowState->userValue.setZero();
        std::cout << "[INFO] Entered RL keyboard mode. Use W/S, A/D, J/L, Space." << std::endl;
    }else{
        std::cout << "[INFO] Entered RL /cmd_vel mode." << std::endl;
    }

    _locomotionReady.store(false);
    _takeoverFailed.store(false);
    _takeoverStartUs = getTime();
    _stableSinceUs = 0;
    _planeTiltGuardHold = false;
    _planeTiltGuardScale = 1.0f;
    _activeBlendDurationSec = _blendDurationSec;
    _activeZeroHoldSec = _zeroHoldSec;
    // FixedStand is safe at the accepted flight-B pose, but the ordinary
    // 2.5 s blend + 3.0 s zero-command hold leaves the dog unsupported on a
    // narrow turning landing for too long.  Round41's isolated replay slid
    // 0.52 m off that landing before readiness could latch.  The stair
    // manager arms this global flag only immediately before its post-stand
    // RL request.  Consume and delete it once so initial startup, later
    // plane-policy reloads, and unrelated RL entries retain the conservative
    // defaults.
    bool fastTakeover = false;
    if(nh.getParam("/simenv/stair_fast_takeover_enabled", fastTakeover)){
        nh.deleteParam("/simenv/stair_fast_takeover_enabled");
    }
    if(fastTakeover){
        double requestedBlend = 0.75;
        double requestedHold = 0.25;
        nh.param("/simenv/stair_fast_takeover_blend_seconds",
                 requestedBlend, requestedBlend);
        nh.param("/simenv/stair_fast_takeover_zero_hold_seconds",
                 requestedHold, requestedHold);
        if(std::isfinite(requestedBlend) && std::isfinite(requestedHold)){
            _activeBlendDurationSec = std::max(
                0.25f, std::min(_blendDurationSec,
                                static_cast<float>(requestedBlend)));
            _activeZeroHoldSec = std::max(
                0.10f, std::min(_zeroHoldSec,
                                static_cast<float>(requestedHold)));
            ROS_WARN("Using one-shot stair RL takeover profile: blend %.2f s, zero hold %.2f s",
                     _activeBlendDurationSec, _activeZeroHoldSec);
        }
    }
    actions_tensor.zero_();
    actions_tensor_scaled.zero_();
    obs_history_tensor.zero_();
    std_msgs::Bool readyMessage;
    readyMessage.data = false;
    _readyPub.publish(readyMessage);
    // A request can be queued while FixedStand owns the joints.  Consume it
    // synchronously before starting inference: during the TorchScript load,
    // Gazebo then retains the last FixedStand motor targets instead of an
    // arbitrary swing-leg RL sample.  Never clear the pending slot here;
    // doing so made FixedStand policy handoffs impossible.
    if(!loadPendingPolicyOnEntry()){
        _takeoverFailed.store(true);
        _lowState->userCmd = UserCommand::L2_B;
        return;
    }
     // if (real == false){
        for(int i=0; i<12; i++){
            _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
            _startPos[i] = _lowState->motorState[i].q;
            _lowCmd->motorCmd[i].mode = 10;
            _lowCmd->motorCmd[i].dq = 0;
            _lowCmd->motorCmd[i].Kp = 80;
            _lowCmd->motorCmd[i].Kd = 1;
            _lowCmd->motorCmd[i].tau = 0;
        }
        for(int i=0; i<4; i++){
             if(_ctrlComp->ctrlPlatform == CtrlPlatform::GAZEBO){
                 _lowCmd->setSimStanceGain(i);
             }
             else if(_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT){
                 _lowCmd->setRealStanceGain(i);
             }
             _lowCmd->setZeroDq(i);
             _lowCmd->setZeroTau(i);
        }
    // }
    // else if(real == true)
    // {
        if(_ctrlComp->ctrlPlatform == CtrlPlatform::REALROBOT && _ctrlComp->ioInterFreeDog){
            for(int i=0; i<12; i++){
                float c_joint = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[i].q;
                std::vector<double> joint{c_joint, 0, 0, 80, 1};
                _ctrlComp->ioInterFreeDog->setCmd(i,joint);
            }
        }
    // }
    for (int i = 0; i < HISTORY_LEN; i++)
    {
        refresh_rl_obs();
    }
    infer_thread_runnning = State_RL::RUNNING;
    infer_thread = new std::thread(&State_RL::infer_thread_callback,this);
    if (debug == true){
        ampthreadRunning = State_RL::RUNNING;
        amp_obs_thread = new std::thread(&State_RL::save_amp_obs_thread,this);
    }
}

bool State_RL::loadPendingPolicyOnEntry()
{
    std::string requested;
    {
        std::lock_guard<std::mutex> guard(_policyMutex);
        _policyReloadRequested.store(false);
        requested = claimPendingPolicyRequest(_pendingPolicyPath,
                                               _loadingPolicyPath);
    }
    if(requested.empty()){
        return true;
    }
    try{
        bool cacheHit = false;
        std::shared_ptr<torch::jit::script::Module> nextModel =
            cachedPolicy(requested, cacheHit);
        device = inferenceDeviceForPolicy(requested);
        model = nextModel;
        model_path = requested;
        bool queued = false;
        {
            std::lock_guard<std::mutex> guard(_policyMutex);
            queued = finishPolicyLoad(requested, true,
                                      _activePolicyPath,
                                      _loadingPolicyPath,
                                      _pendingPolicyPath);
        }
        if(queued){
            _policyReloadRequested.store(true);
        }
        std_msgs::String status;
        if(cacheHit){
            status.data = std::string("policy_cache_hit:") + requested;
            _statusPub.publish(status);
        }
        status.data = std::string("policy_reloaded:") + requested;
        _statusPub.publish(status);
        std::cout << "[INFO] RL policy "
                  << (cacheHit ? "activated from preload cache" : "loaded")
                  << " at FixedStand entry boundary: " << requested
                  << std::endl;
        return true;
    }catch(const c10::Error& error){
        bool queued = false;
        {
            std::lock_guard<std::mutex> guard(_policyMutex);
            queued = finishPolicyLoad(requested, false,
                                      _activePolicyPath,
                                      _loadingPolicyPath,
                                      _pendingPolicyPath);
        }
        if(queued){
            _policyReloadRequested.store(true);
        }
        std_msgs::String status;
        status.data = std::string("policy_reload_failed:") + error.what();
        _statusPub.publish(status);
        std::cerr << "[ERROR] RL entry policy load failed: "
                  << error.what() << std::endl;
        return false;
    }
}

void State_RL::run(){
    // A policy can be swapped only at this FSM boundary.  Stop and join the
    // inference thread before replacing its TorchScript module, then rebuild
    // the observation history under zero command.  This avoids concurrent
    // model access and permits plane->stair handoff without a second joint
    // controller process.
    if(!_policyReloadRequested.exchange(false)){
        return;
    }
    std::string requested;
    {
        // Atomically claim the queued request.  The claim itself is the only
        // valid transition out of the pending slot; re-validating it against
        // the acceptance predicate here would always reject it (it equals the
        // just-claimed loading path).
        std::lock_guard<std::mutex> guard(_policyMutex);
        requested = claimPendingPolicyRequest(_pendingPolicyPath,
                                               _loadingPolicyPath);
    }
    if(requested.empty()){
        return;
    }
    // A genuinely new policy is starting to load: drop any stale readiness
    // immediately and restart the full takeover handshake (blend + zero
    // command stable hold) before readiness can be published again.
    _locomotionReady.store(false);
    std_msgs::Bool readyMessage;
    readyMessage.data = false;
    _readyPub.publish(readyMessage);
    _takeoverFailed.store(false);
    _takeoverStartUs = getTime();
    _stableSinceUs = 0;
    // Hot stair-policy changes retain the ordinary conservative takeover.
    // A plane policy is requested only after the command owner has stopped
    // the robot on a verified level apron/landing.  That caller may arm a
    // separate one-shot plane profile to avoid charging four identical
    // 5.5-second stationary handshakes to the exploration clock.  Consume
    // the enable latch here, after the requested policy path is known.
    _activeBlendDurationSec = _blendDurationSec;
    _activeZeroHoldSec = _zeroHoldSec;
    bool fastPlaneTakeover = false;
    if(requested.find("plane") != std::string::npos &&
       nh.getParam("/simenv/plane_fast_takeover_enabled", fastPlaneTakeover)){
        nh.deleteParam("/simenv/plane_fast_takeover_enabled");
    }
    if(fastPlaneTakeover){
        double requestedBlend = 1.25;
        double requestedHold = 0.75;
        nh.param("/simenv/plane_fast_takeover_blend_seconds",
                 requestedBlend, requestedBlend);
        nh.param("/simenv/plane_fast_takeover_zero_hold_seconds",
                 requestedHold, requestedHold);
        if(std::isfinite(requestedBlend) && std::isfinite(requestedHold)){
            _activeBlendDurationSec = std::max(
                0.25f, std::min(_blendDurationSec,
                                static_cast<float>(requestedBlend)));
            _activeZeroHoldSec = std::max(
                0.10f, std::min(_zeroHoldSec,
                                static_cast<float>(requestedHold)));
            ROS_WARN("Using one-shot plane RL takeover profile: blend %.2f s, zero hold %.2f s",
                     _activeBlendDurationSec, _activeZeroHoldSec);
        }
    }
    for(int i=0; i<12; ++i){
        _startPos[i] = _lowState->motorState[i].q;
    }
    command = {0.0f, 0.0f, 0.0f};
    // Stop and join the inference thread before touching any tensor it
    // reads (actions, scaled actions, observation history).
    infer_thread_runnning = State_RL::STOP;
    if(infer_thread != nullptr){
        if(infer_thread->joinable()) infer_thread->join();
        delete infer_thread;
        infer_thread = nullptr;
    }
    actions_tensor.zero_();
    actions_tensor_scaled.zero_();
    obs_history_tensor.zero_();
    try{
        bool cacheHit = false;
        std::shared_ptr<torch::jit::script::Module> nextModel =
            cachedPolicy(requested, cacheHit);
        device = inferenceDeviceForPolicy(requested);
        model = nextModel;
        model_path = requested;
        for(int i = 0; i < HISTORY_LEN; ++i) refresh_rl_obs();
        {
            std::lock_guard<std::mutex> guard(_policyMutex);
            const bool queued = finishPolicyLoad(requested, true,
                                                 _activePolicyPath,
                                                 _loadingPolicyPath,
                                                 _pendingPolicyPath);
            if(queued){
                _policyReloadRequested.store(true);
            }
        }
        infer_thread_runnning = State_RL::RUNNING;
        infer_thread = new std::thread(&State_RL::infer_thread_callback, this);
        std_msgs::String status;
        if(cacheHit){
            status.data = std::string("policy_cache_hit:") + requested;
            _statusPub.publish(status);
        }
        status.data = std::string("policy_reloaded:") + requested;
        _statusPub.publish(status);
        std::cout << "[INFO] RL policy "
                  << (cacheHit ? "hot-switched from preload cache" : "reloaded")
                  << ": " << requested << std::endl;
    }catch(const c10::Error& error){
        {
            std::lock_guard<std::mutex> guard(_policyMutex);
            const bool queued = finishPolicyLoad(requested, false,
                                                 _activePolicyPath,
                                                 _loadingPolicyPath,
                                                 _pendingPolicyPath);
            if(queued){
                _policyReloadRequested.store(true);
            }
        }
        std_msgs::String status;
        status.data = std::string("policy_reload_failed:") + error.what();
        _statusPub.publish(status);
        std::cerr << "[ERROR] RL policy reload failed: " << error.what() << std::endl;
        // Keep locomotion_ready false and fall back to Passive instead of
        // continuing on the previous policy with a stale ready handshake.
        _lowState->userCmd = UserCommand::L2_B;
    }
}

void State_RL::policyRequestCallback(const std_msgs::String::ConstPtr& message){
    if(message == nullptr || message->data.empty()) return;
    std::lock_guard<std::mutex> guard(_policyMutex);
    if(!shouldAcceptPolicyRequest(message->data, _activePolicyPath,
                                  _pendingPolicyPath, _loadingPolicyPath)){
        // Idempotent: the second_floor manager republishes the same path every
        // 0.75 s; only a different path triggers a reload/handshake.
        return;
    }
    _pendingPolicyPath = message->data;
    _policyReloadRequested.store(true);
}

void State_RL::exit(){
    // locomotion_ready is a state-entry handshake, not a continuously evaluated
    // walking-stability signal.  Clear it only when leaving RL so downstream
    // command consumers stop immediately during a mode transition.
    _locomotionReady.store(false);
    std_msgs::Bool readyMessage;
    readyMessage.data = false;
    _readyPub.publish(readyMessage);
    _percent = 0;
    ampthreadRunning = State_RL::STOP;
    infer_thread_runnning = State_RL::STOP;
    if(amp_obs_thread != nullptr){
        if(amp_obs_thread->joinable()){
            amp_obs_thread->join();
        }
        delete amp_obs_thread;
        amp_obs_thread = nullptr;
        std::cout << "amp_obs_thread退出!" << std::endl;
    }
    if(infer_thread != nullptr){
        if(infer_thread->joinable()){
            infer_thread->join();
        }
        delete infer_thread;
        infer_thread = nullptr;
        std::cout << "infer_thread退出!" << std::endl;
    }
    if (outfile.is_open()) {
        outfile.close();
        std::cout << "文件关闭成功!" << std::endl;
    }
}

FSMStateName State_RL::checkChange(){
    if(_lowState->userCmd == UserCommand::L2_B){
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::L2_A){
        return FSMStateName::FIXEDSTAND;
    }
    else if(_lowState->userCmd == UserCommand::RL_KEYBOARD){
        if(!_keyboardMode.exchange(true)){
            _ctrlComp->ioInter->zeroCmdPanel();
            _lowState->userValue.setZero();
            std::cout << "[INFO] Switched RL command source to keyboard axes." << std::endl;
        }
        _last_cmd = static_cast<int>(_lowState->userCmd);
        return FSMStateName::RL;
    }
    else if(_lowState->userCmd == UserCommand::RL){
        if(_keyboardMode.exchange(false)){
            std::cout << "[INFO] Switched RL command source to /cmd_vel." << std::endl;
        }
        _last_cmd = static_cast<int>(_lowState->userCmd);
        return FSMStateName::RL;
    }
    else if(_lowState->userCmd == UserCommand::L1_X){
        if (_last_cmd==static_cast<int>(UserCommand::RL) ||
            _last_cmd==static_cast<int>(UserCommand::RL_KEYBOARD))
        {
            _cnt = (_cnt+1)%(sizeof(_targetPos_map) / sizeof(_targetPos_map[0]));
            if (real == false){
                for(int i=0; i<12; i++){
                    _lowCmd->motorCmd[i].q = _lowState->motorState[i].q;
                    _startPos[i] = _lowState->motorState[i].q;
                }
            }
            else if(real == true && _ctrlComp->ioInterFreeDog){
                for(int i=0; i<12; i++){
                    _startPos[i] = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[i].q;
                }
            }
            _percent = 0;
            std::cout << "cnt: " << _cnt << std::endl;
            // open_amp_save_file();
            dofPosSwitBeginTime = getTime();
        }
        _last_cmd = static_cast<int>(_lowState->userCmd);
        return FSMStateName::RL;
    }
    else{
        _last_cmd = static_cast<int>(_lowState->userCmd);
        return FSMStateName::RL;
    }
}

void State_RL::infer_thread_callback()
{
    while(infer_thread_runnning == State_RL::RUNNING)
    {
        long long _start_time = getTime();
        // std::cout << "_start_time" << _start_time << std::endl;
        refresh_rl_obs();
        torch::Tensor flattened_obs = obs_history_tensor.view({1, HISTORY_LEN * 45});
        if (debug == true)
        {
            const std::vector<int> sub_sizes = {3, 3, 3, 12, 12, 12};
            int segment_size = 45;
            // std::cout << "printSegments" << std::endl;
            // printSegments(flattened_obs.squeeze(), segment_size, sub_sizes);
        }
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(flattened_obs);
        // std::cout << "flattened_obs: " << flattened_obs << std::endl;
        actions_tensor = model->get_method("act_inference")(inputs).toTensor().to(torch::kCPU).squeeze();
        if (debug==true){
            torch::Tensor input_tensor = torch::arange(1, 226).view({1, 225}).to(torch::kFloat32).to(device); // 注意范围是 [start, end)
            std::vector<torch::jit::IValue> test;
            test.push_back(input_tensor);
            torch::Tensor output = model->get_method("act_inference")(test).toTensor().to(torch::kCPU).squeeze();
            // printTensorHorizontal(output, "same_net_work_test");
        }
        actions_tensor_scaled = actions_tensor.clone() * 0.25;
        const float actionRms = actions_tensor_scaled.pow(2).mean().sqrt().item<float>();
        ROS_INFO_THROTTLE(
            2.0,
            "RL policy input vx=%.3f vy=%.3f wz=%.3f action_rms=%.3f "
            "ready=%d userCmd=%d kbd=%d raw=%.3f,%.3f,%.3f grav=%.2f,%.2f,%.2f",
            commands_tensor[0].item<float>(), commands_tensor[1].item<float>(),
            commands_tensor[2].item<float>(), actionRms,
            _locomotionReady.load() ? 1 : 0,
            static_cast<int>(_lowState->userCmd),
            _keyboardMode.load() ? 1 : 0,
            current_cmd_vel_.linear_x, current_cmd_vel_.linear_y,
            current_cmd_vel_.angular_z,
            projected_gravity_tensor[0].item<float>(),
            projected_gravity_tensor[1].item<float>(),
            projected_gravity_tensor[2].item<float>()
        );
        std::vector<float> actions(actions_tensor_scaled.data_ptr<float>(),
                           actions_tensor_scaled.data_ptr<float>() + actions_tensor_scaled.numel());
        if (debug == true) std::cout << "actions[reindex[j]]  + default_dof_pos" << std::endl;
        const float takeoverElapsed = static_cast<float>(getTime() - _takeoverStartUs) / 1e6f;
        const float rawBlendAlpha = _activeBlendDurationSec <= 0.0f ? 1.0f :
            std::max(0.0f, std::min(
                1.0f, takeoverElapsed / _activeBlendDurationSec));
        // Smoothstep keeps both ends of the takeover free of a velocity step.
        const float blendAlpha = rawBlendAlpha * rawBlendAlpha *
            (3.0f - 2.0f * rawBlendAlpha);
        const Vec3 rpy = rotMatToRPY(_lowState->getRotMat());
        // ioInter::_base_w_pos is not populated by this Gazebo interface and
        // remained exactly zero in failed takeover diagnostics.  The estimator
        // is already updated by the control loop and is the same height source
        // used by FixedStand's stability gate.
        const double baseZ = _ctrlComp->estimator->getPosition()(2);
        double velocitySquare = 0.0;
        for(int i=0; i<12; ++i){
            velocitySquare += _lowState->motorState[i].dq * _lowState->motorState[i].dq;
        }
        const double jointVelocityRms = std::sqrt(velocitySquare / 12.0);
        const bool stable = std::abs(rpy(0)) < 0.15 && std::abs(rpy(1)) < 0.15 &&
            jointVelocityRms < 1.5 && _lowState->isFinite();
        const long long nowUs = getTime();
        if(stable){
            if(_stableSinceUs == 0){
                _stableSinceUs = nowUs;
            }
        }else{
            _stableSinceUs = 0;
        }
        const float stableElapsed = _stableSinceUs == 0 ? 0.0f :
            static_cast<float>(nowUs - _stableSinceUs) / 1e6f;
        // Latch readiness after the stationary takeover check succeeds.  Normal
        // walking necessarily violates the stationary joint-velocity threshold;
        // recomputing readiness here used to make Goal Executor alternate between
        // walking and zero commands until its progress watchdog fired.
        if(!_locomotionReady.load() && stable &&
           takeoverElapsed >= _activeBlendDurationSec +
                              _activeZeroHoldSec &&
           stableElapsed >= _activeZeroHoldSec){
            _locomotionReady.store(true);
            ROS_INFO("RL locomotion_ready latched after %.2f s takeover and %.2f s stable hold",
                     takeoverElapsed, stableElapsed);
        }
        // The failure watchdog belongs exclusively to the *pre-readiness*
        // stationary handoff.  Once readiness has been latched, walking and
        // stair climbing are expected to exceed the stationary joint/RPY
        // thresholds.  Applying this check afterwards was incorrectly
        // switching an otherwise healthy moving robot back to Passive after
        // roughly 7.5 s.
        if(!_locomotionReady.load() && !stable){
            if(takeoverElapsed >= _activeBlendDurationSec +
                                  _activeZeroHoldSec +
                                  _takeoverFailureGraceSec &&
               !_takeoverFailed.exchange(true)){
                _lowState->userCmd = UserCommand::L2_B;
                ROS_ERROR("RL takeover remained unstable for %.2f s; requesting passive state",
                          takeoverElapsed);
            }
        }
        double offsetSquare = 0.0;
        float offsetMaxAbs = 0.0f;
        for(int i=0; i<12; ++i){
            const float offset = joint_pos[i];
            offsetSquare += static_cast<double>(offset) * offset;
            offsetMaxAbs = std::max(offsetMaxAbs, std::abs(offset));
        }
        const double dofOffsetRms = std::sqrt(offsetSquare / 12.0);
        std_msgs::Bool readyMessage;
        readyMessage.data = _locomotionReady.load();
        _readyPub.publish(readyMessage);
        for(int i=0; i<12; i++){
            if (real == false)
            {
                // The joint controller canonicalizes limited revolute
                // joints, so the policy target must stay canonical too:
                // wrapToPi only, never a +2*pi equivalent of the measured q.
                const float policyTarget = wrapToPi(
                    actions[reindex[i]] +
                    default_dof_pos_tensor[reindex[i]].item<float>());
                _lowCmd->motorCmd[i].q =
                    (1.0f - blendAlpha) * _startPos[i] + blendAlpha * policyTarget;
                const int jointInLeg = i % 3;
                const float fixedKp = jointInLeg == 2 ? 140.0f : 95.0f;
                const float fixedKd = jointInLeg == 2 ? 7.0f : 5.0f;
                _lowCmd->motorCmd[i].Kp =
                    (1.0f - blendAlpha) * fixedKp + blendAlpha * 80.0f;
                _lowCmd->motorCmd[i].Kd =
                    (1.0f - blendAlpha) * fixedKd + blendAlpha * 1.0f;
            }
            else if (real == true)
            {
                // float t_joint = actions[reindex[i]]  + default_dof_pos_tensor[reindex[i]].item<float>();
                // std::vector<double> joint{t_joint, 0, 0, 80, 1};
                // _ctrlComp->ioInterFreeDog->setCmd(i,joint);
            }
            if (debug == true) std::cout << actions[reindex[i]]  + default_dof_pos_tensor[reindex[i]].item<float>() << " ";
        }
        if (debug == true)
            std::cout << std::endl;
        // Diagnostics are computed after this iteration's 12 blended commands
        // have been written, so commanded_position_error_rms reflects the
        // commands the next measured state will be compared against (the old
        // placement had a one-inference-period lag).
        double cmdErrorSquare = 0.0;
        for(int i=0; i<12; ++i){
            const float cmdError = shortestAngleDiff(
                _lowCmd->motorCmd[i].q, _lowState->motorState[i].q);
            cmdErrorSquare += static_cast<double>(cmdError) * cmdError;
        }
        const double commandedPositionErrorRms = std::sqrt(cmdErrorSquare / 12.0);
        const float commandVx = commands_tensor[0].item<float>();
        const float commandVy = commands_tensor[1].item<float>();
        const float commandWz = commands_tensor[2].item<float>();
        std::ostringstream status;
        status << "{\"timestamp\":" << ros::Time::now().toSec()
               << ",\"phase\":\"" << (_locomotionReady.load() ?
                    "LOCOMOTION_READY" :
                    (rawBlendAlpha < 1.0f ? "RL_BLEND" : "RL_ZERO_HOLD")) << "\""
               << ",\"blend_alpha\":" << blendAlpha
               << ",\"action_rms\":" << actionRms
               << ",\"roll\":" << rpy(0) << ",\"pitch\":" << rpy(1)
               << ",\"base_z\":" << baseZ
               << ",\"joint_velocity_rms\":" << jointVelocityRms
               << ",\"dof_offset_rms\":" << (std::isfinite(dofOffsetRms) ? dofOffsetRms : 0.0)
               << ",\"dof_offset_max_abs\":" << (std::isfinite(offsetMaxAbs) ? static_cast<double>(offsetMaxAbs) : 0.0)
               << ",\"commanded_position_error_rms\":" << (std::isfinite(commandedPositionErrorRms) ? commandedPositionErrorRms : 0.0)
               << ",\"command_vx\":" << (std::isfinite(commandVx) ? static_cast<double>(commandVx) : 0.0)
               << ",\"command_vy\":" << (std::isfinite(commandVy) ? static_cast<double>(commandVy) : 0.0)
               << ",\"command_wz\":" << (std::isfinite(commandWz) ? static_cast<double>(commandWz) : 0.0)
               << ",\"plane_tilt_guard_scale\":" << static_cast<double>(_planeTiltGuardScale)
               << ",\"plane_tilt_guard_hold\":" << (_planeTiltGuardHold ? "true" : "false")
               << ",\"locomotion_ready\":" << (_locomotionReady.load() ? "true" : "false") << "}";
        std_msgs::String statusMessage;
        statusMessage.data = status.str();
        _statusPub.publish(statusMessage);
        // std::cout << "actions_tensor: " << actions_tensor << std::endl;
        wait(_start_time, (long long)(infer_duration * 1000000));
    }
    infer_thread_runnning = State_RL::OVER;
}

void State_RL::save_amp_obs_thread()
{
    while(ampthreadRunning == State_RL::RUNNING)
    {
        long long _start_time = getTime();
        if ((getTime() - dofPosSwitBeginTime)<_duration) {
            _percent = (float)(getTime() - dofPosSwitBeginTime)/_duration;
            _percent = _percent > 1 ? 1 : _percent;
            std::cout << "_percent" << _percent << std::endl;
            // if (real == false){
                std::cout << "_lowCmd->motorCmd ";
                for(int j=0; j<12; j++){
                    std::cout << _targetPos_map[_cnt][reindex[j]] << " ";
                    _lowCmd->motorCmd[j].q = (1 - _percent)*_startPos[j] + _percent*_targetPos_map[_cnt][reindex[j]];
                }
                std::cout << _lowCmd->motorCmd << std::endl;
                std::cout << std::endl;
            // }
            // else if (real == true){
                std::cout << "target_joint";
                for(int j=0; j<12 && _ctrlComp->ioInterFreeDog; j++){
                    std::cout << _targetPos_map[_cnt][j] << " ";
                    float t_joint = (1 - _percent)*_startPos[j] + _percent*_targetPos_map[_cnt][reindex[j]];
                    std::vector<double> joint{t_joint, 0, 0, 80, 1};
                    _ctrlComp->ioInterFreeDog->setCmd(j,joint);
                }
                std::cout << std::endl;
            // }
            if ((float)(getTime() - dofPosSwitBeginTime)>(float)_duration*0.95)
                close_amp_save_file();
        }
        if (outfile.is_open())
        {
            std::cout << "save data" << std::endl;
            refresh_amp_obs();
        }
        wait(_start_time, (long long)(infer_duration * 1000000));
    }
    ampthreadRunning = State_RL::OVER;
}

void State_RL::updateCommandTensor(){
    // The policy receives zero velocity throughout takeover warmup.  Once ready
    // is latched, commands remain enabled until the RL state is exited.
    if(!_locomotionReady.load()){
        commands_tensor.zero_();
        return;
    }
    if(_keyboardMode.load()){
        _userValue = _lowState->userValue;
        commands_tensor[0] = finiteAxis(_userValue.ly) * _keyboardVxScale;
        commands_tensor[1] = -finiteAxis(_userValue.lx) * _keyboardVyScale;
        commands_tensor[2] = -finiteAxis(_userValue.rx) * _keyboardWzScale;
    }else{
        commands_tensor[0] = this->current_cmd_vel_.linear_x;
        commands_tensor[1] = this->current_cmd_vel_.linear_y;
        commands_tensor[2] = this->current_cmd_vel_.angular_z;
    }

    // The stair policy needs its natural climbing pitch.  On flat floors,
    // however, Round68 progressed from roll 0.27 to a side fall before the
    // global FSM safety threshold could react.  Attenuate the *requested*
    // motion early while leaving the learned plane policy active at zero
    // command so it can rebalance instead of being dropped into Passive.
    const bool planePolicy =
        _activePolicyPath.find("stair") == std::string::npos;
    if(!planePolicy){
        _planeTiltGuardHold = false;
        _planeTiltGuardScale = 1.0f;
        return;
    }
    const Vec3 rpy = rotMatToRPY(_lowState->getRotMat());
    const float tilt = static_cast<float>(std::max(
        std::abs(rpy(0)), std::abs(rpy(1))));
    if(!std::isfinite(tilt)){
        _planeTiltGuardHold = true;
    }else if(tilt >= _planeTiltGuardStopRad){
        if(!_planeTiltGuardHold){
            ROS_WARN("Plane tilt guard holding zero command at %.3f rad", tilt);
        }
        _planeTiltGuardHold = true;
    }else if(_planeTiltGuardHold && tilt <= _planeTiltGuardReleaseRad){
        _planeTiltGuardHold = false;
        ROS_INFO("Plane tilt guard released at %.3f rad", tilt);
    }
    if(_planeTiltGuardHold){
        _planeTiltGuardScale = 0.0f;
    }else if(tilt <= _planeTiltGuardWarningRad){
        _planeTiltGuardScale = 1.0f;
    }else{
        _planeTiltGuardScale = std::max(0.0f, std::min(
            1.0f, (_planeTiltGuardStopRad - tilt) /
                  (_planeTiltGuardStopRad - _planeTiltGuardWarningRad)));
        ROS_WARN_THROTTLE(
            1.0, "Plane tilt guard scaling cmd_vel by %.2f at tilt %.3f rad",
            _planeTiltGuardScale, tilt);
    }
    commands_tensor.mul_(_planeTiltGuardScale);
}



void State_RL::refresh_rl_obs(){
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    //gazebo simulation mode
    if (real == false)
    {
        // The competition stack deliberately disables /ground_truth.  Reading
        // ioInter->_base_w_ori here therefore left the policy with its default
        // non-unit quaternion (0, 0, 0, 0.1), even though /trunk_imu was live.
        // Gazebo's IMU angular velocity is already expressed in the body frame;
        // use it directly and rotate gravity with the measured IMU attitude.
        const Vec3 bodyAngularVelocity = _lowState->getGyro();
        const Vec3 projectedGravity = _lowState->getRotMat().transpose() * gravity;
        base_ang_vel_tensor = torch::tensor({
            static_cast<float>(bodyAngularVelocity(0)),
            static_cast<float>(bodyAngularVelocity(1)),
            static_cast<float>(bodyAngularVelocity(2))
        }, opts);
        projected_gravity_tensor = torch::tensor({
            static_cast<float>(projectedGravity(0)),
            static_cast<float>(projectedGravity(1)),
            static_cast<float>(projectedGravity(2))
        }, opts);
        
        //订阅cmd_vel
        // this->Sub_=nh.subscribe<geometry_msgs::Twist>("/cmd_vel",1000,boost::bind(&FSMState::cmdVelCallback,this,_1));

        updateCommandTensor();


        // std::cout << _ctrlComp->ioInter->axes << std::endl;
        // std::cout << "commands_tensor: " << commands_tensor << std::endl;
        for(int i=0; i<12; i++){
            // Loop slot i is already a policy-order slot after the forward
            // reindex below, so its nominal is default_dof_pos_tensor[i].
            // (The command loop separately uses reindex[i] there because that
            // loop index is controller-order: it maps the policy action back
            // to the controller joint.  Observation forward reindex and
            // command inverse mapping are different directions.)
            // Gazebo can report calf joints as canonical + 2*pi; feed the
            // policy the shortest 2*pi angular distance to its default pose
            // instead of the raw difference.
            joint_pos[i] = shortestAngleDiff(
                _lowState->motorState[reindex[i]].q,
                default_dof_pos_tensor[i].item<float>());
        }
        dof_pos_tensor = torch::from_blob(joint_pos.data(), {int64_t(joint_pos.size())}, opts).clone();
        // printTensorHorizontal(dof_pos_tensor,"dof_pos_tensor");
        for(int i=0; i<12; i++){
            joint_vel[i] = _lowState->motorState[reindex[i]].dq;
        }
        dof_vel_tensor = torch::from_blob(joint_vel.data(), {int64_t(joint_vel.size())}, opts).clone();
        // joint_pos already holds the shortest 2*pi distance to the default
        // pose; subtracting the default again would recreate the wrap error.
        obs_tensor = torch::cat({
            base_ang_vel_tensor * obs_scales_ang_vel,
            projected_gravity_tensor,
            commands_tensor * commands_scale,
            dof_pos_tensor * obs_scales_dof_pos,
            dof_vel_tensor * obs_scales_dof_vel,
            actions_tensor
        }, -1).to(device);
        obs_history_tensor = torch::cat({
            obs_history_tensor.slice(0, 1, HISTORY_LEN).to(device),  // 删除最早的一步
            obs_tensor.unsqueeze(0)  // 将当前 obs_tensor 插入到历史中
        }, 0);  // 按行（第0维）拼接
    }
    else if (real == true && _ctrlComp->ioInterFreeDog)
    {
        _B2G_RotMat = _ctrlComp->ioInterFreeDog->getRotMat();
        _G2B_RotMat = _B2G_RotMat.transpose();
        Vec3 projected_gravity = _G2B_RotMat*gravity;
        projected_gravity_tensor = torch::tensor({projected_gravity(0,0), projected_gravity(1,0), projected_gravity(2,0)});
         for (int i=0; i<3; i++) {
            base_ang_vel_tensor[i] = _ctrlComp->ioInterFreeDog->low_state.imu_gyroscope[i];
        }
        commands_tensor[0] = _ctrlComp->ioInter->axes[1];
        commands_tensor[1] = _ctrlComp->ioInter->axes[0];
        commands_tensor[2] = _ctrlComp->ioInter->axes[3]*3.14;
        for(int i=0; i<12; i++){
            joint_pos[i] = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[reindex[i]].q;
        }
        dof_pos_tensor = torch::from_blob(joint_pos.data(), {int64_t(joint_pos.size())}, opts).clone();
        for(int i=0; i<12; i++){
            joint_vel[i] = _ctrlComp->ioInterFreeDog->low_state.motorState_free_dog[reindex[i]].dq;
        }
        dof_vel_tensor = torch::from_blob(joint_vel.data(), {int64_t(joint_vel.size())}, opts).clone();
        obs_tensor = torch::cat({
            base_ang_vel_tensor * obs_scales_ang_vel,
            projected_gravity_tensor,
            commands_tensor * commands_scale,
            (dof_pos_tensor - default_dof_pos_tensor) * obs_scales_dof_pos,
            dof_vel_tensor * obs_scales_dof_vel,
            actions_tensor
        }, -1).to(device);
        obs_history_tensor = torch::cat({
            obs_history_tensor.slice(0, 1, HISTORY_LEN).to(device),  // 删除最早的一步
            obs_tensor.unsqueeze(0)  // 将当前 obs_tensor 插入到历史中
        }, 0);  // 按行（第0维）拼接
    }
}


void State_RL::refresh_rl_obs_real_robot(){

}

void State_RL::refresh_amp_obs(){
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    motion_time = static_cast<float>(getRosTime() - dofPosSwitBeginTime)/1e6;
    outfile << "motion_time: " << motion_time << std::endl;
    outfile << "base_w_pos: ";
    for (int i=0; i<3; i++) {
        base_w_pos[i] = _ctrlComp->ioInter->_base_w_pos[i];
        outfile << base_w_pos[i] << " ";
    }
    outfile << std::endl;

    outfile << "base_ori: ";
    for (int i=0; i<4; i++) {
        base_w_orientation[i] = _ctrlComp->ioInter->_base_w_ori[i];
        outfile << base_w_orientation[i] << " ";
    }
    outfile << std::endl;

    outfile << "dof_pos: ";
    for(int i=0; i<12; i++){
        joint_pos[i] = _lowState->motorState[reindex[i]].q;
        outfile << joint_pos[i] << " ";
    }
    outfile << std::endl;

    outfile << "foot_pos: ";
    for (int i=0; i<3; i++){
        foot_pos[0*3+i] = _ctrlComp->ioInter->_FL_foot_pos[i];
        foot_vel[0*3+i] = _ctrlComp->ioInter->_FL_foot_vel[i];
    }
    for (int i=0; i<3; i++){
        foot_pos[1*3+i] = _ctrlComp->ioInter->_FR_foot_pos[i];
        foot_vel[1*3+i] = _ctrlComp->ioInter->_FR_foot_vel[i];
    }
    for (int i=0; i<3; i++){
        foot_pos[2*3+i] = _ctrlComp->ioInter->_RL_foot_pos[i];
        foot_vel[2*3+i] = _ctrlComp->ioInter->_RL_foot_vel[i];
    }
    for (int i=0; i<3; i++){
        foot_pos[3*3+i] = _ctrlComp->ioInter->_RR_foot_pos[i];
        foot_vel[3*3+i] = _ctrlComp->ioInter->_RR_foot_vel[i];
    }
    for (  int  i=0;i<12;i++ )
    {
        outfile << foot_pos[i] << " ";
    }
    outfile << std::endl;

    outfile << "base_w_linear_vel: ";
    for (int i=0; i<3; i++) {
        base_w_linear_vel[i] = _ctrlComp->ioInter->_base_w_linear_vel[i];
    }
    torch::Tensor orientation_tensor = torch::from_blob(base_w_orientation.data(), {int64_t(base_w_orientation.size())}, opts).unsqueeze(0).clone();
    torch::Tensor w_linear_vel_tensor = torch::from_blob(base_w_linear_vel.data(), {int64_t(base_w_linear_vel.size())}, opts).unsqueeze(0).clone();
    torch::Tensor result = quat_rotate_inverse(orientation_tensor, w_linear_vel_tensor).squeeze().clone();
    for (int i = 0; i < 3; ++i) {
        base_linear_vel[i] = result[i].item<float>();
        // std::cout << base_linear_vel[i] << " ";
        outfile << base_linear_vel[i] << " ";
    }
    outfile << std::endl;

    outfile << "base_w_angular_vel: ";
    for (int i=0; i<3; i++) {
        base_w_angular_vel[i] = _ctrlComp->ioInter->_base_w_angular_vel[i];
    }
    torch::Tensor w_angular_vel_tensor = torch::from_blob(base_w_angular_vel.data(), {int64_t(base_w_angular_vel.size())}, opts).unsqueeze(0).clone();
    result = quat_rotate_inverse(orientation_tensor, w_angular_vel_tensor).squeeze().clone();
    for (int i = 0; i < 3; ++i) {
        base_angular_vel[i] = result[i].item<float>();
        // std::cout << base_angular_vel[i] << " ";
        outfile << base_angular_vel[i] << " ";
    }
    outfile << std::endl;

    outfile << "dof_vel: ";
    for(int i=0; i<12; i++){
        joint_vel[i] = _lowState->motorState[reindex[i]].dq;
        outfile << joint_vel[i] << " ";
    }
    outfile << std::endl;

    outfile << "foot_vel";
    for (  int  i=0;i<12;i++ )
    {
        outfile << foot_vel[i] << " ";
    }
    outfile << std::endl;
    outfile << std::endl;
    outfile << std::endl;
}

void State_RL::open_amp_save_file()
{
    // 打开文件输出流
    // 获取当前系统时间
    std::time_t cTime = std::time(nullptr);
    std::tm* currentTm = std::localtime(&cTime);
    // 构建文件名，格式为 systime + 年-月-日.txt
    std::ostringstream fileNameStream;
    fileNameStream << "/home/chy/log/gazebo/" << angle_names[_cnt];
    std::string fileName = fileNameStream.str();
    // 以追加模式打开文件
    outfile = std::ofstream(fileName, std::ios::out | std::ios::app);
    if (!outfile) {
        std::cerr << "无法打开文件!" << std::endl;
    } else {
        // std::cout << "文件打开成功!" << std::endl;
    }
}

void State_RL::close_amp_save_file()
{
    if (outfile.is_open()) {
        outfile.close();
        // std::cout << "文件关闭保存成功!" << std::endl;
    }
}

torch::Tensor State_RL::quat_rotate_inverse(const torch::Tensor& q, const torch::Tensor& v) {
    // Ensure q and v are of the correct shape: (batch_size, 4) for quaternions and (batch_size, 3) for vectors
    auto shape = q.sizes();
    // std::cout << "shape: " << shape << std::endl;
    auto q_w = q.index({torch::indexing::Slice(), 3});  // last column is the w component
    // std::cout << "q_w: " << q_w << std::endl;
    auto q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});  // first three columns are the vector part
    // std::cout << "q_vec: " << q_vec << std::endl;
    // a = v * (2.0 * q_w^2 - 1.0).unsqueeze(-1)
    auto a = v * (2.0 * q_w.pow(2) - 1.0).unsqueeze(-1);
    // std::cout << "a: " << a << std::endl;
    // b = cross(q_vec, v) * q_w.unsqueeze(-1) * 2.0
    auto b = torch::cross(q_vec, v, /*dim=*/-1) * q_w.unsqueeze(-1) * 2.0;
    // std::cout << "b: " << b << std::endl;
    // c = q_vec * torch::bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    auto q_vec_reshaped = q_vec.view({shape[0], 1, 3});
    // std::cout << "q_vec_reshaped: " << q_vec_reshaped << std::endl;
    auto v_reshaped = v.view({shape[0], 3, 1});
    // std::cout << "v_reshaped: " << v_reshaped << std::endl;
    auto c = q_vec * torch::bmm(q_vec_reshaped, v_reshaped).squeeze(-1) * 2.0;
    // std::cout << "c: " << c << std::endl;
    // Return a - b + c
    // std::cout << "a - b + c: " << a - b + c << std::endl;
    return a - b + c;
}

void State_RL::load_policy()
{
    // Flat indoor floors: plane policy is far more stable than stair.
    // Override with UNITREE_RL_POLICY=/path/to/policy.pt if needed.
    const char *policy_env = std::getenv("UNITREE_RL_POLICY");
    if(policy_env != nullptr && policy_env[0] != '\0'){
        model_path = policy_env;
    }else{
        model_path = "src/unitree_guide/logs/policy_act_inference_plane.pt";
    }
    std::cout << model_path << std::endl;
    // Load the model on the device selected for this specific policy.  In
    // hybrid mode the plane model stays isolated on CPU while stair models
    // use CUDA; the same selector is also called by both reload paths above.
    bool cacheHit = false;
    model = cachedPolicy(model_path, cacheHit);
    device = inferenceDeviceForPolicy(model_path);
    std::cout << "load model is successed!" << std::endl;
    std::cout << "load model to device!" << std::endl;
}

std::shared_ptr<torch::jit::script::Module> State_RL::cachedPolicy(
    const std::string& policyPath, bool& cacheHit)
{
    const auto found = _policyCache.find(policyPath);
    if(found != _policyCache.end()){
        cacheHit = true;
        return found->second;
    }
    cacheHit = false;
    const torch::DeviceType selectedDevice =
        inferenceDeviceForPolicy(policyPath);
    std::shared_ptr<torch::jit::script::Module> loaded =
        std::make_shared<torch::jit::script::Module>(
            torch::jit::load(policyPath));
    loaded->to(selectedDevice);
    loaded->eval();
    _policyCache.emplace(policyPath, loaded);
    std::cout << "[INFO] RL policy resident in preload cache: "
              << policyPath << std::endl;
    return loaded;
}

void State_RL::preloadConfiguredPolicies()
{
    const char *variables[] = {
        "UNITREE_RL_PRELOAD_PLANE_POLICY",
        "UNITREE_RL_PRELOAD_STAIR_POLICY"
    };
    for(const char *variable : variables){
        const char *value = std::getenv(variable);
        if(value == nullptr || value[0] == '\0') continue;
        const std::string policyPath(value);
        try{
            bool cacheHit = false;
            cachedPolicy(policyPath, cacheHit);
            std::cout << "[INFO] " << variable << " "
                      << (cacheHit ? "already cached: " : "preloaded: ")
                      << policyPath << std::endl;
        }catch(const c10::Error& error){
            // The initial policy has already loaded successfully.  Keep the
            // controller usable and let a later explicit request retry this
            // optional preload path with the existing fail-closed behavior.
            std::cerr << "[WARNING] Could not preload " << variable << ": "
                      << error.what() << std::endl;
        }
    }
}

torch::DeviceType State_RL::inferenceDeviceForPolicy(
    const std::string& policyPath) const
{
    const bool cuda_available = torch::cuda::is_available();
    const char *device_env = std::getenv("UNITREE_RL_DEVICE");
    const std::string requested_device = device_env == nullptr ? "auto" : device_env;
    const bool stair_policy =
        policyPath.find("stair") != std::string::npos;
    std::cout << "cuda::is_available():" << cuda_available << std::endl;
    const bool policy_requests_cuda =
        requested_device == "cuda" ||
        (requested_device == "auto" && cuda_available) ||
        (requested_device == "hybrid" && stair_policy);
    torch::DeviceType selected_device = torch::kCPU;
    if(policy_requests_cuda && !cuda_available){
        std::cout << "[WARNING] UNITREE_RL_DEVICE=" << requested_device
                  << " selected CUDA for this policy but CUDA is unavailable; using CPU."
                  << std::endl;
    }else if(policy_requests_cuda){
        selected_device = torch::kCUDA;
    }
    std::cout << "RL inference device: "
              << (selected_device == torch::kCUDA ? "cuda" : "cpu")
              << " (requested=" << requested_device
              << ", policy=" << (stair_policy ? "stair" : "plane")
              << ")" << std::endl;
    return selected_device;
}

void State_RL::printSegments(const torch::Tensor& tensor, int segment_size, const std::vector<int>& sub_sizes) {
    int num_segments = tensor.size(0) / segment_size;
    std::cout << "num_segments" << num_segments << tensor.size(0) << segment_size << std::endl;
    for (int seg = 0; seg < num_segments; ++seg) {
        auto segment = tensor.slice(0, seg * segment_size, (seg + 1) * segment_size);
        std::cout << "Segment " << seg + 1 << ":\n";

        int start = 0;
        for (size_t i = 0; i < sub_sizes.size(); ++i) {
            int size = sub_sizes[i];
            auto sub_segment = segment.slice(0, start, start + size);  // 按列（第1维）分割
            std::cout << "  Sub-segment " << i + 1 << " (" << size << " elements): ";
            std::string output_str = "  Sub-segment " + std::to_string(i + 1);
            printTensorHorizontal(sub_segment, output_str);
            start += size;
        }
    }
}

// 横排打印函数
void State_RL::printTensorHorizontal(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << " (" << tensor.sizes() << "): [ ";
    auto tensor_cpu = tensor.to(torch::kCPU);  // 确保张量在 CPU 上
    auto accessor = tensor_cpu.accessor<float, 1>();  // 假设是一维张量

    for (int i = 0; i < tensor.size(0); ++i) {
        std::cout << accessor[i] << " ";
    }
    std::cout << "]\n";
}
