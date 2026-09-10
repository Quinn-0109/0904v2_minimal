/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/
#include <iostream>
#include <unistd.h>
#include <csignal>
#include <sched.h>
#include <cerrno>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include "control/ControlFrame.h"
#include "control/CtrlComponents.h"
#include "Gait/WaveGenerator.h"
#include "control/BalanceCtrl.h"

#ifdef COMPILE_WITH_REAL_ROBOT
#include "interface/IOSDK.h"
#endif // COMPILE_WITH_REAL_ROBOT

#ifdef COMPILE_WITH_ROS
#include "interface/KeyBoard.h"
#include "interface/IOROS.h"
#include <std_msgs/Bool.h>
#endif // COMPILE_WITH_ROS

#include "interface/IOFREEDOGSDK.h"
bool running = true;

// over watch the ctrl+c command
void ShutDown(int sig)
{
    std::cout << "stop the controller" << std::endl;
    running = false;
}

void setProcessScheduler()
{
    const char *envValue = std::getenv("UNITREE_ENABLE_REALTIME");
    std::string realtimeMode = envValue == nullptr ? "auto" : envValue;
    std::transform(
        realtimeMode.begin(), realtimeMode.end(), realtimeMode.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    bool enableRealtime = false;
    if(realtimeMode == "auto" || realtimeMode.empty()){
        enableRealtime = geteuid() == 0;
    }else if(realtimeMode == "1" || realtimeMode == "true" || realtimeMode == "yes" || realtimeMode == "on"){
        enableRealtime = true;
    }else if(realtimeMode == "0" || realtimeMode == "false" || realtimeMode == "no" || realtimeMode == "off"){
        enableRealtime = false;
    }else{
        std::cout << "[WARNING] Invalid UNITREE_ENABLE_REALTIME='" << envValue
                  << "', using auto mode." << std::endl;
        enableRealtime = geteuid() == 0;
    }

    if(!enableRealtime){
        std::cout << "[INFO] Realtime scheduler disabled. Set UNITREE_ENABLE_REALTIME=1 to request SCHED_FIFO." << std::endl;
        return;
    }

    pid_t pid = getpid();
    sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(pid, SCHED_FIFO, &param) == -1)
    {
        std::cout << "[WARNING] Could not enable SCHED_FIFO scheduler: "
                  << std::strerror(errno)
                  << ". Controller will continue with the normal scheduler." << std::endl;
    }
    else
    {
        std::cout << "[INFO] SCHED_FIFO realtime scheduler enabled." << std::endl;
    }
}

double readControllerDt()
{
    const double defaultDt = 0.002; // 500 Hz, Unitree upstream default.
    const char *envValue = std::getenv("UNITREE_CTRL_DT");
    if (envValue == nullptr || envValue[0] == '\0')
    {
        return defaultDt;
    }

    char *end = nullptr;
    errno = 0;
    double dt = std::strtod(envValue, &end);
    if (errno != 0 || end == envValue || *end != '\0' || !std::isfinite(dt) || dt < 0.001 || dt > 0.02)
    {
        std::cout << "[WARNING] Invalid UNITREE_CTRL_DT='" << envValue
                  << "', using default " << defaultDt << "s." << std::endl;
        return defaultDt;
    }
    return dt;
}

double readPositiveEnv(const char *name, double fallback, double low, double high)
{
    const char *value = std::getenv(name);
    if(value == nullptr || value[0] == '\0') return fallback;
    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if(errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)
       || parsed < low || parsed > high){
        std::cout << "[WARNING] Invalid " << name << "='" << value
                  << "', using " << fallback << std::endl;
        return fallback;
    }
    return parsed;
}

int main(int argc, char **argv)
{
    /* set real-time process */
    setProcessScheduler();
    /* set the print format */
    std::cout << std::fixed << std::setprecision(3);

#ifdef RUN_ROS
    ros::init(argc, argv, "unitree_gazebo_servo");
#endif // RUN_ROS

    IOInterface *ioInter;
    CtrlPlatform ctrlPlat;

#ifdef COMPILE_WITH_SIMULATION
    ioInter = new IOROS();
    ctrlPlat = CtrlPlatform::GAZEBO;
#endif // COMPILE_WITH_SIMULATION

#ifdef COMPILE_WITH_REAL_ROBOT
    ioInter = new IOSDK();
    ctrlPlat = CtrlPlatform::REALROBOT;
#endif // COMPILE_WITH_REAL_ROBOT
    IOFREEDOGSDK *ioInter_freedog = nullptr;
#ifdef COMPILE_WITH_REAL_ROBOT
    ioInter_freedog = new IOFREEDOGSDK();
#endif // COMPILE_WITH_REAL_ROBOT
    CtrlComponents *ctrlComp = new CtrlComponents(ioInter,ioInter_freedog);
    ctrlComp->ctrlPlatform = ctrlPlat;
    ctrlComp->dt = readControllerDt();
    std::cout << "controller dt: " << ctrlComp->dt
              << " s, frequency: " << 1.0 / ctrlComp->dt << " Hz" << std::endl;
    ctrlComp->running = &running;

#ifdef ROBOT_TYPE_A1
    ctrlComp->robotModel = new A1Robot();
#endif
#ifdef ROBOT_TYPE_Go1
    ctrlComp->robotModel = new Go1Robot();
#endif

    const double gaitPeriod = readPositiveEnv("UNITREE_GAIT_PERIOD", 0.45, 0.30, 1.50);
    const double stanceRatio = readPositiveEnv("UNITREE_GAIT_STANCE_RATIO", 0.50, 0.30, 0.85);
    const char *profileValue = std::getenv("UNITREE_GAIT_PROFILE");
    const std::string gaitProfile = profileValue == nullptr ? "trot" : profileValue;
    const Vec4 gaitBias = gaitProfile == "crawl"
        ? Vec4(0, 0.25, 0.5, 0.75)
        : Vec4(0, 0.5, 0.5, 0);
    std::cout << "classic gait profile=" << gaitProfile
              << " period=" << gaitPeriod
              << "s stance_ratio=" << stanceRatio << std::endl;
    ctrlComp->waveGen = new WaveGenerator(gaitPeriod, stanceRatio, gaitBias);
    // ctrlComp->waveGen = new WaveGenerator(1.1, 0.75, Vec4(0, 0.25, 0.5, 0.75));  //Crawl, only for sim
    // ctrlComp->waveGen = new WaveGenerator(0.4, 0.6, Vec4(0, 0.5, 0.5, 0));  //Walking Trot, only for sim
    // ctrlComp->waveGen = new WaveGenerator(0.4, 0.35, Vec4(0, 0.5, 0.5, 0));  //Running Trot, only for sim
    // ctrlComp->waveGen = new WaveGenerator(0.4, 0.7, Vec4(0, 0, 0, 0));  //Pronk, only for sim

    ctrlComp->geneObj();

    ControlFrame ctrlFrame(ctrlComp);

#ifdef RUN_ROS
    // A latched readiness signal prevents automation from publishing a short
    // Joy command while the legacy controller is still loading its Torch
    // model.  A connected /joy subscriber alone is not sufficient because
    // IOROS is constructed before the FSM starts consuming commands.
    ros::NodeHandle readinessNode;
    ros::Publisher readinessPublisher = readinessNode.advertise<std_msgs::Bool>(
        "/simenv/classic_controller_ready", 1, true);
    std_msgs::Bool readinessMessage;
    readinessMessage.data = true;
    readinessPublisher.publish(readinessMessage);
#endif

    signal(SIGINT, ShutDown);

    while (running)
    {
        ctrlFrame.run();
    }

    delete ctrlComp;
    return 0;
}
