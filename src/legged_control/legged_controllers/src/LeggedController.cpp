//
// Created by qiayuan on 2022/6/24.
//

#include <pinocchio/fwd.hpp>

#include "legged_controllers/LeggedController.h"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_core/thread_support/ExecuteAndSleep.h>
#include <ocs2_core/thread_support/SetThreadPriority.h>
#include <ocs2_legged_robot_ros/gait/GaitReceiver.h>
#include <ocs2_msgs/mpc_observation.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_ros_interfaces/common/RosMsgConversions.h>
#include <ocs2_ros_interfaces/synchronized_module/RosReferenceManager.h>
#include <ocs2_sqp/SqpMpc.h>

#include <angles/angles.h>
#include <cmath>
#include <legged_estimation/FromTopiceEstimate.h>
#include <legged_estimation/LinearKalmanFilter.h>
#include <legged_wbc/HierarchicalWbc.h>
#include <legged_wbc/WeightedWbc.h>
#include <pluginlib/class_list_macros.hpp>

namespace legged {
bool LeggedController::init(hardware_interface::RobotHW* robot_hw,
                            ros::NodeHandle& controller_nh) {
  controllerNh_ = controller_nh;  // 存起來,setupMpc() 會用
    // ===== DEBUG =====
  ROS_WARN_STREAM("[DEBUG] controller_nh.getNamespace() = '" 
                  << controller_nh.getNamespace() << "'");
  ROS_WARN_STREAM("[DEBUG] parent x1 = '" 
                  << ros::names::parentNamespace(controller_nh.getNamespace()) << "'");
  ROS_WARN_STREAM("[DEBUG] parent x2 = '" 
                  << ros::names::parentNamespace(
                     ros::names::parentNamespace(controller_nh.getNamespace())) << "'");
  ROS_WARN_STREAM("[DEBUG] ros::this_node::getNamespace() = '" 
                  << ros::this_node::getNamespace() << "'");
  std::string urdfFile;
  std::string taskFile;
  std::string referenceFile;

  // ===== 修改:去掉 getParam 的 "/" =====
  //
  // 【原始】controller_nh.getParam("/urdfFile", urdfFile);
  // 【問題】帶 "/" → 去全域 /urdfFile 找 → 五隻狗讀到同一個
  // 【修改】去掉 "/" → 從 controller_nh 的 namespace 讀
  //         controller_nh = /dog1/controllers/legged_controller
  //         所以讀的是 /dog1/controllers/legged_controller/urdfFile
  controller_nh.getParam("urdfFile", urdfFile);
  controller_nh.getParam("taskFile", taskFile);
  controller_nh.getParam("referenceFile", referenceFile);
  bool verbose = false;
  loadData::loadCppDataType(taskFile, "legged_robot_interface.verbose", verbose);

  setupLeggedInterface(taskFile, urdfFile, referenceFile, verbose);
  setupMpc();
  setupMrt();

  // ===== 修正:Visualization 的 NodeHandle 要用對的 namespace =====
  //
  // 原始:ros::NodeHandle nh(controller_nh.getNamespace());
  //   這會把 nh 設成 /dog1/controllers/legged_controller(太深)
  //
  // 修正:用 parentNamespace 兩次拉回 /dog1
  //   controller_nh.getNamespace() = /dog1/controllers/legged_controller
  //   第一次 parentNamespace → /dog1/controllers
  //   第二次 parentNamespace → /dog1
  //   所以 nh = /dog1,Visualizer 發的 marker 就在 /dog1/xxx
  ros::NodeHandle nh(ros::names::parentNamespace(
                     ros::names::parentNamespace(
                     controller_nh.getNamespace())));
  accelCmd_.x = 0.0;
  accelCmd_.y = 0.0;
  accelCmd_.z = 0.0;
  accelCmdStamp_ = ros::Time(0);
  controller_nh.param("accelCmdTimeout", accelCmdTimeout_, accelCmdTimeout_);
  accelCmdSubscriber_ = nh.subscribe<geometry_msgs::Vector3>(
      "accel_cmd", 1, &LeggedController::accelCmdCallback, this);

  CentroidalModelPinocchioMapping pinocchioMapping(leggedInterface_->getCentroidalModelInfo());
  eeKinematicsPtr_ = std::make_shared<PinocchioEndEffectorKinematics>(
      leggedInterface_->getPinocchioInterface(), pinocchioMapping,
      leggedInterface_->modelSettings().contactNames3DoF);
  robotVisualizer_ = std::make_shared<LeggedRobotVisualizer>(
      leggedInterface_->getPinocchioInterface(),
      leggedInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_, nh);
  selfCollisionVisualization_.reset(new LeggedSelfCollisionVisualization(
      leggedInterface_->getPinocchioInterface(),
      leggedInterface_->getGeometryInterface(), pinocchioMapping, nh));

  // Hardware interface
  auto* hybridJointInterface = robot_hw->get<HybridJointInterface>();
  if (!hybridJointInterface) {
    ROS_ERROR("HybridJointInterface is null");
    return false;
  }
  std::vector<std::string> joint_names{"LF_HAA", "LF_HFE", "LF_KFE", "LH_HAA", "LH_HFE", "LH_KFE",
                                       "RF_HAA", "RF_HFE", "RF_KFE", "RH_HAA", "RH_HFE", "RH_KFE"};
  for (const auto& joint_name : joint_names) {
    hybridJointHandles_.push_back(hybridJointInterface->getHandle(joint_name));
  }
  auto* contactInterface = robot_hw->get<ContactSensorInterface>();
  if (!contactInterface) {
    ROS_ERROR("ContactSensorInterface is null");
    return false;
  }
  for (const auto& name : leggedInterface_->modelSettings().contactNames3DoF) {
    contactHandles_.push_back(contactInterface->getHandle(name));
  }
  imuSensorHandle_ = robot_hw->get<hardware_interface::ImuSensorInterface>()->getHandle("base_imu");

  // State estimation
  setupStateEstimate(taskFile, verbose);

  // Whole body control
  wbc_ = std::make_shared<WeightedWbc>(leggedInterface_->getPinocchioInterface(),
                                       leggedInterface_->getCentroidalModelInfo(),
                                       *eeKinematicsPtr_);
  wbc_->loadTasksSetting(taskFile, verbose);

  // Safety Checker
  safetyChecker_ = std::make_shared<SafetyChecker>(leggedInterface_->getCentroidalModelInfo());

  return true;
}

void LeggedController::starting(const ros::Time& time) {
  currentObservation_.state.setZero(leggedInterface_->getCentroidalModelInfo().stateDim);
  updateStateEstimation(time, ros::Duration(0.002));
  currentObservation_.input.setZero(leggedInterface_->getCentroidalModelInfo().inputDim);
  currentObservation_.mode = ModeNumber::STANCE;

  TargetTrajectories target_trajectories({currentObservation_.time},
                                          {currentObservation_.state},
                                          {currentObservation_.input});

  mpcMrtInterface_->setCurrentObservation(currentObservation_);
  mpcMrtInterface_->getReferenceManager().setTargetTrajectories(target_trajectories);
  ROS_INFO_STREAM("Waiting for the initial policy ...");
  while (!mpcMrtInterface_->initialPolicyReceived() && ros::ok()) {
    mpcMrtInterface_->advanceMpc();
    ros::WallRate(leggedInterface_->mpcSettings().mrtDesiredFrequency_).sleep();
  }
  ROS_INFO_STREAM("Initial policy has been received.");

  mpcRunning_ = true;
}

void LeggedController::update(const ros::Time& time, const ros::Duration& period) {
  updateStateEstimation(time, period);
  mpcMrtInterface_->setCurrentObservation(currentObservation_);
  mpcMrtInterface_->updatePolicy();

  vector_t optimizedState, optimizedInput;
  size_t plannedMode = 0;
  mpcMrtInterface_->evaluatePolicy(currentObservation_.time, currentObservation_.state,
                                    optimizedState, optimizedInput, plannedMode);

  currentObservation_.input = optimizedInput;

  wbcTimer_.startTimer();
  const bool accelCmdFresh =
      !accelCmdStamp_.isZero() && (time - accelCmdStamp_).toSec() <= accelCmdTimeout_;
  const scalar_t accelCmdX = accelCmdFresh ? accelCmd_.x : 0.0;
  const scalar_t accelCmdY = accelCmdFresh ? accelCmd_.y : 0.0;
  const scalar_t yaw = measuredRbdState_(0);
  const scalar_t cosYaw = std::cos(yaw);
  const scalar_t sinYaw = std::sin(yaw);
  const scalar_t accelXWorld = cosYaw * accelCmdX - sinYaw * accelCmdY;
  const scalar_t accelYWorld = sinYaw * accelCmdX + cosYaw * accelCmdY;
  wbc_->setUpperLayerAccel(accelXWorld, accelYWorld);
  vector_t x = wbc_->update(optimizedState, optimizedInput, measuredRbdState_,
                             plannedMode, period.toSec());
  wbcTimer_.endTimer();

  vector_t torque = x.tail(12);

  vector_t posDes = centroidal_model::getJointAngles(optimizedState,
                                                      leggedInterface_->getCentroidalModelInfo());
  vector_t velDes = centroidal_model::getJointVelocities(optimizedInput,
                                                          leggedInterface_->getCentroidalModelInfo());

  if (!safetyChecker_->check(currentObservation_, optimizedState, optimizedInput)) {
    ROS_ERROR_STREAM("[Legged Controller] Safety check failed, stopping the controller.");
    stopRequest(time);
  }

  for (size_t j = 0; j < leggedInterface_->getCentroidalModelInfo().actuatedDofNum; ++j) {
    hybridJointHandles_[j].setCommand(posDes(j), velDes(j), 0, 3, torque(j));
  }

  robotVisualizer_->update(currentObservation_, mpcMrtInterface_->getPolicy(),
                            mpcMrtInterface_->getCommand());
  selfCollisionVisualization_->update(currentObservation_);

  observationPublisher_.publish(ros_msg_conversions::createObservationMsg(currentObservation_));
}

void LeggedController::accelCmdCallback(const geometry_msgs::Vector3ConstPtr& msg) {
  accelCmd_ = *msg;
  accelCmdStamp_ = ros::Time::now();
}

void LeggedController::updateStateEstimation(const ros::Time& time, const ros::Duration& period) {
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  contact_flag_t contacts;
  Eigen::Quaternion<scalar_t> quat;
  contact_flag_t contactFlag;
  vector3_t angularVel, linearAccel;
  matrix3_t orientationCovariance, angularVelCovariance, linearAccelCovariance;

  for (size_t i = 0; i < hybridJointHandles_.size(); ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }
  for (size_t i = 0; i < contacts.size(); ++i) {
    contactFlag[i] = contactHandles_[i].isContact();
  }
  for (size_t i = 0; i < 4; ++i) {
    quat.coeffs()(i) = imuSensorHandle_.getOrientation()[i];
  }
  for (size_t i = 0; i < 3; ++i) {
    angularVel(i) = imuSensorHandle_.getAngularVelocity()[i];
    linearAccel(i) = imuSensorHandle_.getLinearAcceleration()[i];
  }
  for (size_t i = 0; i < 9; ++i) {
    orientationCovariance(i) = imuSensorHandle_.getOrientationCovariance()[i];
    angularVelCovariance(i) = imuSensorHandle_.getAngularVelocityCovariance()[i];
    linearAccelCovariance(i) = imuSensorHandle_.getLinearAccelerationCovariance()[i];
  }

  stateEstimate_->updateJointStates(jointPos, jointVel);
  stateEstimate_->updateContact(contactFlag);
  stateEstimate_->updateImu(quat, angularVel, linearAccel, orientationCovariance,
                             angularVelCovariance, linearAccelCovariance);
  measuredRbdState_ = stateEstimate_->update(time, period);
  currentObservation_.time += period.toSec();
  scalar_t yawLast = currentObservation_.state(9);
  currentObservation_.state = rbdConversions_->computeCentroidalStateFromRbdModel(measuredRbdState_);
  currentObservation_.state(9) = yawLast + angles::shortest_angular_distance(yawLast,
                                                                              currentObservation_.state(9));
  currentObservation_.mode = stateEstimate_->getMode();
}

LeggedController::~LeggedController() {
  controllerRunning_ = false;
  if (mpcThread_.joinable()) {
    mpcThread_.join();
  }
  std::cerr << "########################################################################";
  std::cerr << "\n### MPC Benchmarking";
  std::cerr << "\n###   Maximum : " << mpcTimer_.getMaxIntervalInMilliseconds() << "[ms].";
  std::cerr << "\n###   Average : " << mpcTimer_.getAverageInMilliseconds() << "[ms]." << std::endl;
  std::cerr << "########################################################################";
  std::cerr << "\n### WBC Benchmarking";
  std::cerr << "\n###   Maximum : " << wbcTimer_.getMaxIntervalInMilliseconds() << "[ms].";
  std::cerr << "\n###   Average : " << wbcTimer_.getAverageInMilliseconds() << "[ms].";
}

void LeggedController::setupLeggedInterface(const std::string& taskFile,
                                             const std::string& urdfFile,
                                             const std::string& referenceFile,
                                             bool verbose) {
  leggedInterface_ = std::make_shared<LeggedInterface>(taskFile, urdfFile, referenceFile);
  leggedInterface_->setupOptimalControlProblem(taskFile, urdfFile, referenceFile, verbose);
}

void LeggedController::setupMpc() {
  // ===== 補齊:建立 SqpMpc 和 rbdConversions =====
  // 原版的程式碼(你的版本漏了這兩行,mpc_ 會是 nullptr 直接 segfault)
  mpc_ = std::make_shared<SqpMpc>(leggedInterface_->mpcSettings(),
                                   leggedInterface_->sqpSettings(),
                                   leggedInterface_->getOptimalControlProblem(),
                                   leggedInterface_->getInitializer());
  rbdConversions_ = std::make_shared<CentroidalModelRbdConversions>(
      leggedInterface_->getPinocchioInterface(),
      leggedInterface_->getCentroidalModelInfo());

  // ===== 從參數讀 robotName =====
  // 原版:const std::string robotName = "legged_robot";  硬寫
  // 問題:五隻狗的 MPC topic 都叫 legged_robot_mpc_observation → 相撞
  // 修正:從 /dog1/controllers/legged_controller/robot_name 讀,得到 "dog1"
  std::string robotName;
  if (!controllerNh_.getParam("robot_name", robotName)) {
    robotName = "legged_robot";
    ROS_WARN("[LeggedController::setupMpc] robot_name not found, using default: legged_robot");
  }

  // ===== 用帶 namespace 的 NodeHandle =====
  // 原版:ros::NodeHandle nh;  namespace 是 / (全域)
  // 問題:即使 topic 名是 dog1_mpc_observation,也會被解析成 /dog1_mpc_observation
  //       而不是 /dog1/dog1_mpc_observation
  // 修正:用 parentNamespace 兩次把深度拉回 /dog1/
  //       controllerNh_ = /dog1/controllers/legged_controller
  //       parentNamespace x2 → /dog1/
  ros::NodeHandle nh(ros::names::parentNamespace(
                     ros::names::parentNamespace(
                     controllerNh_.getNamespace())));

  // Gait receiver:訂閱 robotName + "_mpc_mode_schedule"
  auto gaitReceiverPtr = std::make_shared<GaitReceiver>(
      nh, leggedInterface_->getSwitchedModelReferenceManagerPtr()->getGaitSchedule(), robotName);

  // ROS ReferenceManager:訂閱 robotName + "_mpc_target"
  auto rosReferenceManagerPtr = std::make_shared<RosReferenceManager>(
      robotName, leggedInterface_->getReferenceManagerPtr());
  rosReferenceManagerPtr->subscribe(nh);

  mpc_->getSolverPtr()->addSynchronizedModule(gaitReceiverPtr);
  mpc_->getSolverPtr()->setReferenceManager(rosReferenceManagerPtr);

  // 發布 robotName + "_mpc_observation"
  observationPublisher_ = nh.advertise<ocs2_msgs::mpc_observation>(
      robotName + "_mpc_observation", 1);
}

void LeggedController::setupMrt() {
  mpcMrtInterface_ = std::make_shared<MPC_MRT_Interface>(*mpc_);
  mpcMrtInterface_->initRollout(&leggedInterface_->getRollout());
  mpcTimer_.reset();

  controllerRunning_ = true;
  mpcThread_ = std::thread([&]() {
    while (controllerRunning_) {
      try {
        executeAndSleep(
            [&]() {
              if (mpcRunning_) {
                mpcTimer_.startTimer();
                mpcMrtInterface_->advanceMpc();
                mpcTimer_.endTimer();
              }
            },
            leggedInterface_->mpcSettings().mpcDesiredFrequency_);
      } catch (const std::exception& e) {
        controllerRunning_ = false;
        ROS_ERROR_STREAM("[Ocs2 MPC thread] Error : " << e.what());
        stopRequest(ros::Time());
      }
    }
  });
  setThreadPriority(leggedInterface_->sqpSettings().threadPriority, mpcThread_);
}

void LeggedController::setupStateEstimate(const std::string& taskFile, bool verbose) {
  stateEstimate_ = std::make_shared<KalmanFilterEstimate>(
      leggedInterface_->getPinocchioInterface(),
      leggedInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_);
  dynamic_cast<KalmanFilterEstimate&>(*stateEstimate_).loadSettings(taskFile, verbose);
  currentObservation_.time = 0;
}

void LeggedCheaterController::setupStateEstimate(const std::string& /*taskFile*/, bool /*verbose*/) {
  stateEstimate_ = std::make_shared<FromTopicStateEstimate>(
      leggedInterface_->getPinocchioInterface(),
      leggedInterface_->getCentroidalModelInfo(), *eeKinematicsPtr_);
}

}  // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::LeggedController, controller_interface::ControllerBase)
PLUGINLIB_EXPORT_CLASS(legged::LeggedCheaterController, controller_interface::ControllerBase)
