#include <ocs2_core/Types.h>
#include <ocs2_core/misc/CommandLine.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_legged_robot_ros/gait/GaitKeyboardPublisher.h>
#include <ros/init.h>
#include <ros/package.h>

using namespace ocs2;
using namespace legged_robot;

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "legged_robot_gait_command");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // ===== 修正:從 private 參數讀 robot_name =====
  std::string robotName;
  pnh.param<std::string>("robot_name", robotName, "legged_robot");

  // ===== 修正:從 namespace 讀 gaitCommandFile(去掉 "/") =====
  std::string gaitCommandFile;
  nh.getParam("gaitCommandFile", gaitCommandFile);

  GaitKeyboardPublisher gaitCommand(nh, gaitCommandFile, robotName, true);

  while (ros::ok() && ros::master::check()) {
    gaitCommand.getKeyboardCommand();
  }

  return 0;
}