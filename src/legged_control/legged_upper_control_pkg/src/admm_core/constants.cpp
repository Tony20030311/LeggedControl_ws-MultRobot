// Matrix builders for admm_constants.hpp (mirrors constants.py np.block layout).
#include "legged_upper_control/admm_constants.hpp"

namespace admm {

Eigen::Matrix4d make_Ad() {
    Eigen::Matrix4d Ad = Eigen::Matrix4d::Identity();
    Ad(0, 2) = TS;
    Ad(1, 3) = TS;
    return Ad;
}

Eigen::Matrix<double, 4, 2> make_Bd() {
    Eigen::Matrix<double, 4, 2> Bd = Eigen::Matrix<double, 4, 2>::Zero();
    Bd(0, 0) = BD_POS_COEF;
    Bd(1, 1) = BD_POS_COEF;
    Bd(2, 0) = BD_VEL_COEF;
    Bd(3, 1) = BD_VEL_COEF;
    return Bd;
}

Eigen::Vector3d hocbf_coefs() { return Eigen::Vector3d(COEF_HK, COEF_HK1, COEF_HK2); }

}  // namespace admm
