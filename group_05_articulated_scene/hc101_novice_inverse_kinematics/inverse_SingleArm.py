#!/usr/bin/env python3

# Orginal Source: https://github.com/dbddqy/visual_kinematics?tab=readme-ov-file

# Disclosure of Usage of AI Assistant ---------------------------------------------------------------------------------------------------
# I used Gemini (version 2.5 Pro) for this assignment. 
# Access link: https://gemini.google.com/ 
# The tool was used several times in September 2025.
#  
# Initially, I used Gemini to understand the code.
# Then I used AI for drafting code to generate various kinds of results.

# All AI-generated content was thoroughly reviewed and revised by me to ensure its accuracy and relevance. 
# I can provide the unedited transcripts with prompts, interactions. 
# I take full responsibility for the final work, ensuring that it is my own.
# ----------------------------------------------------------------------------------------------------------------------------------------

from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi


def main():
    np.set_printoptions(precision=3, suppress=True)

    # DH parameter uses 4 prameters (this is used only for the initialization)
    # Rotation angle in z-axis, distance in z-axis
    # Rotation angle in x-axis, distance in x-axis
    # 6 DoF (2) which is tipical in robot arm
    # Degree-of-Freedom(1): # of variables in one joint
    # Degree-of-Freedom(2): total degrees of freedom of a robot system are the # of joints
    dh_params = np.array([[0.163, 0., 0.5 * pi, 0.],
                          [0., 0.632, pi, 0.5 * pi],
                          [0., 0.6005, pi, 0.],
                          [0.2013, 0., -0.5 * pi, -0.5 * pi],
                          [0.1025, 0., 0.5 * pi, 0.],
                          [0.094, 0., 0., 0.]])

    robot = RobotSerial(dh_params) # Initialize robot

    # There are successful, and unsucessful cases
    # When the target pose is out of workspace or the length of root arm, it becomes failure case
    # Example of unsuccessful case: [2.2], [0.], [1.9]
    xyz = np.array([[2.2], [0.], [1.9]]) # Object end-effector location 
    abc = np.array([0.5 * pi, 0., pi]) # ZYX eular angle
    end = Frame.from_euler_3(abc, xyz)

    # Depending on the randome seed, the results can be different
    robot.axis_values = np.array(87897, dtype=float)  
    
    # Main inverse kinematics function
    robot.inverse(end)

    print("inverse is successful: {0}".format(robot.is_reachable_inverse))
    print("axis values: \n{0}".format(robot.axis_values))
    robot.show()

# Functions for the reference inside the library
# Below part is the code analysis for the study
def inverse(self, end_frame):
    if self.analytical_inv is not None:
        # Analytical method is fast but hard to generalized to complex structures
        return self.inverse_analytical(end_frame, self.analytical_inv)
    else:
        # Numerical methods is slow but can be used for complex structures
        # Numerical + AI hybrid approaches are now used in many cases
        return self.inverse_numerical(end_frame) # My focus
    
def inverse_numerical(self, end_frame):
    last_dx = np.zeros([6, 1])
    for _ in range(self.max_iter):
        if self.inv_m == "jac_t":
            jac = self.jacobian.T
        else:
            jac = np.linalg.pinv(self.jacobian)
        # Current end-effector's frame (location, pose)
        end = self.end_frame
        dx = np.zeros([6, 1])
        # Differnce between target and current location
        dx[0:3, 0] = (end_frame.t_3_1 - end.t_3_1).reshape([3, ])
        # Difference between target and current rotation
        diff = end.inv * end_frame # Current frame -> target frame
        # Vectorization
        dx[3:6, 0] = end.r_3_3.dot(diff.r_3.reshape([3, 1])).reshape([3, ])
        # If the parameters are converged, return the values
        if np.linalg.norm(dx, ord=2) < self.final_loss or np.linalg.norm(dx - last_dx, ord=2) < 0.1*self.final_loss:
            self.axis_values = simplify_angles(self.axis_values)
            self.is_reachable_inverse = True
            return self.axis_values
        # Calculating updating amount using Jacobian
        # Jacobian is the derivation of the error between target and current end-effector
        # As a result, this value can be used to update joint parameters
        dq = self.step_size * jac.dot(dx)
        # Updating joint variables (location, angles)
        # At here, the Jacobian is also updated
        self.forward(self.axis_values + dq.reshape([self.num_axis, ])) 
        # Error update
        last_dx = dx
    logging.error("Pose cannot be reached!")
    self.is_reachable_inverse = False

if __name__ == "__main__":
    main()
