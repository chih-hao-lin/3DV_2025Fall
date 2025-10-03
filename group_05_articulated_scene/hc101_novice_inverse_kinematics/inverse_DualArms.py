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
import matplotlib.pyplot as plt


def main():
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[0.163, 0., 0.5 * pi, 0.],
                          [0., 0.632, pi, 0.5 * pi],
                          [0., 0.6005, pi, 0.],
                          [0.2013, 0., -0.5 * pi, -0.5 * pi],
                          [0.1025, 0., 0.5 * pi, 0.],
                          [0.094, 0., 0., 0.]])

    robot1 = RobotSerial(dh_params.copy())
    robot2 = RobotSerial(dh_params.copy())

    robot1.base_frame = Frame.from_euler_3(np.zeros(3), np.array([[0],[0],[0]]))
    robot2.base_frame = Frame.from_euler_3(np.zeros(3), np.array([[0.6],[0],[0]]))

    xyz1 = np.array([[0.3],[0.0],[1.0]])
    xyz2 = np.array([[0.3],[0.3],[1.0]])
    abc = np.array([0,0,0])  

    robot1.inverse(Frame.from_euler_3(abc, xyz1))
    robot2.inverse(Frame.from_euler_3(abc, xyz2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(robot1.axis_frames)-1):
        p1 = robot1.axis_frames[i].t_3_1.flatten()
        p2 = robot1.axis_frames[i+1].t_3_1.flatten()
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color='lightblue', marker='o')

    for i in range(len(robot2.axis_frames)-1):
        p1 = robot2.axis_frames[i].t_3_1.flatten()
        p2 = robot2.axis_frames[i+1].t_3_1.flatten()
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            color='lightpink', marker='o')

    plt.show()


if __name__ == "__main__":
    main()
