# Robot Control - Homework 2

This is second homework for Robot Control Course at MIMUW (2024/25). Project is divided into three subtasks. In the
first task a car has to reach the red sphere. In the second task it has to escape the maze and reach the sphere again.
In the last exercise, it has to determine its position using infomration about Aruco cube, teleport to the red sphere (
position of the sphere is known) and grab the ball.

Everything is done using only infromation from robot dash camera.

## Prerequsites

In order to run this script you need to have installed:

- mujoco==3.2.3
- numpy==2.2.0
- opencv_python==4.10.0.84

While writing the script I used Python 3.11

## Task 1

In the first task a robot makes a rotation until it sees the red ball. After that it goes towards it, correcting its
route if necessary. In the end it slows down to get close to the ball.

## Task 2

In the second task, the robot moves to the middle of the corridor and starts rotating. It localizes a pole in the middle
of the map and from now on starts escaping the maze. In the end it gets close to the sphere.

## Task 3

In the last exercise, the robot localizes coordinates of the aruco markers on the image. Thanks to intrinsic matrix and
distortion parameters that are known, we know the relative position of Aruco markers to the dashcam. Thanks to that, the
robot can teleport and later catch the sphere and move it up.