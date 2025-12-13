# Requirements Document

## Introduction

本功能旨在改进现有的距离测量模块，通过利用MediaPipe的3D世界坐标（pose_world_landmarks）来校正因人物姿态角度导致的2D投影偏差。核心目标是实现基于肩宽和胯宽变化的相对距离测量，即使在人物侧身情况下也能保持测量的准确性。

现有实现的问题：
- 仅使用2D像素距离进行距离估算
- 未利用MediaPipe提供的3D世界坐标
- 当人物侧身时，2D肩宽会显著缩小，导致距离估算严重偏差

解决方案：
- 使用MediaPipe的pose_world_landmarks获取真实3D世界坐标（以髋部中心为原点，单位为米）
- 通过3D坐标计算身体朝向角度，校正2D投影的姿态偏差
- 基于校正后的肩宽/胯宽变化来测量距离的相对变化

## Glossary

- **Pose_World_Landmarks**: MediaPipe输出的3D世界坐标，以髋部中心为原点，单位为米，不受相机位置影响
- **Pose_Landmarks**: MediaPipe输出的归一化2D图像坐标（0-1）加相对深度z
- **Body_Yaw**: 身体偏航角，表示人物左右转动的角度，正值表示向右转
- **Relative_Distance_Ratio**: 相对距离比值，相对于参考帧的距离变化，>1表示远离，<1表示靠近
- **Corrected_Width**: 校正后的宽度，消除姿态角度影响后的2D投影宽度
- **Reference_Frame**: 参考帧，用于计算相对距离变化的基准帧
- **Distance_Estimator**: 距离估算器，负责处理图像并输出距离测量结果

## Requirements

### Requirement 1

**User Story:** As a developer, I want to extract 3D world coordinates from MediaPipe pose detection, so that I can access pose-invariant body measurements.

#### Acceptance Criteria

1. WHEN the Distance_Estimator processes an image frame THEN the Distance_Estimator SHALL extract both pose_landmarks (2D) and pose_world_landmarks (3D) from MediaPipe results
2. WHEN pose_world_landmarks are available THEN the Distance_Estimator SHALL compute 3D Euclidean distances for shoulder width and hip width in meters
3. WHEN either shoulder or hip landmarks have visibility below 0.5 THEN the Distance_Estimator SHALL exclude those measurements from the result and set confidence accordingly
4. WHEN 3D coordinates are extracted THEN the Distance_Estimator SHALL store them in a structured Landmark3D data type containing x, y, z coordinates and visibility

### Requirement 2

**User Story:** As a developer, I want to estimate body orientation from 3D coordinates, so that I can correct for pose-induced projection distortions.

#### Acceptance Criteria

1. WHEN left and right shoulder 3D coordinates are available THEN the Distance_Estimator SHALL compute Body_Yaw angle using the arctangent of the z-difference over x-difference between shoulders
2. WHEN Body_Yaw is computed THEN the Distance_Estimator SHALL classify the pose as frontal (absolute yaw less than 30 degrees) or side view (absolute yaw 30 degrees or greater)
3. WHEN Body_Yaw exceeds 70 degrees THEN the Distance_Estimator SHALL reduce confidence by a factor of 0.5 to indicate unreliable measurement
4. WHEN computing body orientation THEN the Distance_Estimator SHALL output the yaw angle in degrees with positive values indicating rightward rotation

### Requirement 3

**User Story:** As a developer, I want to correct 2D projection measurements using body orientation, so that measurements remain stable regardless of pose angle.

#### Acceptance Criteria

1. WHEN 2D shoulder width and Body_Yaw are available THEN the Distance_Estimator SHALL compute Corrected_Width by dividing 2D width by the absolute cosine of Body_Yaw
2. WHEN the cosine of Body_Yaw is less than 0.1 THEN the Distance_Estimator SHALL clamp the divisor to 0.1 to prevent division by near-zero values
3. WHEN a person rotates from frontal to 45-degree side view THEN the Corrected_Width SHALL remain within 15 percent of the frontal measurement for the same physical distance
4. WHEN both shoulder and hip measurements are available THEN the Distance_Estimator SHALL apply the same correction formula to both measurements independently

### Requirement 4

**User Story:** As a user, I want to calibrate the system with a reference frame, so that I can measure relative distance changes from a known position.

#### Acceptance Criteria

1. WHEN the user triggers calibration THEN the Distance_Estimator SHALL store the current BodyMeasurement as Reference_Frame
2. WHEN calibration succeeds THEN the Distance_Estimator SHALL set is_calibrated flag to true and log the reference shoulder and hip widths
3. WHEN calibration is attempted with confidence below 0.5 THEN the Distance_Estimator SHALL reject the calibration and maintain previous state
4. WHEN the system is reset THEN the Distance_Estimator SHALL clear the Reference_Frame and set is_calibrated to false

### Requirement 5

**User Story:** As a user, I want to measure relative distance changes based on body width variations, so that I can track movement toward or away from the camera.

#### Acceptance Criteria

1. WHEN Reference_Frame exists and current measurement is available THEN the Distance_Estimator SHALL compute Relative_Distance_Ratio as reference Corrected_Width divided by current Corrected_Width
2. WHEN Relative_Distance_Ratio is greater than 1.0 THEN the Distance_Estimator SHALL indicate the subject is moving away from the camera
3. WHEN Relative_Distance_Ratio is less than 1.0 THEN the Distance_Estimator SHALL indicate the subject is approaching the camera
4. WHEN both shoulder and hip ratios are available THEN the Distance_Estimator SHALL compute a weighted combined ratio with shoulder weight of 0.6 and hip weight of 0.4
5. WHEN the subject moves 50 percent farther from the reference position THEN the Relative_Distance_Ratio SHALL be within 10 percent of 1.5

### Requirement 6

**User Story:** As a developer, I want the relative distance measurement to be robust to pose changes, so that side-facing poses do not cause false distance readings.

#### Acceptance Criteria

1. WHEN a subject maintains constant distance but rotates from frontal to 60-degree side view THEN the Relative_Distance_Ratio SHALL remain within 20 percent of 1.0
2. WHEN pose correction is applied THEN the standard deviation of Relative_Distance_Ratio over a 30-frame window during rotation SHALL be less than 0.15
3. WHEN comparing corrected versus uncorrected measurements during 45-degree rotation THEN the corrected measurement error SHALL be at least 50 percent smaller than uncorrected error
4. WHEN 3D world coordinates indicate constant shoulder width THEN the system SHALL use this as ground truth for validating pose correction effectiveness

### Requirement 7

**User Story:** As a developer, I want to visualize the distance measurement results, so that I can verify the system behavior in real-time.

#### Acceptance Criteria

1. WHEN processing a frame THEN the Distance_Estimator SHALL draw the pose skeleton on the output image
2. WHEN measurements are available THEN the Distance_Estimator SHALL display 3D shoulder width, 3D hip width, Body_Yaw angle, and Corrected_Width on the output image
3. WHEN Reference_Frame is set THEN the Distance_Estimator SHALL display Relative_Distance_Ratio with color coding (green for stable, orange for significant change)
4. WHEN the subject is approaching or moving away THEN the Distance_Estimator SHALL display a directional indicator text (APPROACHING, MOVING AWAY, or STABLE)

### Requirement 8

**User Story:** As a developer, I want to test the pose correction algorithm with simulated data, so that I can validate correctness without requiring live camera input.

#### Acceptance Criteria

1. WHEN testing pose correction THEN the test framework SHALL generate synthetic landmark data with known 3D positions and corresponding 2D projections
2. WHEN synthetic data simulates rotation THEN the test SHALL verify that Corrected_Width remains within specified tolerance of the true width
3. WHEN testing relative distance calculation THEN the test SHALL verify that distance ratio matches expected values for known position changes
4. WHEN testing edge cases THEN the test SHALL include scenarios with 0-degree, 45-degree, and 90-degree body rotations
