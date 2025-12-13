# Implementation Plan

- [x] 1. Set up project structure and data models
  - [x] 1.1 Create data structure classes (Landmark3D, Landmark2D, BodyMeasurement, RelativeDistanceResult)
    - Define dataclasses with all required fields
    - Add type hints and documentation
    - _Requirements: 1.4_
  - [x] 1.2 Write property test for 3D Euclidean distance calculation
    - **Property 1: 3D Euclidean Distance Calculation**
    - **Validates: Requirements 1.2**
  - [x] 1.3 Write unit tests for data structure validation
    - Test dataclass creation with valid and invalid values
    - _Requirements: 1.4_

- [x] 2. Implement landmark extraction from MediaPipe
  - [x] 2.1 Create landmark extraction functions
    - Extract pose_landmarks (2D) from MediaPipe results
    - Extract pose_world_landmarks (3D) from MediaPipe results
    - Convert to Landmark2D and Landmark3D data types
    - _Requirements: 1.1, 1.4_
  - [x] 2.2 Implement visibility filtering logic
    - Filter landmarks with visibility below 0.5
    - Adjust confidence based on visible landmarks
    - _Requirements: 1.3_
  - [x] 2.3 Write property test for visibility filtering
    - **Property 2: Visibility Filtering**
    - **Validates: Requirements 1.3**

- [x] 3. Implement body measurement calculation
  - [x] 3.1 Implement 3D Euclidean distance calculation
    - Calculate shoulder width from 3D coordinates
    - Calculate hip width from 3D coordinates
    - _Requirements: 1.2_
  - [x] 3.2 Implement 2D pixel distance calculation
    - Calculate shoulder width in pixels
    - Calculate hip width in pixels
    - _Requirements: 3.1_
  - [x] 3.3 Create BodyMeasurement computation function
    - Combine 3D and 2D measurements
    - Set confidence based on visibility
    - _Requirements: 1.2, 1.3_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement body orientation estimation
  - [x] 5.1 Implement body yaw calculation from 3D shoulder coordinates
    - Use atan2(dz, dx) formula
    - Convert to degrees with correct sign convention
    - _Requirements: 2.1, 2.4_
  - [x] 5.2 Write property test for body yaw calculation
    - **Property 3: Body Yaw Calculation**
    - **Validates: Requirements 2.1, 2.4**
  - [x] 5.3 Implement frontal/side view classification
    - Classify as frontal if |yaw| < 30°
    - Classify as side view if |yaw| >= 30°
    - _Requirements: 2.2_
  - [x] 5.4 Write property test for frontal/side classification
    - **Property 4: Frontal/Side Classification**
    - **Validates: Requirements 2.2**
  - [x] 5.5 Implement confidence adjustment for extreme yaw
    - Reduce confidence by 0.5 when |yaw| > 70°
    - _Requirements: 2.3_

- [x] 6. Implement pose correction algorithm
  - [x] 6.1 Implement corrected width calculation
    - Apply formula: Corrected_Width = 2D_Width / |cos(yaw)|
    - Clamp divisor to minimum 0.1 for extreme angles
    - _Requirements: 3.1, 3.2_
  - [x] 6.2 Write property test for pose correction formula
    - **Property 5: Pose Correction Formula**
    - **Validates: Requirements 3.1, 3.2**
  - [x] 6.3 Apply correction to both shoulder and hip measurements
    - Correct shoulder width independently
    - Correct hip width independently
    - _Requirements: 3.4_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement calibration and reference frame management
  - [x] 8.1 Implement reference frame storage
    - Store current BodyMeasurement as reference
    - Set is_calibrated flag on success
    - _Requirements: 4.1, 4.2_
  - [x] 8.2 Implement calibration validation
    - Reject calibration if confidence < 0.5
    - Maintain previous state on rejection
    - _Requirements: 4.3_
  - [x] 8.3 Implement reset functionality
    - Clear reference frame
    - Set is_calibrated to false
    - _Requirements: 4.4_
  - [x] 8.4 Write property test for calibration state management
    - **Property 7: Calibration State Management**
    - **Validates: Requirements 4.1, 4.2, 4.3**
  - [x] 8.5 Write property test for reset behavior
    - **Property 8: Reset Behavior**
    - **Validates: Requirements 4.4**

- [x] 9. Implement relative distance calculation
  - [x] 9.1 Implement relative distance ratio calculation
    - Compute ratio = Corrected_Width_ref / Corrected_Width_current
    - _Requirements: 5.1_
  - [x] 9.2 Write property test for relative distance ratio formula
    - **Property 9: Relative Distance Ratio Formula**
    - **Validates: Requirements 5.1**
  - [x] 9.3 Implement direction indication logic
    - Return "MOVING AWAY" for ratio > 1.0
    - Return "APPROACHING" for ratio < 1.0
    - Return "STABLE" for ratio == 1.0
    - _Requirements: 5.2, 5.3_
  - [x] 9.4 Write property test for direction indication
    - **Property 10: Direction Indication**
    - **Validates: Requirements 5.2, 5.3**
  - [x] 9.5 Implement weighted ratio combination
    - Combine shoulder (0.6) and hip (0.4) ratios
    - _Requirements: 5.4_
  - [x] 9.6 Write property test for weighted ratio combination
    - **Property 11: Weighted Ratio Combination**
    - **Validates: Requirements 5.4**

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement synthetic data generator for testing
  - [x] 11.1 Create synthetic landmark generator
    - Generate 3D landmarks with known geometry
    - Compute corresponding 2D projections based on camera model
    - Support configurable distance, yaw angle, and body dimensions
    - _Requirements: 8.1_
  - [x] 11.2 Implement rotation simulation
    - Generate landmark sequences for rotation from 0° to 90°
    - Maintain constant 3D shoulder width during rotation
    - _Requirements: 8.4_

- [x] 12. Implement pose correction effectiveness tests




  - [x] 12.1 Write property test for pose correction stability
    - **Property 6: Pose Correction Stability**
    - Test that corrected width remains within 15% during 0° to 45° rotation
    - **Validates: Requirements 3.3**
  - [x] 12.2 Write property test for distance ratio accuracy


    - **Property 12: Distance Ratio Accuracy**
    - Test that 50% distance change produces ratio within 10% of 1.5
    - **Validates: Requirements 5.5**
  - [x] 12.3 Write property test for rotation robustness


    - **Property 13: Rotation Robustness**
    - Test that ratio remains within 20% of 1.0 during 0° to 60° rotation at constant distance
    - **Validates: Requirements 6.1**

  - [x] 12.4 Write property test for correction improvement

    - **Property 14: Correction Improvement**
    - Test that corrected error is at least 50% smaller than uncorrected error at 45° rotation
    - **Validates: Requirements 6.3**

- [x] 13. Implement visualization
  - [x] 13.1 Implement pose skeleton drawing
    - Draw MediaPipe pose landmarks on output image
    - _Requirements: 7.1_
  - [x] 13.2 Implement measurement info display
    - Display 3D shoulder/hip width, Body_Yaw, Corrected_Width
    - _Requirements: 7.2_
  - [x] 13.3 Implement distance ratio display with color coding
    - Green for stable (0.95-1.05)
    - Orange for significant change
    - _Requirements: 7.3_
  - [x] 13.4 Implement direction indicator
    - Display APPROACHING, MOVING AWAY, or STABLE text
    - _Requirements: 7.4_

- [x] 14. Integrate components into main estimator class
  - [x] 14.1 Implement PoseCorrectedDistanceEstimator class
    - Integrate all components
    - Implement process_frame method
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_
  - [x] 14.2 Implement RelativeDistanceTracker class
    - Add history management
    - Add statistics computation
    - _Requirements: 5.1, 6.2_

- [x] 15. Create live demo application
  - [x] 15.1 Implement run_live_demo function
    - Open camera capture
    - Process frames in real-time
    - Handle keyboard input for calibration (C), reset (R), statistics (S), quit (Q)
    - _Requirements: 4.1, 4.4, 7.1-7.4_

- [x] 16. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
