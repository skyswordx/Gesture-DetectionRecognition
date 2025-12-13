# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create `modules/fisheye_validation/` directory structure
  - Add Hypothesis to requirements.txt for property-based testing
  - Create `__init__.py` with module exports
  - _Requirements: 1.1, 3.2, 3.3_

- [x] 2. Implement CalibrationLoader component
  - [x] 2.1 Create ParsedCalibration dataclass and CalibrationLoader class
    - Implement `load()` method for JSON parsing
    - Implement `_detect_matlab_format()` for format detection
    - Implement `_convert_matlab_to_opencv()` for parameter conversion
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [x] 2.2 Write property test for calibration round-trip
    - **Property 1: Calibration Serialization Round-Trip**
    - **Validates: Requirements 1.1, 6.4**
  - [x] 2.3 Write property test for MATLAB format conversion
    - **Property 2: MATLAB Format Conversion Correctness**
    - **Validates: Requirements 1.2**
  - [x] 2.4 Write property test for input validation
    - **Property 3: Input Validation Completeness**
    - **Validates: Requirements 1.3**

- [x] 3. Implement DistortionSimulator component
  - [x] 3.1 Create DistortionSimulator class with fisheye model
    - Implement `_fisheye_distort_point()` using equidistant projection
    - Implement `_build_distortion_map()` for efficient remapping
    - Implement `apply_distortion()` main method
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x] 3.2 Write property test for fisheye distortion model
    - **Property 4: Fisheye Distortion Model Correctness**
    - **Validates: Requirements 2.1, 2.2**
  - [x] 3.3 Write property test for output dimension invariant
    - **Property 5: Output Dimension Invariant**
    - **Validates: Requirements 2.4**

- [x] 4. Implement Pretty-Printer functionality


  - [x] 4.1 Add pretty-print and serialization methods to CalibrationLoader
    - Implement `to_json()` method for serialization
    - Implement `pretty_print()` method for formatted output
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [x] 4.2 Write property test for pretty-print formatting





    - **Property 12: Pretty-Print Formatting**
    - **Validates: Requirements 6.1, 6.2, 6.3**




- [x] 5. Checkpoint - Ensure all tests pass


  - Ensure all tests pass, ask the user if questions arise.




- [x] 6. Implement ValidationEngine component



  - [x] 6.1 Create ValidationResult dataclass and ValidationEngine class

    - Implement `_compute_psnr()` method
    - Implement `_compute_ssim()` method
    - Implement `_compute_difference_map()` method
    - Implement `run_round_trip()` main method
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [ ]* 6.2 Write property test for round-trip image recovery
    - **Property 6: Round-Trip Image Recovery**
    - **Validates: Requirements 3.1**
  - [ ]* 6.3 Write property test for PSNR computation
    - **Property 7: PSNR Computation Correctness**
    - **Validates: Requirements 3.2**
  - [ ]* 6.4 Write property test for SSIM computation
    - **Property 8: SSIM Computation Correctness**
    - **Validates: Requirements 3.3**
  - [ ]* 6.5 Write property test for consistency threshold
    - **Property 9: Consistency Threshold Classification**
    - **Validates: Requirements 3.4**


- [x] 7. Implement PatternGenerator component





  - [x] 7.1 Create PatternGenerator class with static methods

    - Implement `checkerboard()` method
    - Implement `grid_lines()` method
    - Implement `radial_pattern()` method
    - Implement `load_image()` method
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [ ]* 7.2 Write property test for pattern generation
    - **Property 10: Pattern Generation Correctness**
    - **Validates: Requirements 4.1, 4.2, 4.3**




- [ ] 8. Implement ResultVisualizer component

  - [x] 8.1 Create ResultVisualizer class

    - Implement `show_comparison()` method
    - Implement `show_difference_heatmap()` method
    - Implement `annotate_metrics()` method
    - Implement `save_report()` method
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [ ]* 8.2 Write property test for difference heatmap
    - **Property 11: Difference Heatmap Correctness**
    - **Validates: Requirements 5.2**


- [x] 9. Checkpoint - Ensure all tests pass




  - Ensure all tests pass, ask the user if questions arise.


- [-] 10. Create validation script and integrate with existing system



  - [x] 10.1 Create main validation script

    - Create `validate_fisheye_calibration.py` as entry point
    - Integrate with existing `calibration_data/fisheye_calibration.json`
    - Add command-line argument parsing
    - _Requirements: 1.1, 3.1, 5.1_
  - [ ] 10.2 Update `modules/fisheye_validation/__init__.py` to export new components
    - Export ValidationEngine, PatternGenerator, ResultVisualizer
    - Ensure backward compatibility with existing code
    - _Requirements: 1.1, 2.1_



- [x] 11. Final Checkpoint - Ensure all tests pass



  - Ensure all tests pass, ask the user if questions arise.
