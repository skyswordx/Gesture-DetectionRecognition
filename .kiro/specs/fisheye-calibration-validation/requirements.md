# Requirements Document

## Introduction

本功能旨在通过虚拟数据集验证鱼眼相机标定参数的正确性。核心思路是实现一个 round-trip 测试：将无畸变图像通过标定参数添加鱼眼畸变，再使用同一组参数去畸变，最后与原图对比以量化评估标定精度。这种方法可以在没有真实鱼眼相机的情况下验证标定参数的数学一致性。

## Glossary

- **Fisheye_Distortion_Model**: OpenCV 鱼眼畸变模型，使用等距投影 (equidistant projection)，畸变公式为 θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
- **Camera_Matrix (K)**: 3×3 相机内参矩阵，包含焦距 (fx, fy) 和主点 (cx, cy)
- **Distortion_Coefficients (D)**: 鱼眼畸变系数向量 [k1, k2, k3, k4]
- **Forward_Projection**: 将无畸变图像坐标映射到畸变图像坐标的过程（添加畸变）
- **Backward_Projection**: 将畸变图像坐标映射到无畸变图像坐标的过程（去畸变）
- **Round_Trip_Test**: 对图像先添加畸变再去畸变，验证能否恢复原图的测试方法
- **PSNR**: 峰值信噪比，用于量化图像相似度的指标
- **SSIM**: 结构相似性指数，用于评估图像结构相似度

## Requirements

### Requirement 1

**User Story:** As a developer, I want to load and parse MATLAB-exported fisheye calibration parameters, so that I can use them for validation testing.

#### Acceptance Criteria

1. WHEN the system loads a calibration JSON file THEN the Calibration_Loader SHALL parse camera_matrix and dist_coeffs fields correctly
2. WHEN the camera_matrix contains normalized focal lengths (fx ≈ 1, fy ≈ 1) and dist_coeffs[0] is a large scaling factor THEN the Calibration_Loader SHALL automatically convert to pixel-level intrinsics using the formula fx_pixel = alpha × fx_normalized
3. WHEN parsing is complete THEN the Calibration_Loader SHALL validate that camera_matrix is 3×3 and dist_coeffs has exactly 4 elements
4. IF the calibration file is missing required fields THEN the Calibration_Loader SHALL return an error with a descriptive message

### Requirement 2

**User Story:** As a developer, I want to apply fisheye distortion to an undistorted image, so that I can create synthetic distorted test images.

#### Acceptance Criteria

1. WHEN applying forward projection to an undistorted image THEN the Distortion_Simulator SHALL use the OpenCV fisheye equidistant model: θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
2. WHEN computing distorted coordinates THEN the Distortion_Simulator SHALL apply the camera matrix K to convert between normalized and pixel coordinates
3. WHEN the distorted coordinate falls outside the image bounds THEN the Distortion_Simulator SHALL handle boundary pixels gracefully using interpolation or border handling
4. WHEN forward projection is complete THEN the Distortion_Simulator SHALL output an image with the same dimensions as the input

### Requirement 3

**User Story:** As a developer, I want to perform round-trip validation (distort then undistort), so that I can verify the mathematical consistency of calibration parameters.

#### Acceptance Criteria

1. WHEN performing round-trip validation THEN the Validation_Engine SHALL first apply forward projection (add distortion) then backward projection (remove distortion)
2. WHEN round-trip is complete THEN the Validation_Engine SHALL compute PSNR between the original and recovered images
3. WHEN round-trip is complete THEN the Validation_Engine SHALL compute SSIM between the original and recovered images
4. WHEN PSNR exceeds 30 dB and SSIM exceeds 0.95 THEN the Validation_Engine SHALL report the calibration as mathematically consistent
5. IF round-trip produces significant artifacts or black regions THEN the Validation_Engine SHALL flag potential parameter issues

### Requirement 4

**User Story:** As a developer, I want to generate various test patterns for validation, so that I can thoroughly test the calibration across different image content.

#### Acceptance Criteria

1. WHEN generating test patterns THEN the Pattern_Generator SHALL create checkerboard patterns with configurable square sizes
2. WHEN generating test patterns THEN the Pattern_Generator SHALL create grid line patterns for visual distortion assessment
3. WHEN generating test patterns THEN the Pattern_Generator SHALL create radial patterns to highlight barrel/pincushion distortion
4. WHEN a custom image path is provided THEN the Pattern_Generator SHALL load and use that image as the test source

### Requirement 5

**User Story:** As a developer, I want to visualize the validation results, so that I can qualitatively assess the calibration quality.

#### Acceptance Criteria

1. WHEN displaying results THEN the Visualizer SHALL show side-by-side comparison of original, distorted, and recovered images
2. WHEN displaying results THEN the Visualizer SHALL overlay a difference heatmap highlighting pixel-level discrepancies
3. WHEN displaying results THEN the Visualizer SHALL annotate the image with computed metrics (PSNR, SSIM, max error)
4. WHEN the user requests it THEN the Visualizer SHALL save the comparison images to a specified output directory

### Requirement 6

**User Story:** As a developer, I want to pretty-print calibration parameters for debugging, so that I can verify the parsed values are correct.

#### Acceptance Criteria

1. WHEN pretty-printing calibration THEN the Pretty_Printer SHALL display the camera matrix in a formatted 3×3 layout
2. WHEN pretty-printing calibration THEN the Pretty_Printer SHALL display distortion coefficients with scientific notation for small values
3. WHEN pretty-printing calibration THEN the Pretty_Printer SHALL indicate whether MATLAB-to-OpenCV conversion was applied
4. WHEN serializing calibration THEN the Pretty_Printer SHALL output valid JSON that can be re-loaded by the Calibration_Loader
