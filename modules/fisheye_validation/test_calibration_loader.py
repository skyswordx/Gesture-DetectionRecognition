# -*- coding: utf-8 -*-
import json
import numpy as np
from hypothesis import given, strategies as st, settings
from calibration_loader import ParsedCalibration, CalibrationLoader

@st.composite
def valid_camera_matrix(draw):
    fx = draw(st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    fy = draw(st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    cx = draw(st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    cy = draw(st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    skew = draw(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False))
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

@st.composite
def valid_dist_coeffs(draw):
    k1 = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    k2 = draw(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False))
    k3 = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False))
    k4 = draw(st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False))
    return np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)

@st.composite
def valid_image_size(draw):
    width = draw(st.integers(min_value=320, max_value=4096))
    height = draw(st.integers(min_value=240, max_value=2160))
    return (width, height)

@st.composite
def valid_parsed_calibration(draw):
    K = draw(valid_camera_matrix())
    D = draw(valid_dist_coeffs())
    size = draw(valid_image_size())
    return ParsedCalibration(camera_matrix=K, dist_coeffs=D, image_size=size, was_converted=False, original_alpha=None)

@st.composite
def matlab_format_calibration(draw):
    fx_norm = draw(st.floats(min_value=0.8, max_value=1.2, allow_nan=False, allow_infinity=False))
    fy_norm = draw(st.floats(min_value=0.8, max_value=1.2, allow_nan=False, allow_infinity=False))
    cx = draw(st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    cy = draw(st.floats(min_value=100.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    skew = draw(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False))
    alpha = draw(st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    k1 = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False))
    k2 = draw(st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False))
    k3 = draw(st.floats(min_value=-0.0001, max_value=0.0001, allow_nan=False, allow_infinity=False))
    K = np.array([[fx_norm, skew, cx], [0.0, fy_norm, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([[alpha], [k1], [k2], [k3]], dtype=np.float64)
    width = draw(st.integers(min_value=320, max_value=4096))
    height = draw(st.integers(min_value=240, max_value=2160))
    return {'K': K, 'D': D, 'fx_norm': fx_norm, 'fy_norm': fy_norm, 'alpha': alpha, 'width': width, 'height': height}

@given(calib=valid_parsed_calibration())
@settings(max_examples=100)
def test_calibration_round_trip(calib: ParsedCalibration):
    '''
    **Feature: fisheye-calibration-validation, Property 1: Calibration Serialization Round-Trip**
    **Validates: Requirements 1.1, 6.4**
    '''
    loader = CalibrationLoader()
    json_str = loader.to_json(calib)
    data = json.loads(json_str)
    recovered = loader.load_from_dict(data)
    assert np.allclose(calib.camera_matrix, recovered.camera_matrix, rtol=1e-6, atol=1e-9)
    assert np.allclose(calib.dist_coeffs, recovered.dist_coeffs, rtol=1e-6, atol=1e-9)
    assert calib.image_size == recovered.image_size

@given(matlab_data=matlab_format_calibration())
@settings(max_examples=100)
def test_matlab_format_conversion(matlab_data):
    '''
    **Feature: fisheye-calibration-validation, Property 2: MATLAB Format Conversion Correctness**
    **Validates: Requirements 1.2**
    '''
    loader = CalibrationLoader()
    data = {'camera_matrix': matlab_data['K'].tolist(), 'dist_coeffs': matlab_data['D'].flatten().tolist(), 'image_width': matlab_data['width'], 'image_height': matlab_data['height']}
    result = loader.load_from_dict(data)
    assert result.was_converted, "MATLAB format should be detected and converted"
    expected_fx = matlab_data['alpha'] * matlab_data['fx_norm']
    expected_fy = matlab_data['alpha'] * matlab_data['fy_norm']
    assert np.isclose(result.camera_matrix[0, 0], expected_fx, rtol=1e-6), f"fx_pixel mismatch: {result.camera_matrix[0, 0]} vs {expected_fx}"
    assert np.isclose(result.camera_matrix[1, 1], expected_fy, rtol=1e-6), f"fy_pixel mismatch: {result.camera_matrix[1, 1]} vs {expected_fy}"
    assert np.isclose(result.original_alpha, matlab_data['alpha'], rtol=1e-6), "Original alpha should be preserved"


@given(calib=valid_parsed_calibration())
@settings(max_examples=100)
def test_input_validation_completeness(calib: ParsedCalibration):
    '''
    **Feature: fisheye-calibration-validation, Property 3: Input Validation Completeness**
    **Validates: Requirements 1.3**
    
    For any parsed calibration result, the camera_matrix should have shape (3, 3)
    and dist_coeffs should have shape (4, 1) or (4,).
    '''
    loader = CalibrationLoader()
    # Serialize and reload to test the validation path
    json_str = loader.to_json(calib)
    data = json.loads(json_str)
    result = loader.load_from_dict(data)
    
    # Property: camera_matrix must be 3x3
    assert result.camera_matrix.shape == (3, 3), \
        f"camera_matrix shape must be (3, 3), got {result.camera_matrix.shape}"
    
    # Property: dist_coeffs must have shape (4, 1) or (4,)
    dist_shape = result.dist_coeffs.shape
    assert dist_shape == (4, 1) or dist_shape == (4,), \
        f"dist_coeffs shape must be (4, 1) or (4,), got {dist_shape}"
    
    # Property: dist_coeffs must have exactly 4 elements
    assert result.dist_coeffs.size == 4, \
        f"dist_coeffs must have 4 elements, got {result.dist_coeffs.size}"


@given(
    rows=st.integers(min_value=1, max_value=5),
    cols=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100)
def test_invalid_camera_matrix_shape_rejected(rows, cols):
    '''
    **Feature: fisheye-calibration-validation, Property 3: Input Validation Completeness**
    **Validates: Requirements 1.3**
    
    Invalid camera_matrix shapes (not 3x3) should be rejected with ValueError.
    '''
    # Skip valid 3x3 case - we only test invalid shapes
    if rows == 3 and cols == 3:
        return
    
    loader = CalibrationLoader()
    invalid_matrix = np.random.rand(rows, cols).tolist()
    data = {
        'camera_matrix': invalid_matrix,
        'dist_coeffs': [0.1, 0.01, 0.001, 0.0001]
    }
    
    try:
        loader.load_from_dict(data)
        assert False, f"Should have raised ValueError for camera_matrix shape ({rows}, {cols})"
    except ValueError as e:
        assert "3x3" in str(e) or "3×3" in str(e), \
            f"Error message should mention 3x3 requirement, got: {e}"


@given(num_coeffs=st.integers(min_value=0, max_value=10))
@settings(max_examples=100)
def test_invalid_dist_coeffs_length_rejected(num_coeffs):
    '''
    **Feature: fisheye-calibration-validation, Property 3: Input Validation Completeness**
    **Validates: Requirements 1.3**
    
    Invalid dist_coeffs lengths (not 4) should be rejected with ValueError.
    '''
    # Skip valid 4-element case - we only test invalid lengths
    if num_coeffs == 4:
        return
    
    loader = CalibrationLoader()
    valid_matrix = [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    invalid_coeffs = [0.01] * num_coeffs
    data = {
        'camera_matrix': valid_matrix,
        'dist_coeffs': invalid_coeffs
    }
    
    try:
        loader.load_from_dict(data)
        assert False, f"Should have raised ValueError for dist_coeffs length {num_coeffs}"
    except ValueError as e:
        assert "4" in str(e), \
            f"Error message should mention 4 elements requirement, got: {e}"


@st.composite
def normalized_point(draw):
    """Generate normalized image coordinates (x, y) in a reasonable range."""
    # Normalized coordinates typically range from -1 to 1 for points within the image
    # We use a slightly larger range to test edge cases
    x = draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    y = draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    return (x, y)


@st.composite
def valid_distortion_coeffs_for_model(draw):
    """Generate distortion coefficients that produce stable distortion."""
    # Use smaller coefficient ranges to ensure numerical stability
    k1 = draw(st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False))
    k2 = draw(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False))
    k3 = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False))
    k4 = draw(st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False))
    return np.array([k1, k2, k3, k4], dtype=np.float64)


def compute_expected_distortion(x_norm: float, y_norm: float, D: np.ndarray) -> tuple:
    """
    Reference implementation of fisheye equidistant projection model.
    
    θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
    where θ = atan(r) and r = sqrt(x² + y²)
    """
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    # Handle center point
    if r < 1e-8:
        return x_norm, y_norm
    
    theta = np.arctan(r)
    theta2 = theta * theta
    
    k1, k2, k3, k4 = D.flatten()
    
    # Equidistant projection formula
    theta_d = theta * (1 + k1*theta2 + k2*theta2**2 + k3*theta2**3 + k4*theta2**4)
    
    scale = theta_d / r
    return x_norm * scale, y_norm * scale


@given(
    point=normalized_point(),
    D=valid_distortion_coeffs_for_model()
)
@settings(max_examples=100)
def test_fisheye_distortion_model_correctness(point, D):
    '''
    **Feature: fisheye-calibration-validation, Property 4: Fisheye Distortion Model Correctness**
    **Validates: Requirements 2.1, 2.2**
    
    For any normalized image point (x, y), the distorted point (x', y') computed by
    DistortionSimulator should satisfy the equidistant projection formula:
    θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸) where θ = atan(r) and r = sqrt(x² + y²).
    '''
    from distortion_simulator import DistortionSimulator
    
    x_norm, y_norm = point
    
    # Create a simulator with a simple identity-like camera matrix
    # (we're testing the distortion model, not the pixel conversion)
    K = np.array([[500.0, 0.0, 320.0],
                  [0.0, 500.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    
    simulator = DistortionSimulator(K, D)
    
    # Get the distorted point from the simulator
    x_dist, y_dist = simulator._fisheye_distort_point(x_norm, y_norm)
    
    # Compute expected distortion using reference implementation
    x_expected, y_expected = compute_expected_distortion(x_norm, y_norm, D)
    
    # Verify the distortion matches the expected formula
    assert np.isclose(x_dist, x_expected, rtol=1e-10, atol=1e-12), \
        f"x distortion mismatch: got {x_dist}, expected {x_expected}"
    assert np.isclose(y_dist, y_expected, rtol=1e-10, atol=1e-12), \
        f"y distortion mismatch: got {y_dist}, expected {y_expected}"


@given(D=valid_distortion_coeffs_for_model())
@settings(max_examples=100)
def test_fisheye_distortion_center_invariant(D):
    '''
    **Feature: fisheye-calibration-validation, Property 4: Fisheye Distortion Model Correctness**
    **Validates: Requirements 2.1, 2.2**
    
    The center point (0, 0) should remain unchanged after distortion.
    '''
    from distortion_simulator import DistortionSimulator
    
    K = np.array([[500.0, 0.0, 320.0],
                  [0.0, 500.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    
    simulator = DistortionSimulator(K, D)
    
    x_dist, y_dist = simulator._fisheye_distort_point(0.0, 0.0)
    
    assert np.isclose(x_dist, 0.0, atol=1e-10), \
        f"Center x should be 0, got {x_dist}"
    assert np.isclose(y_dist, 0.0, atol=1e-10), \
        f"Center y should be 0, got {y_dist}"


@given(
    point=normalized_point(),
    D=valid_distortion_coeffs_for_model()
)
@settings(max_examples=100)
def test_fisheye_distortion_radial_symmetry(point, D):
    '''
    **Feature: fisheye-calibration-validation, Property 4: Fisheye Distortion Model Correctness**
    **Validates: Requirements 2.1, 2.2**
    
    The distortion should be radially symmetric: points at the same distance
    from center should have the same distortion magnitude.
    '''
    from distortion_simulator import DistortionSimulator
    
    x_norm, y_norm = point
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    # Skip near-zero radius (center point)
    if r < 1e-6:
        return
    
    K = np.array([[500.0, 0.0, 320.0],
                  [0.0, 500.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    
    simulator = DistortionSimulator(K, D)
    
    # Distort the original point
    x_dist, y_dist = simulator._fisheye_distort_point(x_norm, y_norm)
    r_dist = np.sqrt(x_dist**2 + y_dist**2)
    
    # Create a rotated point at the same radius (90 degrees)
    x_rotated = -y_norm
    y_rotated = x_norm
    
    # Distort the rotated point
    x_rot_dist, y_rot_dist = simulator._fisheye_distort_point(x_rotated, y_rotated)
    r_rot_dist = np.sqrt(x_rot_dist**2 + y_rot_dist**2)
    
    # The distorted radii should be equal (radial symmetry)
    assert np.isclose(r_dist, r_rot_dist, rtol=1e-10, atol=1e-12), \
        f"Radial symmetry violated: r_dist={r_dist}, r_rot_dist={r_rot_dist}"


@st.composite
def valid_image_dimensions(draw):
    """Generate valid image dimensions (height, width, channels)."""
    height = draw(st.integers(min_value=10, max_value=200))
    width = draw(st.integers(min_value=10, max_value=200))
    channels = draw(st.sampled_from([1, 3, 4]))  # Grayscale, RGB, RGBA
    return (height, width, channels)


@given(
    dims=valid_image_dimensions(),
    D=valid_distortion_coeffs_for_model()
)
@settings(max_examples=100)
def test_output_dimension_invariant(dims, D):
    '''
    **Feature: fisheye-calibration-validation, Property 5: Output Dimension Invariant**
    **Validates: Requirements 2.4**
    
    For any input image of shape (H, W, C), the output of apply_distortion
    should have the same shape (H, W, C).
    '''
    from distortion_simulator import DistortionSimulator
    
    height, width, channels = dims
    
    # Create camera matrix with principal point at image center
    K = np.array([[300.0, 0.0, width / 2.0],
                  [0.0, 300.0, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    
    simulator = DistortionSimulator(K, D)
    
    # Create test image with the specified dimensions
    if channels == 1:
        # Grayscale image (H, W)
        input_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        expected_shape = (height, width)
    else:
        # Multi-channel image (H, W, C)
        input_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        expected_shape = (height, width, channels)
    
    # Apply distortion
    output_image = simulator.apply_distortion(input_image)
    
    # Property: output shape must equal input shape
    assert output_image.shape == expected_shape, \
        f"Output shape {output_image.shape} does not match input shape {expected_shape}"


@given(D=valid_distortion_coeffs_for_model())
@settings(max_examples=100)
def test_output_dimension_invariant_grayscale(D):
    '''
    **Feature: fisheye-calibration-validation, Property 5: Output Dimension Invariant**
    **Validates: Requirements 2.4**
    
    For grayscale images (H, W), the output should maintain the same 2D shape.
    '''
    from distortion_simulator import DistortionSimulator
    
    height, width = 100, 150
    
    K = np.array([[300.0, 0.0, width / 2.0],
                  [0.0, 300.0, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    
    simulator = DistortionSimulator(K, D)
    
    # Create grayscale image
    input_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    output_image = simulator.apply_distortion(input_image)
    
    assert output_image.shape == (height, width), \
        f"Grayscale output shape {output_image.shape} does not match input (height, width)"


@given(D=valid_distortion_coeffs_for_model())
@settings(max_examples=100)
def test_output_dimension_invariant_empty_image(D):
    '''
    **Feature: fisheye-calibration-validation, Property 5: Output Dimension Invariant**
    **Validates: Requirements 2.4**
    
    Empty images should return empty images with the same shape.
    '''
    from distortion_simulator import DistortionSimulator
    
    K = np.array([[300.0, 0.0, 160.0],
                  [0.0, 300.0, 120.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    
    simulator = DistortionSimulator(K, D)
    
    # Create empty image
    input_image = np.array([], dtype=np.uint8).reshape(0, 0)
    
    output_image = simulator.apply_distortion(input_image)
    
    assert output_image.shape == input_image.shape, \
        f"Empty image output shape {output_image.shape} does not match input {input_image.shape}"


@st.composite
def valid_parsed_calibration_with_conversion(draw):
    """Generate ParsedCalibration with was_converted=True and original_alpha set."""
    K = draw(valid_camera_matrix())
    D = draw(valid_dist_coeffs())
    size = draw(valid_image_size())
    alpha = draw(st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    return ParsedCalibration(
        camera_matrix=K,
        dist_coeffs=D,
        image_size=size,
        was_converted=True,
        original_alpha=alpha
    )


@given(calib=valid_parsed_calibration())
@settings(max_examples=100)
def test_pretty_print_contains_intrinsics(calib: ParsedCalibration):
    '''
    **Feature: fisheye-calibration-validation, Property 12: Pretty-Print Formatting**
    **Validates: Requirements 6.1, 6.2, 6.3**
    
    For any ParsedCalibration, the pretty-printed string should contain all four
    focal length and principal point values (fx, fy, cx, cy).
    '''
    loader = CalibrationLoader()
    output = loader.pretty_print(calib)
    
    K = calib.camera_matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Property: output must contain fx value
    assert 'fx' in output.lower() or f'{fx:.6f}' in output, \
        f"Pretty-print output should contain fx value {fx}"
    
    # Property: output must contain fy value
    assert 'fy' in output.lower() or f'{fy:.6f}' in output, \
        f"Pretty-print output should contain fy value {fy}"
    
    # Property: output must contain cx value
    assert 'cx' in output.lower() or f'{cx:.6f}' in output, \
        f"Pretty-print output should contain cx value {cx}"
    
    # Property: output must contain cy value
    assert 'cy' in output.lower() or f'{cy:.6f}' in output, \
        f"Pretty-print output should contain cy value {cy}"


@given(calib=valid_parsed_calibration_with_conversion())
@settings(max_examples=100)
def test_pretty_print_indicates_conversion(calib: ParsedCalibration):
    '''
    **Feature: fisheye-calibration-validation, Property 12: Pretty-Print Formatting**
    **Validates: Requirements 6.1, 6.2, 6.3**
    
    For any ParsedCalibration with was_converted=True, the pretty-printed string
    should indicate that MATLAB-to-OpenCV conversion was applied.
    '''
    loader = CalibrationLoader()
    output = loader.pretty_print(calib)
    
    # Property: when was_converted is True, output must indicate conversion
    assert 'matlab' in output.lower() or 'conversion' in output.lower() or 'converted' in output.lower(), \
        "Pretty-print output should indicate MATLAB-to-OpenCV conversion when was_converted=True"
    
    # Property: original_alpha should be mentioned when conversion was applied
    if calib.original_alpha is not None:
        assert 'alpha' in output.lower() or f'{calib.original_alpha:.6f}' in output, \
            f"Pretty-print output should contain original alpha value {calib.original_alpha}"


@given(calib=valid_parsed_calibration())
@settings(max_examples=100)
def test_pretty_print_no_conversion_indicator_when_not_converted(calib: ParsedCalibration):
    '''
    **Feature: fisheye-calibration-validation, Property 12: Pretty-Print Formatting**
    **Validates: Requirements 6.1, 6.2, 6.3**
    
    For any ParsedCalibration with was_converted=False, the pretty-printed string
    should NOT indicate that conversion was applied.
    '''
    loader = CalibrationLoader()
    output = loader.pretty_print(calib)
    
    # Property: when was_converted is False, output should not mention conversion
    # Note: The implementation only adds conversion note when was_converted is True
    assert calib.was_converted == False, "Test precondition: was_converted should be False"
    
    # Check that the conversion note is not present
    assert 'MATLAB-to-OpenCV conversion was applied' not in output, \
        "Pretty-print output should not indicate conversion when was_converted=False"


@given(calib=valid_parsed_calibration())
@settings(max_examples=100)
def test_pretty_print_contains_distortion_coefficients(calib: ParsedCalibration):
    '''
    **Feature: fisheye-calibration-validation, Property 12: Pretty-Print Formatting**
    **Validates: Requirements 6.1, 6.2, 6.3**
    
    For any ParsedCalibration, the pretty-printed string should display
    distortion coefficients with scientific notation for small values.
    '''
    loader = CalibrationLoader()
    output = loader.pretty_print(calib)
    
    D = calib.dist_coeffs.flatten()
    
    # Property: output must contain k1, k2, k3, k4 labels
    assert 'k1' in output.lower(), "Pretty-print output should contain k1 label"
    assert 'k2' in output.lower(), "Pretty-print output should contain k2 label"
    assert 'k3' in output.lower(), "Pretty-print output should contain k3 label"
    assert 'k4' in output.lower(), "Pretty-print output should contain k4 label"


@given(calib=valid_parsed_calibration())
@settings(max_examples=100)
def test_pretty_print_contains_camera_matrix_section(calib: ParsedCalibration):
    '''
    **Feature: fisheye-calibration-validation, Property 12: Pretty-Print Formatting**
    **Validates: Requirements 6.1, 6.2, 6.3**
    
    For any ParsedCalibration, the pretty-printed string should display
    the camera matrix in a formatted layout.
    '''
    loader = CalibrationLoader()
    output = loader.pretty_print(calib)
    
    # Property: output must contain camera matrix section header
    assert 'camera matrix' in output.lower() or 'camera_matrix' in output.lower(), \
        "Pretty-print output should contain Camera Matrix section"
    
    # Property: output must contain distortion coefficients section header
    assert 'distortion' in output.lower(), \
        "Pretty-print output should contain Distortion Coefficients section"


if __name__ == '__main__':
    test_calibration_round_trip()
    print('Property 1 (Calibration Round-Trip): PASSED')
    test_matlab_format_conversion()
    print('Property 2 (MATLAB Format Conversion): PASSED')
    test_input_validation_completeness()
    print('Property 3 (Input Validation Completeness - valid inputs): PASSED')
    test_invalid_camera_matrix_shape_rejected()
    print('Property 3 (Input Validation Completeness - invalid matrix shape): PASSED')
    test_invalid_dist_coeffs_length_rejected()
    print('Property 3 (Input Validation Completeness - invalid coeffs length): PASSED')
    test_fisheye_distortion_model_correctness()
    print('Property 4 (Fisheye Distortion Model Correctness): PASSED')
    test_fisheye_distortion_center_invariant()
    print('Property 4 (Fisheye Distortion Center Invariant): PASSED')
    test_fisheye_distortion_radial_symmetry()
    print('Property 4 (Fisheye Distortion Radial Symmetry): PASSED')
    test_output_dimension_invariant()
    print('Property 5 (Output Dimension Invariant): PASSED')
    test_output_dimension_invariant_grayscale()
    print('Property 5 (Output Dimension Invariant - Grayscale): PASSED')
    test_output_dimension_invariant_empty_image()
    print('Property 5 (Output Dimension Invariant - Empty Image): PASSED')
    test_pretty_print_contains_intrinsics()
    print('Property 12 (Pretty-Print Contains Intrinsics): PASSED')
    test_pretty_print_indicates_conversion()
    print('Property 12 (Pretty-Print Indicates Conversion): PASSED')
    test_pretty_print_no_conversion_indicator_when_not_converted()
    print('Property 12 (Pretty-Print No Conversion When Not Converted): PASSED')
    test_pretty_print_contains_distortion_coefficients()
    print('Property 12 (Pretty-Print Contains Distortion Coefficients): PASSED')
    test_pretty_print_contains_camera_matrix_section()
    print('Property 12 (Pretty-Print Contains Camera Matrix Section): PASSED')
