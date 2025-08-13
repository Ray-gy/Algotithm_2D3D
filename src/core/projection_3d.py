"""
3D Model Projection and Contour Extraction Module

Implements 3D model projection with oral space constraints for 2D-3D registration
Based on the multimodal data matching algorithm described in the research paper.

Key Features:
- 3D dental model loading and preprocessing
- Camera calibration and projection matrix computation
- Oral space constraint-based projection optimization
- 2D contour extraction from projected 3D models
- Multiple 3D file format support (STL, PLY, OBJ)

Author: Assistant
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
import logging
from dataclasses import dataclass
import yaml

try:
    import trimesh
    import open3d as o3d
except ImportError:
    logging.warning("3D modeling libraries not installed. Run: pip install trimesh open3d")
    trimesh = None
    o3d = None


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters"""
    # Intrinsic parameters
    focal_length: Tuple[float, float]  # (fx, fy)
    principal_point: Tuple[float, float]  # (cx, cy)
    distortion_coeffs: np.ndarray  # Distortion coefficients
    
    # Extrinsic parameters  
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation_vector: np.ndarray  # 3x1 translation vector
    
    # Image dimensions
    image_width: int
    image_height: int


@dataclass 
class OralSpaceConstraints:
    """Oral space geometric constraints for projection optimization"""
    # Anatomical constraints
    tooth_crown_height_range: Tuple[float, float] = (5.0, 15.0)  # mm
    root_length_range: Tuple[float, float] = (8.0, 25.0)  # mm
    jaw_opening_angle_range: Tuple[float, float] = (0.0, 60.0)  # degrees
    
    # Viewing constraints
    microscope_distance_range: Tuple[float, float] = (200.0, 400.0)  # mm
    viewing_angle_range: Tuple[float, float] = (-45.0, 45.0)  # degrees
    
    # Quality constraints
    min_visible_surface_ratio: float = 0.3  # Minimum visible surface percentage
    max_occlusion_ratio: float = 0.7  # Maximum allowed occlusion


class Model3DProjector:
    """3D dental model projector with oral space constraints"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize 3D model projector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.camera_params: Optional[CameraParameters] = None
        self.oral_constraints = OralSpaceConstraints()
        self.current_model: Optional[Any] = None
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            # Default configuration
            return {
                'projection': {
                    'default_focal_length': [800.0, 800.0],
                    'default_principal_point': [320.0, 240.0],
                    'default_image_size': [640, 480],
                    'projection_method': 'perspective',
                    'enable_depth_test': True,
                    'contour_extraction_method': 'canny'
                },
                'model_preprocessing': {
                    'enable_smoothing': True,
                    'smoothing_iterations': 2,
                    'enable_decimation': False,
                    'target_triangle_count': 5000,
                    'remove_isolated_vertices': True
                },
                'oral_constraints': {
                    'enable_constraints': True,
                    'constraint_weight': 0.3,
                    'optimization_method': 'gradient_descent',
                    'max_iterations': 100
                }
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            return self._load_config(None)
    
    def load_3d_model(self, model_path: str) -> bool:
        """Load 3D dental model from file
        
        Args:
            model_path: Path to 3D model file (STL, PLY, OBJ)
            
        Returns:
            True if successful, False otherwise
        """
        if trimesh is None:
            self.logger.error("Trimesh library not available. Cannot load 3D models.")
            return False
            
        try:
            # Load model using trimesh
            self.current_model = trimesh.load(model_path)
            
            if not isinstance(self.current_model, trimesh.Trimesh):
                self.logger.error(f"Loaded model is not a valid mesh: {type(self.current_model)}")
                return False
                
            self.logger.info(f"Loaded 3D model: {len(self.current_model.vertices)} vertices, "
                           f"{len(self.current_model.faces)} faces")
            
            # Preprocess model
            self._preprocess_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load 3D model from {model_path}: {e}")
            return False
    
    def _preprocess_model(self):
        """Preprocess the loaded 3D model"""
        if self.current_model is None:
            return
            
        config = self.config.get('model_preprocessing', {})
        
        try:
            # Remove isolated vertices
            if config.get('remove_isolated_vertices', True):
                self.current_model.remove_unreferenced_vertices()
                
            # Smoothing
            if config.get('enable_smoothing', True):
                iterations = config.get('smoothing_iterations', 2)
                self.current_model = self.current_model.smoothed(iterations=iterations)
                
            # Decimation (reduce triangle count)
            if config.get('enable_decimation', False):
                target_count = config.get('target_triangle_count', 5000)
                if len(self.current_model.faces) > target_count:
                    self.current_model = self.current_model.simplify_quadric_decimation(target_count)
                    
            self.logger.info(f"Preprocessed model: {len(self.current_model.vertices)} vertices, "
                           f"{len(self.current_model.faces)} faces")
                           
        except Exception as e:
            self.logger.warning(f"Model preprocessing failed: {e}")
    
    def set_camera_parameters(self, camera_params: CameraParameters):
        """Set camera parameters for projection
        
        Args:
            camera_params: Camera intrinsic and extrinsic parameters
        """
        self.camera_params = camera_params
        self.logger.info("Camera parameters updated")
    
    def calibrate_camera_from_images(self, calibration_images: List[np.ndarray], 
                                   chessboard_size: Tuple[int, int]) -> bool:
        """Calibrate camera from chessboard calibration images
        
        Args:
            calibration_images: List of calibration images
            chessboard_size: Chessboard pattern size (rows, cols)
            
        Returns:
            True if calibration successful, False otherwise
        """
        if not calibration_images:
            self.logger.error("No calibration images provided")
            return False
            
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane
        
        # Find chessboard corners in each image
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        if len(objpoints) == 0:
            self.logger.error("No chessboard patterns found in calibration images")
            return False
            
        # Camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if not ret:
            self.logger.error("Camera calibration failed")
            return False
            
        # Create camera parameters
        h, w = calibration_images[0].shape[:2]
        self.camera_params = CameraParameters(
            focal_length=(mtx[0, 0], mtx[1, 1]),
            principal_point=(mtx[0, 2], mtx[1, 2]),
            distortion_coeffs=dist,
            rotation_matrix=np.eye(3),  # Default to identity
            translation_vector=np.zeros((3, 1)),  # Default to origin
            image_width=w,
            image_height=h
        )
        
        self.logger.info(f"Camera calibration successful with {len(objpoints)} images")
        return True
    
    def project_model_with_constraints(self, pose_estimate: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Project 3D model to 2D with oral space constraints
        
        Args:
            pose_estimate: Initial 6DOF pose estimate [rx, ry, rz, tx, ty, tz]
            
        Returns:
            Tuple of (projected_image, projection_info)
        """
        if self.current_model is None:
            raise ValueError("No 3D model loaded")
            
        if self.camera_params is None:
            raise ValueError("Camera parameters not set")
            
        # Use default pose if not provided
        if pose_estimate is None:
            pose_estimate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -300.0])  # 30cm in front of camera
            
        # Apply oral space constraints optimization
        if self.config.get('oral_constraints', {}).get('enable_constraints', True):
            optimized_pose = self._optimize_pose_with_constraints(pose_estimate)
        else:
            optimized_pose = pose_estimate
            
        # Project model with optimized pose
        projected_image, projection_info = self._project_model(optimized_pose)
        
        # Add constraint validation info
        constraint_info = self._validate_constraints(optimized_pose)
        projection_info.update(constraint_info)
        
        return projected_image, projection_info
    
    def _optimize_pose_with_constraints(self, initial_pose: np.ndarray) -> np.ndarray:
        """Optimize pose using oral space constraints
        
        Args:
            initial_pose: Initial 6DOF pose [rx, ry, rz, tx, ty, tz]
            
        Returns:
            Optimized pose
        """
        config = self.config.get('oral_constraints', {})
        method = config.get('optimization_method', 'gradient_descent')
        max_iterations = config.get('max_iterations', 100)
        constraint_weight = config.get('constraint_weight', 0.3)
        
        current_pose = initial_pose.copy()
        learning_rate = 0.01
        
        for iteration in range(max_iterations):
            # Compute constraint violations
            violations = self._compute_constraint_violations(current_pose)
            total_violation = np.sum(violations)
            
            if total_violation < 1e-6:  # Converged
                break
                
            # Compute gradients (simplified numerical differentiation)
            gradients = self._compute_pose_gradients(current_pose)
            
            # Update pose
            current_pose -= learning_rate * constraint_weight * gradients
            
            # Apply pose bounds
            current_pose = self._apply_pose_bounds(current_pose)
            
        self.logger.info(f"Pose optimization converged in {iteration + 1} iterations")
        return current_pose
    
    def _compute_constraint_violations(self, pose: np.ndarray) -> np.ndarray:
        """Compute constraint violation values for given pose"""
        violations = []
        
        # Distance constraint
        distance = np.linalg.norm(pose[3:6])  # Translation magnitude
        min_dist, max_dist = self.oral_constraints.microscope_distance_range
        if distance < min_dist:
            violations.append(min_dist - distance)
        elif distance > max_dist:
            violations.append(distance - max_dist)
        else:
            violations.append(0.0)
            
        # Viewing angle constraint
        viewing_angle = np.arctan2(pose[4], pose[5]) * 180 / np.pi  # Angle in XY plane
        min_angle, max_angle = self.oral_constraints.viewing_angle_range
        if viewing_angle < min_angle:
            violations.append(min_angle - viewing_angle)
        elif viewing_angle > max_angle:
            violations.append(viewing_angle - max_angle)
        else:
            violations.append(0.0)
            
        return np.array(violations)
    
    def _compute_pose_gradients(self, pose: np.ndarray) -> np.ndarray:
        """Compute gradients for pose optimization using numerical differentiation"""
        epsilon = 1e-6
        gradients = np.zeros_like(pose)
        
        base_violations = self._compute_constraint_violations(pose)
        base_cost = np.sum(base_violations)
        
        for i in range(len(pose)):
            pose_plus = pose.copy()
            pose_plus[i] += epsilon
            violations_plus = self._compute_constraint_violations(pose_plus)
            cost_plus = np.sum(violations_plus)
            
            gradients[i] = (cost_plus - base_cost) / epsilon
            
        return gradients
    
    def _apply_pose_bounds(self, pose: np.ndarray) -> np.ndarray:
        """Apply bounds to pose parameters"""
        bounded_pose = pose.copy()
        
        # Rotation bounds (-π to π)
        bounded_pose[:3] = np.clip(bounded_pose[:3], -np.pi, np.pi)
        
        # Translation bounds (reasonable range for dental microscopy)
        bounded_pose[3:6] = np.clip(bounded_pose[3:6], -500.0, 500.0)
        
        return bounded_pose
    
    def _project_model(self, pose: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Project 3D model to 2D image using camera parameters
        
        Args:
            pose: 6DOF pose [rx, ry, rz, tx, ty, tz]
            
        Returns:
            Tuple of (projected_image, projection_info)
        """
        # Convert pose to rotation matrix and translation vector
        rvec = pose[:3].reshape((3, 1))
        tvec = pose[3:6].reshape((3, 1))
        
        # Get model vertices
        vertices = self.current_model.vertices.astype(np.float32)
        
        # Create camera matrix
        camera_matrix = np.array([
            [self.camera_params.focal_length[0], 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length[1], self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])
        
        # Project 3D points to 2D
        projected_points, jacobian = cv2.projectPoints(
            vertices, rvec, tvec, camera_matrix, self.camera_params.distortion_coeffs)
        
        projected_points = projected_points.reshape(-1, 2)
        
        # Create projection image
        img_width = self.camera_params.image_width
        img_height = self.camera_params.image_height
        projected_image = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Draw projected points
        for point in projected_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < img_width and 0 <= y < img_height:
                projected_image[y, x] = 255
        
        # Apply morphological operations to connect points
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        projected_image = cv2.morphologyEx(projected_image, cv2.MORPH_CLOSE, kernel)
        
        # Compute projection statistics
        visible_points = np.sum((projected_points[:, 0] >= 0) & 
                              (projected_points[:, 0] < img_width) &
                              (projected_points[:, 1] >= 0) & 
                              (projected_points[:, 1] < img_height))
        
        projection_info = {
            'total_vertices': len(vertices),
            'visible_vertices': visible_points,
            'visibility_ratio': visible_points / len(vertices),
            'projected_bounds': {
                'min_x': float(np.min(projected_points[:, 0])),
                'max_x': float(np.max(projected_points[:, 0])),
                'min_y': float(np.min(projected_points[:, 1])),
                'max_y': float(np.max(projected_points[:, 1]))
            },
            'pose': pose.tolist()
        }
        
        return projected_image, projection_info
    
    def _validate_constraints(self, pose: np.ndarray) -> Dict[str, Any]:
        """Validate oral space constraints for given pose"""
        distance = np.linalg.norm(pose[3:6])
        viewing_angle = np.arctan2(pose[4], pose[5]) * 180 / np.pi
        
        constraints_satisfied = {
            'distance_in_range': (self.oral_constraints.microscope_distance_range[0] <= 
                                distance <= self.oral_constraints.microscope_distance_range[1]),
            'angle_in_range': (self.oral_constraints.viewing_angle_range[0] <= 
                             viewing_angle <= self.oral_constraints.viewing_angle_range[1])
        }
        
        return {
            'constraint_validation': constraints_satisfied,
            'distance': distance,
            'viewing_angle': viewing_angle,
            'all_constraints_satisfied': all(constraints_satisfied.values())
        }
    
    def extract_projected_contours(self, projected_image: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Extract contours from projected 2D image
        
        Args:
            projected_image: Projected 2D image from 3D model
            
        Returns:
            Tuple of (contour_list, contour_info)
        """
        method = self.config.get('projection', {}).get('contour_extraction_method', 'canny')
        
        if method == 'canny':
            return self._extract_contours_canny(projected_image)
        else:
            return self._extract_contours_simple(projected_image)
    
    def _extract_contours_canny(self, image: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Extract contours using Canny edge detection"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 50
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        contour_info = {
            'method': 'canny',
            'total_contours': len(contours),
            'filtered_contours': len(filtered_contours),
            'min_area_threshold': min_area
        }
        
        return filtered_contours, contour_info
    
    def _extract_contours_simple(self, image: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Extract contours using simple thresholding"""
        # Threshold the image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 50
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        contour_info = {
            'method': 'simple',
            'total_contours': len(contours), 
            'filtered_contours': len(filtered_contours),
            'min_area_threshold': min_area
        }
        
        return filtered_contours, contour_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded 3D model"""
        if self.current_model is None:
            return {'status': 'no_model_loaded'}
            
        return {
            'status': 'model_loaded',
            'vertices_count': len(self.current_model.vertices),
            'faces_count': len(self.current_model.faces),
            'bounding_box': self.current_model.bounds.tolist(),
            'volume': float(self.current_model.volume),
            'surface_area': float(self.current_model.area),
            'is_watertight': bool(self.current_model.is_watertight),
            'is_winding_consistent': bool(self.current_model.is_winding_consistent)
        }


def create_default_camera_params(image_width: int = 640, image_height: int = 480) -> CameraParameters:
    """Create default camera parameters for testing"""
    return CameraParameters(
        focal_length=(800.0, 800.0),
        principal_point=(image_width / 2, image_height / 2),
        distortion_coeffs=np.zeros((4, 1)),
        rotation_matrix=np.eye(3),
        translation_vector=np.zeros((3, 1)),
        image_width=image_width,
        image_height=image_height
    )