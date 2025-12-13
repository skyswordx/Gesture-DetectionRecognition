# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import cv2
import numpy as np

try:
    from .validation_engine import ValidationResult
except ImportError:
    from validation_engine import ValidationResult


class ResultVisualizer:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1
    FONT_COLOR = (255, 255, 255)
    BG_COLOR = (0, 0, 0)
    
    def show_comparison(self, result: ValidationResult, title: str = "") -> None:
        original = self._ensure_3channel(result.original_image)
        distorted = self._ensure_3channel(result.distorted_image)
        recovered = self._ensure_3channel(result.recovered_image)
        original_labeled = self._add_label(original.copy(), "Original")
        distorted_labeled = self._add_label(distorted.copy(), "Distorted")
        recovered_labeled = self._add_label(recovered.copy(), "Recovered")
        comparison = np.hstack([original_labeled, distorted_labeled, recovered_labeled])
        window_title = title if title else "Fisheye Validation Comparison"
        cv2.imshow(window_title, comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_difference_heatmap(self, result: ValidationResult) -> None:
        heatmap = cv2.applyColorMap(result.difference_map, cv2.COLORMAP_JET)
        heatmap_labeled = self._add_label(heatmap, "Difference Heatmap")
        cv2.imshow("Difference Heatmap", heatmap_labeled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def annotate_metrics(self, image: np.ndarray, result: ValidationResult) -> np.ndarray:
        annotated = image.copy()
        metrics = [
            f"PSNR: {result.psnr:.2f} dB",
            f"SSIM: {result.ssim:.4f}",
            f"Max Error: {result.max_pixel_error:.2f}",
            f"Mean Error: {result.mean_pixel_error:.2f}",
            f"Consistent: {'Yes' if result.is_consistent else 'No'}"
        ]
        y_offset = annotated.shape[0] - 20
        x_offset = 10
        line_height = 25
        for i, metric in enumerate(reversed(metrics)):
            y = y_offset - i * line_height
            (text_width, text_height), _ = cv2.getTextSize(metric, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            cv2.rectangle(annotated, (x_offset - 2, y - text_height - 2), (x_offset + text_width + 2, y + 4), self.BG_COLOR, -1)
            cv2.putText(annotated, metric, (x_offset, y), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
        return annotated
    
    def save_report(self, result: ValidationResult, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        original = self._ensure_3channel(result.original_image)
        distorted = self._ensure_3channel(result.distorted_image)
        recovered = self._ensure_3channel(result.recovered_image)
        original_labeled = self._add_label(original.copy(), "Original")
        distorted_labeled = self._add_label(distorted.copy(), "Distorted")
        recovered_labeled = self._add_label(recovered.copy(), "Recovered")
        comparison = np.hstack([original_labeled, distorted_labeled, recovered_labeled])
        cv2.imwrite(os.path.join(output_dir, "comparison.png"), comparison)
        heatmap = cv2.applyColorMap(result.difference_map, cv2.COLORMAP_JET)
        heatmap_labeled = self._add_label(heatmap, "Difference Heatmap")
        cv2.imwrite(os.path.join(output_dir, "heatmap.png"), heatmap_labeled)
        annotated = self.annotate_metrics(recovered, result)
        cv2.imwrite(os.path.join(output_dir, "annotated.png"), annotated)
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("Fisheye Calibration Validation Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"PSNR: {result.psnr:.4f} dB\n")
            f.write(f"SSIM: {result.ssim:.6f}\n")
            f.write(f"Max Pixel Error: {result.max_pixel_error:.4f}\n")
            f.write(f"Mean Pixel Error: {result.mean_pixel_error:.4f}\n")
            f.write(f"Is Consistent: {result.is_consistent}\n")
    
    def _ensure_3channel(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    
    def _add_label(self, image: np.ndarray, label: str) -> np.ndarray:
        (text_width, text_height), baseline = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
        cv2.rectangle(image, (5, 5), (text_width + 15, text_height + baseline + 10), self.BG_COLOR, -1)
        cv2.putText(image, label, (10, text_height + 8), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
        return image
