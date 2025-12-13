"""
3D骨骼点可视化生成器

生成展示不同侧身角度下人体骨骼点的：
1. 3D视图 - 俯视图展示旋转效果
2. 2D投影图 - 模拟相机视角的像素分布
3. 对比图 - 同时展示多个角度的骨骼点变化

用于测试报告，帮助读者理解姿态校正算法的应用场景。
"""

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from modules.distance_estimation.synthetic_data_generator import (
    generate_synthetic_landmarks,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP
)


class Arrow3D(FancyArrowPatch):
    """3D箭头类，用于绘制坐标轴"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def generate_body_skeleton(yaw_angle, distance=2.0, shoulder_width=0.40, hip_width=0.30, torso_height=0.50):
    """
    生成完整的人体骨骼点数据
    
    返回: dict包含各关键点的3D和2D坐标
    """
    result = generate_synthetic_landmarks(
        shoulder_width_3d=shoulder_width,
        distance=distance,
        yaw_angle=yaw_angle,
        hip_width_3d=hip_width,
        torso_height=torso_height
    )
    
    # 提取关键点
    ls_3d = result.landmarks_3d[LEFT_SHOULDER]
    rs_3d = result.landmarks_3d[RIGHT_SHOULDER]
    lh_3d = result.landmarks_3d[LEFT_HIP]
    rh_3d = result.landmarks_3d[RIGHT_HIP]
    
    ls_2d = result.landmarks_2d[LEFT_SHOULDER]
    rs_2d = result.landmarks_2d[RIGHT_SHOULDER]
    lh_2d = result.landmarks_2d[LEFT_HIP]
    rh_2d = result.landmarks_2d[RIGHT_HIP]
    
    return {
        'yaw': yaw_angle,
        'distance': distance,
        '3d': {
            'left_shoulder': (ls_3d.x, ls_3d.y, ls_3d.z),
            'right_shoulder': (rs_3d.x, rs_3d.y, rs_3d.z),
            'left_hip': (lh_3d.x, lh_3d.y, lh_3d.z),
            'right_hip': (rh_3d.x, rh_3d.y, rh_3d.z),
        },
        '2d': {
            'left_shoulder': (ls_2d.x * 1280, ls_2d.y * 720),
            'right_shoulder': (rs_2d.x * 1280, rs_2d.y * 720),
            'left_hip': (lh_2d.x * 1280, lh_2d.y * 720),
            'right_hip': (rh_2d.x * 1280, rh_2d.y * 720),
        },
        'shoulder_width_2d': result.shoulder_width_2d_pixels,
        'shoulder_width_3d': result.shoulder_width_3d,
    }


def create_3d_skeleton_comparison(output_dir):
    """
    图1: 3D骨骼点俯视图对比 - 展示不同旋转角度下的骨骼点分布
    """
    fig = plt.figure(figsize=(16, 6))
    
    angles = [0, 30, 45, 60]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    # 左图: 3D俯视图 (从上方看)
    ax1 = fig.add_subplot(131, projection='3d')
    
    for angle, color in zip(angles, colors):
        skeleton = generate_body_skeleton(angle)
        pts = skeleton['3d']
        
        # 绘制肩膀连线
        ax1.plot([pts['left_shoulder'][0], pts['right_shoulder'][0]],
                 [pts['left_shoulder'][2], pts['right_shoulder'][2]],
                 [pts['left_shoulder'][1], pts['right_shoulder'][1]],
                 'o-', color=color, linewidth=3, markersize=10,
                 label=f'θ={angle}°')
        
        # 绘制髋部连线
        ax1.plot([pts['left_hip'][0], pts['right_hip'][0]],
                 [pts['left_hip'][2], pts['right_hip'][2]],
                 [pts['left_hip'][1], pts['right_hip'][1]],
                 'o-', color=color, linewidth=2, markersize=8, alpha=0.7)
        
        # 绘制躯干连线
        ax1.plot([pts['left_shoulder'][0], pts['left_hip'][0]],
                 [pts['left_shoulder'][2], pts['left_hip'][2]],
                 [pts['left_shoulder'][1], pts['left_hip'][1]],
                 '-', color=color, linewidth=1.5, alpha=0.5)
        ax1.plot([pts['right_shoulder'][0], pts['right_hip'][0]],
                 [pts['right_shoulder'][2], pts['right_hip'][2]],
                 [pts['right_shoulder'][1], pts['right_hip'][1]],
                 '-', color=color, linewidth=1.5, alpha=0.5)
    
    ax1.set_xlabel('X (米)', fontsize=11)
    ax1.set_ylabel('Z (深度)', fontsize=11)
    ax1.set_zlabel('Y (米)', fontsize=11)
    ax1.set_title('3D骨骼点分布 (斜视图)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.view_init(elev=25, azim=-60)
    
    # 中图: 俯视图 (XZ平面)
    ax2 = fig.add_subplot(132)
    
    for angle, color in zip(angles, colors):
        skeleton = generate_body_skeleton(angle)
        pts = skeleton['3d']
        
        # 绘制肩膀
        ax2.plot([pts['left_shoulder'][0], pts['right_shoulder'][0]],
                 [pts['left_shoulder'][2], pts['right_shoulder'][2]],
                 'o-', color=color, linewidth=3, markersize=12,
                 label=f'θ={angle}°')
        
        # 标注左右肩
        if angle == 0:
            ax2.annotate('L', (pts['left_shoulder'][0]-0.03, pts['left_shoulder'][2]+0.02),
                        fontsize=9, color=color)
            ax2.annotate('R', (pts['right_shoulder'][0]+0.01, pts['right_shoulder'][2]+0.02),
                        fontsize=9, color=color)
    
    # 绘制相机位置
    ax2.plot(0, 0, 'k^', markersize=15, label='相机')
    ax2.annotate('相机', (0.02, -0.02), fontsize=10)
    
    # 绘制视线方向
    ax2.arrow(0, 0.05, 0, 0.15, head_width=0.02, head_length=0.02, fc='gray', ec='gray')
    
    ax2.set_xlabel('X (米)', fontsize=11)
    ax2.set_ylabel('Z (深度方向)', fontsize=11)
    ax2.set_title('俯视图 - 身体旋转示意', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-0.35, 0.35)
    ax2.set_ylim(-0.1, 0.35)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 右图: 正视图 (XY平面) - 相机视角
    ax3 = fig.add_subplot(133)
    
    for angle, color in zip(angles, colors):
        skeleton = generate_body_skeleton(angle)
        pts = skeleton['3d']
        
        # 绘制肩膀 (投影到XY平面)
        ax3.plot([pts['left_shoulder'][0], pts['right_shoulder'][0]],
                 [-pts['left_shoulder'][1], -pts['right_shoulder'][1]],
                 'o-', color=color, linewidth=3, markersize=12,
                 label=f'θ={angle}° (宽度={skeleton["shoulder_width_2d"]:.1f}px)')
        
        # 绘制髋部
        ax3.plot([pts['left_hip'][0], pts['right_hip'][0]],
                 [-pts['left_hip'][1], -pts['right_hip'][1]],
                 'o-', color=color, linewidth=2, markersize=8, alpha=0.7)
        
        # 绘制躯干
        ax3.plot([pts['left_shoulder'][0], pts['left_hip'][0]],
                 [-pts['left_shoulder'][1], -pts['left_hip'][1]],
                 '-', color=color, linewidth=1.5, alpha=0.5)
        ax3.plot([pts['right_shoulder'][0], pts['right_hip'][0]],
                 [-pts['right_shoulder'][1], -pts['right_hip'][1]],
                 '-', color=color, linewidth=1.5, alpha=0.5)
    
    ax3.set_xlabel('X (米)', fontsize=11)
    ax3.set_ylabel('Y (米)', fontsize=11)
    ax3.set_title('正视图 - 3D肩宽恒定', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(-0.35, 0.35)
    ax3.set_ylim(-0.6, 0.1)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig_3d_skeleton_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {filepath}")


def create_2d_projection_comparison(output_dir):
    """
    图2: 2D像素投影对比 - 展示相机成像平面上的骨骼点分布
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    angles = [0, 30, 45, 60]
    
    for idx, angle in enumerate(angles):
        skeleton = generate_body_skeleton(angle)
        pts_2d = skeleton['2d']
        
        # 上排: 2D骨骼点在图像中的位置
        ax = axes[0, idx]
        
        # 绘制图像边框
        ax.add_patch(plt.Rectangle((0, 0), 1280, 720, fill=False, 
                                    edgecolor='gray', linewidth=2))
        
        # 绘制骨骼点和连线
        # 肩膀
        ax.plot([pts_2d['left_shoulder'][0], pts_2d['right_shoulder'][0]],
                [pts_2d['left_shoulder'][1], pts_2d['right_shoulder'][1]],
                'o-', color='#e74c3c', linewidth=3, markersize=12)
        
        # 髋部
        ax.plot([pts_2d['left_hip'][0], pts_2d['right_hip'][0]],
                [pts_2d['left_hip'][1], pts_2d['right_hip'][1]],
                'o-', color='#3498db', linewidth=3, markersize=10)
        
        # 躯干连线
        ax.plot([pts_2d['left_shoulder'][0], pts_2d['left_hip'][0]],
                [pts_2d['left_shoulder'][1], pts_2d['left_hip'][1]],
                '-', color='#2ecc71', linewidth=2)
        ax.plot([pts_2d['right_shoulder'][0], pts_2d['right_hip'][0]],
                [pts_2d['right_shoulder'][1], pts_2d['right_hip'][1]],
                '-', color='#2ecc71', linewidth=2)
        
        # 标注肩宽
        shoulder_width = skeleton['shoulder_width_2d']
        mid_y = (pts_2d['left_shoulder'][1] + pts_2d['right_shoulder'][1]) / 2
        ax.annotate('', xy=(pts_2d['right_shoulder'][0], mid_y - 30),
                   xytext=(pts_2d['left_shoulder'][0], mid_y - 30),
                   arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text((pts_2d['left_shoulder'][0] + pts_2d['right_shoulder'][0]) / 2,
                mid_y - 50, f'{shoulder_width:.1f}px',
                ha='center', fontsize=11, color='purple', fontweight='bold')
        
        ax.set_xlim(-50, 1330)
        ax.set_ylim(770, -50)
        ax.set_title(f'偏航角 θ = {angle}°', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (像素)', fontsize=10)
        ax.set_ylabel('Y (像素)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 下排: 肩宽变化柱状图
        ax2 = axes[1, idx]
        
        # 计算校正后宽度
        yaw_rad = math.radians(angle)
        cos_yaw = max(0.1, abs(math.cos(yaw_rad)))
        corrected_width = shoulder_width / cos_yaw
        frontal_width = generate_body_skeleton(0)['shoulder_width_2d']
        
        bars = ax2.bar(['原始2D\n宽度', '校正后\n宽度', '正面\n参考'],
                      [shoulder_width, corrected_width, frontal_width],
                      color=['#e74c3c', '#2ecc71', '#95a5a6'],
                      edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, val in zip(bars, [shoulder_width, corrected_width, frontal_width]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
        
        ax2.set_ylim(0, 130)
        ax2.set_ylabel('宽度 (像素)', fontsize=10)
        ax2.set_title(f'宽度对比 (θ={angle}°)', fontsize=11)
        ax2.axhline(y=frontal_width, color='gray', linestyle='--', alpha=0.5)
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='肩膀'),
        mpatches.Patch(color='#3498db', label='髋部'),
        mpatches.Patch(color='#2ecc71', label='躯干'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
               fontsize=11, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filepath = os.path.join(output_dir, 'fig_2d_projection_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {filepath}")


def create_pose_correction_principle(output_dir):
    """
    图3: 姿态校正原理示意图 - 展示为什么需要校正以及如何校正
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 子图1: 问题说明 - 侧身导致2D宽度缩小
    ax1 = fig.add_subplot(231)
    
    # 正面人体示意
    ax1.add_patch(plt.Rectangle((-0.2, 0), 0.4, 0.6, fill=True, 
                                 facecolor='#3498db', alpha=0.3, edgecolor='#2980b9', linewidth=2))
    ax1.plot([-0.2, 0.2], [0.5, 0.5], 'ro-', markersize=10, linewidth=3, label='肩膀')
    ax1.annotate('W', xy=(0, 0.55), ha='center', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.1, 0.8)
    ax1.set_title('正面视图 (θ=0°)\n2D宽度 = W', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 子图2: 侧身人体示意
    ax2 = fig.add_subplot(232)
    
    # 45度侧身
    angle = 45
    w_projected = 0.4 * math.cos(math.radians(angle))
    ax2.add_patch(plt.Rectangle((-w_projected/2, 0), w_projected, 0.6, fill=True,
                                 facecolor='#e74c3c', alpha=0.3, edgecolor='#c0392b', linewidth=2))
    ax2.plot([-w_projected/2, w_projected/2], [0.5, 0.5], 'ro-', markersize=10, linewidth=3)
    ax2.annotate(f'W×cos({angle}°)', xy=(0, 0.55), ha='center', fontsize=12, fontweight='bold')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.1, 0.8)
    ax2.set_title(f'侧身视图 (θ={angle}°)\n2D宽度 = W×cos(θ)', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # 子图3: 俯视图解释
    ax3 = fig.add_subplot(233)
    
    # 绘制相机
    ax3.plot(0, 0, 'k^', markersize=20)
    ax3.annotate('相机', (0.05, -0.05), fontsize=11)
    
    # 绘制正面肩膀
    ax3.plot([-0.2, 0.2], [1.0, 1.0], 'b-', linewidth=4, label='正面 (θ=0°)')
    ax3.plot([-0.2, 0.2], [1.0, 1.0], 'bo', markersize=10)
    
    # 绘制45度侧身肩膀
    angle_rad = math.radians(45)
    x1 = -0.2 * math.cos(angle_rad)
    z1 = 1.0 - 0.2 * math.sin(angle_rad)
    x2 = 0.2 * math.cos(angle_rad)
    z2 = 1.0 + 0.2 * math.sin(angle_rad)
    ax3.plot([x1, x2], [z1, z2], 'r-', linewidth=4, label='侧身 (θ=45°)')
    ax3.plot([x1, x2], [z1, z2], 'ro', markersize=10)
    
    # 绘制投影线
    ax3.plot([0, -0.2], [0, 1.0], 'b--', alpha=0.3)
    ax3.plot([0, 0.2], [0, 1.0], 'b--', alpha=0.3)
    ax3.plot([0, x1], [0, z1], 'r--', alpha=0.3)
    ax3.plot([0, x2], [0, z2], 'r--', alpha=0.3)
    
    ax3.set_xlabel('X', fontsize=11)
    ax3.set_ylabel('Z (深度)', fontsize=11)
    ax3.set_title('俯视图 - 旋转导致投影缩小', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_xlim(-0.4, 0.4)
    ax3.set_ylim(-0.2, 1.4)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 校正公式
    ax4 = fig.add_subplot(234)
    ax4.text(0.5, 0.85, '姿态校正公式', ha='center', va='center', 
             fontsize=16, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.65, r'$W_{corrected} = \frac{W_{2D}}{|\cos(\theta)|}$', 
             ha='center', va='center', fontsize=20, transform=ax4.transAxes)
    ax4.text(0.5, 0.45, '原理:', ha='center', va='center', 
             fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.5, 0.30, r'由于 $W_{2D} = W_{real} \times \cos(\theta)$', 
             ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.5, 0.18, r'所以 $W_{corrected} = \frac{W_{real} \times \cos(\theta)}{|\cos(\theta)|} = W_{real}$', 
             ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.axis('off')
    
    # 子图5: 校正效果对比
    ax5 = fig.add_subplot(235)
    
    angles_test = [0, 15, 30, 45, 60]
    raw_widths = []
    corrected_widths = []
    frontal = generate_body_skeleton(0)['shoulder_width_2d']
    
    for a in angles_test:
        skeleton = generate_body_skeleton(a)
        raw_widths.append(skeleton['shoulder_width_2d'])
        yaw_rad = math.radians(a)
        cos_yaw = max(0.1, abs(math.cos(yaw_rad)))
        corrected_widths.append(skeleton['shoulder_width_2d'] / cos_yaw)
    
    x = np.arange(len(angles_test))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, raw_widths, width, label='原始2D宽度', color='#e74c3c', alpha=0.8)
    bars2 = ax5.bar(x + width/2, corrected_widths, width, label='校正后宽度', color='#2ecc71', alpha=0.8)
    
    ax5.axhline(y=frontal, color='gray', linestyle='--', linewidth=2, label=f'正面参考 ({frontal:.1f}px)')
    ax5.set_xlabel('偏航角 (度)', fontsize=11)
    ax5.set_ylabel('肩宽 (像素)', fontsize=11)
    ax5.set_title('校正效果对比', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{a}°' for a in angles_test])
    ax5.legend(loc='upper right', fontsize=9)
    ax5.set_ylim(0, 130)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 子图6: 误差对比
    ax6 = fig.add_subplot(236)
    
    raw_errors = [abs(w - frontal) / frontal * 100 for w in raw_widths]
    corrected_errors = [abs(w - frontal) / frontal * 100 for w in corrected_widths]
    
    ax6.plot(angles_test, raw_errors, 'ro-', linewidth=2, markersize=10, label='未校正误差')
    ax6.plot(angles_test, corrected_errors, 'go-', linewidth=2, markersize=10, label='校正后误差')
    ax6.fill_between(angles_test, 0, raw_errors, alpha=0.2, color='red')
    ax6.fill_between(angles_test, 0, corrected_errors, alpha=0.2, color='green')
    
    ax6.set_xlabel('偏航角 (度)', fontsize=11)
    ax6.set_ylabel('测量误差 (%)', fontsize=11)
    ax6.set_title('误差减少效果', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=10)
    ax6.set_xlim(-5, 65)
    ax6.set_ylim(0, 55)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig_pose_correction_principle.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {filepath}")


def create_simulated_camera_view(output_dir):
    """
    图4: 模拟相机视角 - 展示不同角度下的人体骨骼识别效果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    angles = [0, 30, 45, 60, 75, 85]
    
    for idx, angle in enumerate(angles):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        skeleton = generate_body_skeleton(angle)
        pts_2d = skeleton['2d']
        
        # 创建模拟的图像背景 (灰色渐变)
        gradient = np.linspace(0.9, 0.7, 720).reshape(-1, 1)
        gradient = np.tile(gradient, (1, 1280))
        ax.imshow(gradient, cmap='gray', extent=[0, 1280, 720, 0], aspect='auto')
        
        # 绘制人体轮廓 (简化的椭圆)
        center_x = (pts_2d['left_shoulder'][0] + pts_2d['right_shoulder'][0]) / 2
        center_y = (pts_2d['left_shoulder'][1] + pts_2d['left_hip'][1]) / 2
        body_width = skeleton['shoulder_width_2d'] * 1.2
        body_height = abs(pts_2d['left_hip'][1] - pts_2d['left_shoulder'][1]) * 1.5
        
        # 绘制身体轮廓
        body_ellipse = plt.matplotlib.patches.Ellipse(
            (center_x, center_y), body_width, body_height,
            facecolor='#f5d6ba', edgecolor='#d4a574', linewidth=2, alpha=0.6
        )
        ax.add_patch(body_ellipse)
        
        # 绘制头部
        head_y = pts_2d['left_shoulder'][1] - 60
        head = plt.Circle((center_x, head_y), 35, facecolor='#f5d6ba', 
                          edgecolor='#d4a574', linewidth=2, alpha=0.6)
        ax.add_patch(head)
        
        # 绘制骨骼点
        # 肩膀 (红色)
        ax.plot([pts_2d['left_shoulder'][0], pts_2d['right_shoulder'][0]],
                [pts_2d['left_shoulder'][1], pts_2d['right_shoulder'][1]],
                'o-', color='#e74c3c', linewidth=4, markersize=14, 
                markeredgecolor='white', markeredgewidth=2)
        
        # 髋部 (蓝色)
        ax.plot([pts_2d['left_hip'][0], pts_2d['right_hip'][0]],
                [pts_2d['left_hip'][1], pts_2d['right_hip'][1]],
                'o-', color='#3498db', linewidth=4, markersize=12,
                markeredgecolor='white', markeredgewidth=2)
        
        # 躯干连线 (绿色)
        ax.plot([pts_2d['left_shoulder'][0], pts_2d['left_hip'][0]],
                [pts_2d['left_shoulder'][1], pts_2d['left_hip'][1]],
                '-', color='#2ecc71', linewidth=3)
        ax.plot([pts_2d['right_shoulder'][0], pts_2d['right_hip'][0]],
                [pts_2d['right_shoulder'][1], pts_2d['right_hip'][1]],
                '-', color='#2ecc71', linewidth=3)
        
        # 脊柱中线
        spine_top = ((pts_2d['left_shoulder'][0] + pts_2d['right_shoulder'][0]) / 2,
                     pts_2d['left_shoulder'][1])
        spine_bottom = ((pts_2d['left_hip'][0] + pts_2d['right_hip'][0]) / 2,
                        pts_2d['left_hip'][1])
        ax.plot([spine_top[0], spine_bottom[0]], [spine_top[1], spine_bottom[1]],
                '-', color='#9b59b6', linewidth=2, alpha=0.7)
        
        # 标注关键点名称
        ax.annotate('L肩', (pts_2d['left_shoulder'][0]-40, pts_2d['left_shoulder'][1]-15),
                   fontsize=9, color='#e74c3c', fontweight='bold')
        ax.annotate('R肩', (pts_2d['right_shoulder'][0]+5, pts_2d['right_shoulder'][1]-15),
                   fontsize=9, color='#e74c3c', fontweight='bold')
        
        # 显示测量信息
        shoulder_width = skeleton['shoulder_width_2d']
        yaw_rad = math.radians(angle)
        cos_yaw = max(0.1, abs(math.cos(yaw_rad)))
        corrected = shoulder_width / cos_yaw
        
        info_text = f'偏航角: {angle}°\n'
        info_text += f'2D肩宽: {shoulder_width:.1f}px\n'
        info_text += f'校正后: {corrected:.1f}px'
        
        ax.text(50, 50, info_text, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')
        
        # 绘制肩宽标注
        mid_y = pts_2d['left_shoulder'][1] + 30
        ax.annotate('', xy=(pts_2d['right_shoulder'][0], mid_y),
                   xytext=(pts_2d['left_shoulder'][0], mid_y),
                   arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
        ax.text(center_x, mid_y + 15, f'{shoulder_width:.1f}px',
               ha='center', fontsize=10, color='#e74c3c', fontweight='bold')
        
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)
        ax.set_title(f'模拟识别效果 - θ={angle}°', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (像素)', fontsize=10)
        ax.set_ylabel('Y (像素)', fontsize=10)
        
        # 添加置信度指示
        if angle <= 45:
            confidence = '高'
            conf_color = '#2ecc71'
        elif angle <= 70:
            confidence = '中'
            conf_color = '#f39c12'
        else:
            confidence = '低'
            conf_color = '#e74c3c'
        
        ax.text(1230, 50, f'置信度: {confidence}', fontsize=10, fontweight='bold',
               ha='right', color=conf_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig_simulated_camera_view.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {filepath}")


def create_3d_rotation_animation_frames(output_dir):
    """
    图5: 3D旋转序列 - 展示从正面到侧面的连续旋转过程
    """
    fig = plt.figure(figsize=(18, 8))
    
    angles = [0, 15, 30, 45, 60, 75]
    
    for idx, angle in enumerate(angles):
        # 3D视图
        ax = fig.add_subplot(2, 6, idx + 1, projection='3d')
        
        skeleton = generate_body_skeleton(angle)
        pts = skeleton['3d']
        
        # 绘制骨骼
        # 肩膀
        ax.plot([pts['left_shoulder'][0], pts['right_shoulder'][0]],
                [pts['left_shoulder'][2], pts['right_shoulder'][2]],
                [pts['left_shoulder'][1], pts['right_shoulder'][1]],
                'ro-', linewidth=3, markersize=8)
        
        # 髋部
        ax.plot([pts['left_hip'][0], pts['right_hip'][0]],
                [pts['left_hip'][2], pts['right_hip'][2]],
                [pts['left_hip'][1], pts['right_hip'][1]],
                'bo-', linewidth=2, markersize=6)
        
        # 躯干
        ax.plot([pts['left_shoulder'][0], pts['left_hip'][0]],
                [pts['left_shoulder'][2], pts['left_hip'][2]],
                [pts['left_shoulder'][1], pts['left_hip'][1]],
                'g-', linewidth=2)
        ax.plot([pts['right_shoulder'][0], pts['right_hip'][0]],
                [pts['right_shoulder'][2], pts['right_hip'][2]],
                [pts['right_shoulder'][1], pts['right_hip'][1]],
                'g-', linewidth=2)
        
        # 绘制相机方向指示
        ax.quiver(0, -0.3, 0, 0, 0.2, 0, color='gray', arrow_length_ratio=0.3, alpha=0.5)
        
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Z', fontsize=8)
        ax.set_zlabel('Y', fontsize=8)
        ax.set_title(f'θ={angle}°', fontsize=11, fontweight='bold')
        ax.view_init(elev=20, azim=-70)
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.2, 0.3)
        ax.set_zlim(-0.1, 0.6)
        
        # 2D投影视图
        ax2 = fig.add_subplot(2, 6, idx + 7)
        
        pts_2d = skeleton['2d']
        
        # 绘制图像框
        ax2.add_patch(plt.Rectangle((400, 200), 480, 320, fill=False,
                                     edgecolor='gray', linewidth=1))
        
        # 绘制骨骼点
        ax2.plot([pts_2d['left_shoulder'][0], pts_2d['right_shoulder'][0]],
                [pts_2d['left_shoulder'][1], pts_2d['right_shoulder'][1]],
                'ro-', linewidth=3, markersize=10)
        ax2.plot([pts_2d['left_hip'][0], pts_2d['right_hip'][0]],
                [pts_2d['left_hip'][1], pts_2d['right_hip'][1]],
                'bo-', linewidth=2, markersize=8)
        ax2.plot([pts_2d['left_shoulder'][0], pts_2d['left_hip'][0]],
                [pts_2d['left_shoulder'][1], pts_2d['left_hip'][1]],
                'g-', linewidth=2)
        ax2.plot([pts_2d['right_shoulder'][0], pts_2d['right_hip'][0]],
                [pts_2d['right_shoulder'][1], pts_2d['right_hip'][1]],
                'g-', linewidth=2)
        
        # 标注肩宽
        sw = skeleton['shoulder_width_2d']
        ax2.text(640, 280, f'{sw:.0f}px', ha='center', fontsize=10, 
                color='red', fontweight='bold')
        
        ax2.set_xlim(350, 930)
        ax2.set_ylim(550, 150)
        ax2.set_title(f'2D投影 ({sw:.0f}px)', fontsize=10)
        ax2.set_aspect('equal')
        ax2.axis('off')
    
    plt.suptitle('身体旋转过程中的3D骨骼点与2D投影变化', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig_3d_rotation_sequence.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {filepath}")


def create_measurement_scenario_diagram(output_dir):
    """
    图6: 测量场景示意图 - 展示实际应用场景
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    # 场景1: 正面接近
    ax1 = axes[0]
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(0, 5)
    
    # 相机
    ax1.plot(0, 0, 'k^', markersize=20)
    ax1.annotate('相机/无人机', (0, -0.3), ha='center', fontsize=10)
    
    # 人物在不同距离
    distances = [1.5, 2.5, 4.0]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    for dist, color in zip(distances, colors):
        # 人物轮廓
        person = plt.matplotlib.patches.Ellipse((0, dist), 0.4, 0.6, 
                                                 facecolor=color, alpha=0.5)
        ax1.add_patch(person)
        ax1.plot(0, dist, 'o', color=color, markersize=8)
        ax1.annotate(f'{dist}m', (0.3, dist), fontsize=10, color=color)
    
    ax1.set_title('场景1: 正面接近/远离\n(θ=0°, 距离变化)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (米)', fontsize=10)
    ax1.set_ylabel('Z - 距离 (米)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 场景2: 侧身行走
    ax2 = axes[1]
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(0, 4)
    
    # 相机
    ax2.plot(0, 0, 'k^', markersize=20)
    ax2.annotate('相机', (0, -0.2), ha='center', fontsize=10)
    
    # 人物在不同角度
    angles_scene = [0, 30, 60]
    x_positions = [-0.8, 0, 0.8]
    
    for angle, x_pos, color in zip(angles_scene, x_positions, colors):
        # 计算旋转后的椭圆
        width = 0.4 * math.cos(math.radians(angle))
        person = plt.matplotlib.patches.Ellipse((x_pos, 2.5), max(0.1, width), 0.6,
                                                 facecolor=color, alpha=0.5)
        ax2.add_patch(person)
        ax2.annotate(f'θ={angle}°', (x_pos, 3.2), ha='center', fontsize=10, color=color)
    
    ax2.set_title('场景2: 侧身行走\n(恒定距离, 角度变化)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (米)', fontsize=10)
    ax2.set_ylabel('Z - 距离 (米)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 场景3: 综合场景
    ax3 = axes[2]
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(0, 5)
    
    # 相机
    ax3.plot(0, 0, 'k^', markersize=20)
    ax3.annotate('无人机', (0, -0.3), ha='center', fontsize=10)
    
    # 多个人物
    scenarios = [
        (0, 1.5, 0, '#2ecc71', '近距正面'),
        (-0.5, 2.5, 30, '#3498db', '中距侧身'),
        (0.8, 3.5, 45, '#e74c3c', '远距侧身'),
        (-1.0, 4.0, 60, '#9b59b6', '远距大角度'),
    ]
    
    for x, z, angle, color, label in scenarios:
        width = 0.35 * math.cos(math.radians(angle))
        person = plt.matplotlib.patches.Ellipse((x, z), max(0.1, width), 0.5,
                                                 facecolor=color, alpha=0.5)
        ax3.add_patch(person)
        ax3.annotate(label, (x, z + 0.4), ha='center', fontsize=9, color=color)
    
    ax3.set_title('场景3: 综合场景\n(多人、多角度、多距离)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (米)', fontsize=10)
    ax3.set_ylabel('Z - 距离 (米)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig_measurement_scenarios.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {filepath}")


def main():
    """生成所有3D可视化图表"""
    print("\n" + "=" * 70)
    print("  3D骨骼点可视化生成器")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), 'test_report_output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    print("\n[1/6] 生成3D骨骼点对比图...")
    create_3d_skeleton_comparison(output_dir)
    
    print("\n[2/6] 生成2D投影对比图...")
    create_2d_projection_comparison(output_dir)
    
    print("\n[3/6] 生成姿态校正原理图...")
    create_pose_correction_principle(output_dir)
    
    print("\n[4/6] 生成模拟相机视角图...")
    create_simulated_camera_view(output_dir)
    
    print("\n[5/6] 生成3D旋转序列图...")
    create_3d_rotation_animation_frames(output_dir)
    
    print("\n[6/6] 生成测量场景示意图...")
    create_measurement_scenario_diagram(output_dir)
    
    print("\n" + "=" * 70)
    print("  3D可视化图表生成完成!")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  - fig_3d_skeleton_comparison.png    (3D骨骼点俯视图对比)")
    print(f"  - fig_2d_projection_comparison.png  (2D像素投影对比)")
    print(f"  - fig_pose_correction_principle.png (姿态校正原理示意)")
    print(f"  - fig_simulated_camera_view.png     (模拟相机视角)")
    print(f"  - fig_3d_rotation_sequence.png      (3D旋转序列)")
    print(f"  - fig_measurement_scenarios.png     (测量场景示意)")
    print()


if __name__ == "__main__":
    main()
