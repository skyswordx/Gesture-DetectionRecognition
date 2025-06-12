"""
生成项目结构树
"""

import os

def generate_tree(path, prefix="", max_depth=3, current_depth=0):
    """生成目录树"""
    if current_depth > max_depth:
        return ""
    
    items = []
    try:
        # 获取目录内容，排序
        entries = sorted(os.listdir(path))
        
        # 分离文件和目录
        dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not e.startswith('.')]
        files = [e for e in entries if os.path.isfile(os.path.join(path, e)) and not e.startswith('.')]
        
        # 过滤一些不重要的文件和目录
        exclude_dirs = {'__pycache__', '.git', '.vscode', 'node_modules'}
        exclude_files = {'tempCodeRunnerFile.py', '.DS_Store'}
        
        dirs = [d for d in dirs if d not in exclude_dirs]
        files = [f for f in files if f not in exclude_files]
        
        all_items = dirs + files
        
        for i, item in enumerate(all_items):
            item_path = os.path.join(path, item)
            is_last = i == len(all_items) - 1
            
            # 选择分支符号
            if is_last:
                current_prefix = "└── "
                next_prefix = prefix + "    "
            else:
                current_prefix = "├── "
                next_prefix = prefix + "│   "
            
            # 添加图标
            if os.path.isdir(item_path):
                icon = "📁"
                items.append(f"{prefix}{current_prefix}{icon} {item}/")
                
                # 递归处理子目录
                if current_depth < max_depth:
                    subtree = generate_tree(item_path, next_prefix, max_depth, current_depth + 1)
                    if subtree:
                        items.append(subtree)
            else:
                # 根据文件类型选择图标
                if item.endswith('.py'):
                    icon = "🐍"
                elif item.endswith('.md'):
                    icon = "📋"
                elif item.endswith('.txt'):
                    icon = "📄"
                elif item.endswith('.yml') or item.endswith('.yaml'):
                    icon = "⚙️"
                elif item.endswith('.json'):
                    icon = "📊"
                else:
                    icon = "📄"
                
                items.append(f"{prefix}{current_prefix}{icon} {item}")
        
    except PermissionError:
        items.append(f"{prefix}[Permission Denied]")
    
    return "\n".join(items)

def main():
    """主函数"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(project_root)
    
    print("=" * 60)
    print(f"  {project_name} 项目结构")
    print("=" * 60)
    print(f"🏠 {project_name}/")
    
    tree = generate_tree(project_root, max_depth=3)
    if tree:
        # 缩进所有行
        indented_lines = ["   " + line for line in tree.split("\n") if line.strip()]
        print("\n".join(indented_lines))
    
    print("\n" + "=" * 60)
    print("📋 主要文件说明:")
    print("   📋 README_Modular.md - 项目主文档")
    print("   📋 QUICK_START.md - 快速入门指南")
    print("   📄 requirements.txt - 依赖包列表")
    print("   🐍 integrated_demo.py - 综合演示")
    print("   🐍 simple_demo.py - 简单示例")
    print("   🐍 test_basic.py - 基础测试")
    print("=" * 60)

if __name__ == "__main__":
    main()
