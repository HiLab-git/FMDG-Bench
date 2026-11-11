import os
import ast
import sys
from pathlib import Path

def analyze_python_file(file_path):
    """分析单个Python文件，提取其中的类和函数定义"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        classes = []
        functions = []
        
        # 使用简单的遍历方法
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 获取类的方法
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        is_async = 'async ' if isinstance(item, ast.AsyncFunctionDef) else ''
                        methods.append(f"{is_async}{item.name}()")
                
                classes.append({
                    'name': node.name,
                    'methods': methods
                })
        
        # 单独获取顶级函数
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_async = 'async ' if isinstance(node, ast.AsyncFunctionDef) else ''
                functions.append(f"{is_async}{node.name}()")
        
        return classes, functions
        
    except (UnicodeDecodeError, SyntaxError) as e:
        print(f"  [!] 解析文件失败 {file_path}: {e}")
        return [], []
    except Exception as e:
        print(f"  [!] 处理文件时发生错误 {file_path}: {e}")
        return [], []

def print_structure(prefix, filename, classes, functions, is_last):
    """打印文件的结构信息"""
    # 确定前缀符号
    connector = "└── " if is_last else "├── "
    next_prefix = prefix + ("    " if is_last else "│   ")
    
    print(f"{prefix}{connector}{filename}")
    
    if not classes and not functions:
        print(f"{next_prefix}└── (无类或函数)")
        return next_prefix + "    "
    
    all_items = []
    
    # 处理类
    for j, cls in enumerate(classes):
        is_last_class = (j == len(classes) - 1 and not functions)
        class_connector = "└── " if is_last_class else "├── "
        
        # 打印类名
        print(f"{next_prefix}{class_connector}类: {cls['name']}")
        
        # 打印类的方法
        method_prefix = next_prefix + ("    " if is_last_class else "│   ")
        for k, method in enumerate(cls['methods']):
            method_connector = "└── " if k == len(cls['methods']) - 1 else "├── "
            print(f"{method_prefix}{method_connector}{method}")
        
        if cls['methods'] and (j < len(classes) - 1 or functions):
            print(f"{method_prefix}")
    
    # 处理顶级函数
    if functions:
        if classes:
            print(f"{next_prefix}├── 函数:")
        else:
            print(f"{next_prefix}├── 函数:")
        
        func_prefix = next_prefix + ("    " if not classes else "│   ")
        for k, func in enumerate(functions):
            is_last_func = (k == len(functions) - 1)
            func_connector = "└── " if is_last_func else "├── "
            print(f"{func_prefix}{func_connector}{func}")
    
    return next_prefix + "    "

def scan_directory(directory, prefix="", is_last=True, level=0, max_depth=None):
    """递归扫描目录并分析Python文件"""
    if max_depth is not None and level >= max_depth:
        return
    
    # 获取目录下所有条目并排序
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        print(f"{prefix}[!] 无权限访问: {directory}")
        return
    
    # 过滤出我们关心的文件类型
    py_files = []
    dirs = []
    
    for entry in entries:
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            # 跳过一些特殊目录
            skip_dirs = ['__pycache__', '.git', '.idea', '.vscode', 
                        'node_modules', 'venv', 'env', '.pytest_cache',
                        'build', 'dist', '.eggs', '*.egg-info']
            if any(entry.startswith(skip_dir) if '*' in skip_dir else entry == skip_dir 
                  for skip_dir in skip_dirs):
                continue
            dirs.append((entry, full_path))
        elif entry.endswith('.py'):
            py_files.append((entry, full_path))
    
    # 计算总数
    total_items = len(py_files) + len(dirs)
    processed_items = 0
    
    # 处理Python文件
    for i, (filename, filepath) in enumerate(py_files):
        processed_items += 1
        is_last_item = (processed_items == total_items)
        
        # 分析Python文件
        classes, functions = analyze_python_file(filepath)
        
        # 打印文件结构
        next_prefix = print_structure(prefix, filename, classes, functions, is_last_item and not dirs)
        
        # 如果这是最后一个文件但有目录，调整前缀
        if is_last_item and dirs:
            print(next_prefix[:-4] + "│   ")
    
    # 递归处理子目录
    for i, (dirname, dirpath) in enumerate(dirs):
        processed_items += 1
        is_last_dir = (processed_items == total_items)
        
        # 确定前缀符号
        if is_last and is_last_dir:
            connector = "└── "
            next_prefix = prefix + "    "
        elif is_last and not is_last_dir:
            connector = "├── "
            next_prefix = prefix + "│   "
        elif not is_last and is_last_dir:
            connector = "└── "
            next_prefix = prefix + "    "
        else:
            connector = "├── "
            next_prefix = prefix + "│   "
        
        print(f"{prefix}{connector}{dirname}/")
        scan_directory(dirpath, next_prefix, is_last_dir, level + 1, max_depth)

def main():
    """主函数"""
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = "."
    
    # 检查目录是否存在
    if not os.path.exists(target_dir):
        print(f"错误: 目录 '{target_dir}' 不存在")
        return
    
    target_dir = os.path.abspath(target_dir)
    print(f"扫描目录: {target_dir}")
    print(f"{'=' * 60}")
    
    # 解析命令行参数
    max_depth = None
    show_stats = False
    
    for arg in sys.argv[2:]:
        if arg.startswith("--depth="):
            try:
                max_depth = int(arg.split("=")[1])
            except ValueError:
                print("警告: 深度参数应为整数，使用默认深度")
        elif arg == "--stats":
            show_stats = True
    
    # 开始扫描
    file_count = 0
    class_count = 0
    function_count = 0
    
    def count_items(directory, level=0):
        nonlocal file_count, class_count, function_count
        
        if max_depth is not None and level >= max_depth:
            return
            
        try:
            entries = os.listdir(directory)
        except (PermissionError, FileNotFoundError):
            return
            
        for entry in entries:
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                skip_dirs = ['__pycache__', '.git', '.idea', '.vscode',
                           'node_modules', 'venv', 'env', '.pytest_cache',
                           'build', 'dist', '.eggs', '*.egg-info']
                if any(entry.startswith(skip_dir) if '*' in skip_dir else entry == skip_dir 
                      for skip_dir in skip_dirs):
                    continue
                count_items(full_path, level + 1)
            elif entry.endswith('.py'):
                file_count += 1
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_count += 1
                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # 检查是否是顶级函数
                            parent = getattr(node, 'parent', None)
                            if not parent or not isinstance(parent, ast.ClassDef):
                                function_count += 1
                except:
                    pass
    
    if show_stats:
        count_items(target_dir)
    
    scan_directory(target_dir, max_depth=max_depth)
    
    if show_stats:
        print(f"{'=' * 60}")
        print(f"统计信息:")
        print(f"  扫描文件数: {file_count}")
        print(f"  类数量: {class_count}")
        print(f"  函数数量: {function_count}")
    
    print(f"{'=' * 60}")
    print("扫描完成!")

if __name__ == "__main__":
    main()