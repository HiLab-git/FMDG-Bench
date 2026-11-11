# debug_clip_output.py
import torch
import sys
import os
sys.path.append('.')

def debug_clip_output():
    # 设备设置
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    from train_medclip_D_Fundus import load_clip_model_simple
    import clip
    
    print("测试 PubMedCLIP 输出...")
    try:
        backbone, preprocess = load_clip_model_simple(
            'medclip',
            'ViT-B/32',
            device,
            local_model_dir="./models"
        )
    except Exception as e:
        print(f"PubMedCLIP 加载失败: {e}")
        print("使用标准 CLIP 进行测试...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.float()
        backbone = model.visual
    
    backbone.eval()
    
    # 测试输入
    batch_size = 1
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"\n测试输入形状: {test_input.shape}")
    print(f"输入设备: {test_input.device}")
    print(f"模型设备: {next(backbone.parameters()).device}")
    
    with torch.no_grad():
        # 获取输出
        if hasattr(backbone, 'model'):
            print("\nPubMedCLIP 结构:")
            print(f"Backbone 类型: {type(backbone)}")
            print(f"Backbone.model 类型: {type(backbone.model)}")
            
            # 调用模型
            outputs = backbone.model(test_input)
            print(f"\n模型输出类型: {type(outputs)}")
            
            if hasattr(outputs, 'last_hidden_state'):
                hidden_state = outputs.last_hidden_state
                print(f"✓ 找到 last_hidden_state")
            elif hasattr(outputs, 'hidden_states'):
                hidden_state = outputs.hidden_states[-1]
                print(f"✓ 找到 hidden_states")
            else:
                hidden_state = outputs
                print(f"⚠ 没有找到 last_hidden_state 或 hidden_states，使用直接输出")
            
            print(f"输出形状: {hidden_state.shape}")
            
            # 检查是否是元组
            if isinstance(hidden_state, tuple):
                print(f"输出是元组，长度: {len(hidden_state)}")
                for i, item in enumerate(hidden_state):
                    print(f"  第{i}项形状: {item.shape}")
                hidden_state = hidden_state[0]  # 通常第一个是最后一个隐藏状态
            
        else:
            # 标准 CLIP
            print("\n标准 CLIP 结构:")
            hidden_state = backbone(test_input)
            print(f"输出形状: {hidden_state.shape}")
    
    # 分析 token 数量
    if len(hidden_state.shape) == 3:  # [B, L, C]
        batch_size, num_tokens, channels = hidden_state.shape
        print(f"\nToken 分析:")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of tokens: {num_tokens}")
        print(f"  Channels: {channels}")
        
        # 尝试计算可能的特征图尺寸
        import math
        # 减去 cls token（如果存在）
        if num_tokens > 1:
            # 检查是否可能是正方形特征图
            grid_size = int(math.sqrt(num_tokens - 1))  # 假设有 cls token
            if grid_size * grid_size == num_tokens - 1:
                print(f"  可能是 {grid_size}x{grid_size} 特征图 + cls token")
            else:
                print(f"  Token 数量 {num_tokens} 不是正方形+1，尝试其他可能性...")
                
                # 检查是否是正方形
                grid_size = int(math.sqrt(num_tokens))
                if grid_size * grid_size == num_tokens:
                    print(f"  可能是 {grid_size}x{grid_size} 特征图（无 cls token）")
                else:
                    print(f"  Token 数量 {num_tokens} 不是完全平方数")
                    
                    # 尝试找到最接近的平方数
                    for i in range(1, 20):
                        if i * i >= num_tokens:
                            print(f"  最接近的平方数: {i}x{i} = {i*i}")
                            break
    
    return hidden_state.shape

if __name__ == "__main__":
    shape = debug_clip_output()
    print(f"\n最终输出形状: {shape}")