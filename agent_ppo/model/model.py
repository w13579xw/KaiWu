#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Simple MLP policy network for Robot Vacuum.
清扫大作战策略网络。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer

class GlobalAttentionModule(nn.Module):
    """
    注意力机制模块：动态关注 充电桩、NPC、污渍 等实体
    """
    def __init__(self, embed_dim=32):
        super().__init__()
        # 将18D全局特征分为3组，每组6D，分别代表：
        # 1. 自身状态 (步数/电量/已清扫/剩余/X坐标/Z坐标)
        # 2. 局部环境 (四向射线/最近污渍/接近污渍delta)
        # 3. 关键实体 (充电桩距离/NPC距离/NPC警报/低电量警报/探索率/是否新区域)
        self.proj_self = nn.Linear(6, embed_dim)
        self.proj_env = nn.Linear(6, embed_dim)
        self.proj_entity = nn.Linear(6, embed_dim)
        
        # 多头注意力机制 (捕捉自身电量与充电桩距离、NPC距离之间的动态关系)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.flatten = nn.Flatten()
        
    def forward(self, global_state):
        # 切片分组
        g_self = global_state[:, 0:6]
        g_env = global_state[:, 6:12]
        g_entity = global_state[:, 12:18]
        
        # 映射到相同的嵌入维度并增加 Sequence 维度
        t_self = self.proj_self(g_self).unsqueeze(1)     # [batch, 1, embed_dim]
        t_env = self.proj_env(g_env).unsqueeze(1)        # [batch, 1, embed_dim]
        t_entity = self.proj_entity(g_entity).unsqueeze(1) # [batch, 1, embed_dim]
        
        # 拼接为序列 (Sequence Length = 3)
        seq = torch.cat([t_self, t_env, t_entity], dim=1)  # [batch, 3, embed_dim]
        
        # 自注意力交互
        attn_out, _ = self.attn(seq, seq, seq)
        
        return self.flatten(attn_out) # 输出维度: 3 * embed_dim = 96

class FeatureExtractor(nn.Module):
    """
    独立特征提取器：包含 CNN 分支与 Attention 分支
    """
    def __init__(self):
        super().__init__()
        # [修改此处] in_channels=3，让 CNN 能够同时看懂三层地图
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            _make_fc(32 * 6 * 6, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 注意力分支 (处理 18D 全局状态)
        self.attn_net = GlobalAttentionModule(embed_dim=32)
        
        # 双流特征融合
        self.fusion = nn.Sequential(
            _make_fc(256 + 96, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
    def forward(self, local_view, global_state):
        cnn_feat = self.cnn(local_view)
        attn_feat = self.attn_net(global_state)
        fused = torch.cat([cnn_feat, attn_feat], dim=1)
        return self.fusion(fused)

class RNDNetwork(nn.Module):
    """
    RND (Random Network Distillation) 好奇心网络
    """
    def __init__(self):
        super().__init__()
        # Target 网络 (参数固定，不参与反向传播)
        self.target = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16 * 11 * 11, 128)
        )
        for param in self.target.parameters():
            param.requires_grad = False
            
        # Predictor 网络 (不断训练以逼近 Target 的输出)
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16 * 11 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
    def forward(self, next_local_view):
        target_feat = self.target(next_local_view)
        pred_feat = self.predictor(next_local_view)
        return target_feat, pred_feat


class Model(nn.Module):
    """
    清扫大作战顶配策略网络：独立 Actor-Critic + Attention + RND + 多头价值
    """
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device
        act_num = Config.ACTION_NUM  # 8

        # ==========================================
        # 1. 独立 Actor-Critic 骨干网络 (分离策略和价值梯度)
        # ==========================================
        self.actor_extractor = FeatureExtractor()
        self.critic_extractor = FeatureExtractor()

        # ==========================================
        # 2. 动作策略头
        # ==========================================
        self.actor_head = _make_fc(256, act_num, gain=0.01)

        # ==========================================
        # 3. 多头价值分解网络 (Multi-head Critic)
        # ==========================================
        self.critic_clean = _make_fc(256, 1, gain=1.0)  # 评估清扫价值
        self.critic_charge = _make_fc(256, 1, gain=1.0) # 评估充电生存价值
        self.critic_safety = _make_fc(256, 1, gain=1.0) # 评估躲避NPC的安全价值

        # ==========================================
        # 4. RND 探索网络
        # ==========================================
        self.rnd = RNDNetwork()

    def forward(self, s, inference=False):
        x = s.to(torch.float32)

        # [修改此处] 切出前 1323 维，并 Reshape 为 (Batch, 3, 21, 21)
        local_view = x[:, :1323].view(-1, 3, 21, 21)
        
        # 切出随后的 18D 全局状态
        global_state = x[:, 1323:-8]

        # --- Actor 前向 ---
        actor_feat = self.actor_extractor(local_view, global_state)
        logits = self.actor_head(actor_feat)

        # --- Critic 前向 (多头价值分解) ---
        critic_feat = self.critic_extractor(local_view, global_state)
        v_clean = self.critic_clean(critic_feat)
        v_charge = self.critic_charge(critic_feat)
        v_safety = self.critic_safety(critic_feat)
        
        # 暂通过求和合并成一个价值，以兼容现有的 PPO 算法接口
        value = v_clean + v_charge + v_safety

        # --- RND 前向 ---
        # 计算好奇心特征
        rnd_target, rnd_pred = self.rnd(local_view)

        # 返回列表扩充 (rst_list[0]是logits, rst_list[1]是value)
        # 将 RND 的结果附带返回，便于后续在 algorithm.py 中计算 Intrinsic Reward 和 Loss
        return [logits, value, rnd_target, rnd_pred]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
