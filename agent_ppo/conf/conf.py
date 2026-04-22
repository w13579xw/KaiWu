#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
清扫大作战 PPO 配置。
"""


class Config:
    # 特征维度修改为 75D
    # [0]: 441D 局部视野
    # [1]: 18D 全局状态 (原12D + 新增6D充电桩/NPC/轨迹特征)
    # [2]: 8D 合法动作
    FEATURES = [
        3 * 21 * 21,  # [修改此处] 1323维 (3通道的 21x21 图像)
        18,  # 修改此处：从 12 增加到 18
        8,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 directional moves
    # 动作空间：8个方向移动
    ACTION_NUM = 8

    # Single-head value
    # 单头价值
    VALUE_NUM = 1

    # PPO hyperparameters
    # PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 0.5

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
