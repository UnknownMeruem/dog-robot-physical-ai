import math

def smoothstep(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)

def trajectory(t, offset, stride, lift, rate):
    p = (rate * t + offset) % 1.0

    if p < 0.65:
        # 支持期：X方向は線形
        u = p / 0.65
        x = -stride / 2 + stride * u
        z = -0.163

    else:
        # 遊脚期：X方向はSmoothstep
        u = (p - 0.65) / 0.35
        s = smoothstep(u)
        x = stride / 2 - stride * s
        z = -0.163 + lift * math.sin(math.pi * u)

    return x, z
