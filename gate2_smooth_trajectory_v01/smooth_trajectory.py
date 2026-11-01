import math

def smoothstep(u):
    u = max(0.0, min(1.0, u))
    return u*u*(3.0 - 2.0*u)

def trajectory(t, offset, stride, lift, rate):
    p = (rate*t + offset) % 1.0

    if p < 0.65:
        u = p / 0.65
        s = smoothstep(u)
        x = -stride/2 + stride*s
        z = -0.163
    else:
        u = (p - 0.65) / 0.35
        s = smoothstep(u)
        x = stride/2 - stride*s
        z = -0.163 + lift*math.sin(math.pi*u)

    return x, z
