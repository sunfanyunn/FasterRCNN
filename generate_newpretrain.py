import numpy as np
f = '/tmp2/ImageNet-ResNet50.npz'
ff = '/tmp2/ImageNet-ResNet50-TwoStream.npz'

f = dict(np.load(f))
print(len(f))
res = {}
for k, v in f.items():
    if k.startswith('group3'):
        res[k] = v
        continue

    # separate for mr and ct
    res['mr' + k] = v
    res['ct' + k] = v

print(len(res))
np.savez(ff, **res)
