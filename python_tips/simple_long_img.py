import cv2
import numpy as np
import os
import sys

# trace_id, gt_file, if no gt_file, just use trace_id
folder = sys.argv[1]
save_path = os.path.join(folder, "result.png")

fig_list = os.listdir(folder)
fig_list.sort()

# stitch image
width = 1024
result = []
for f in fig_list:
    if f.startswith('.'):
        continue
    f = os.path.join(folder, f)
    print(f"processing {f}")
    im = cv2.imread(f)
    h, w, _= im.shape
    dim = (width, int(h/w*width)) # width, height
    im = cv2.resize(im, dim, interpolation=cv2.INTER_CUBIC)
    if len(result) <= 0:
        result = im
    else:
        result = np.vstack((result, im))

cv2.imwrite(save_path, result)
cv2.destroyAllWindows()
