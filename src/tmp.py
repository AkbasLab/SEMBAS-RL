import numpy as np

heading = 2.0778202327528685
wp_heading = 2.0776

a = np.array([np.cos(heading), np.sin(heading)])
b = np.array([np.cos(wp_heading), np.sin(wp_heading)])

print(np.acos(a.dot(b)))