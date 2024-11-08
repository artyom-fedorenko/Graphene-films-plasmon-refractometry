import matplotlib.pyplot as plt
import pandas as pd


pic = pd.read_csv(r"C:\Users\belco\Documents\Graphene\Graphene-films-plasmon-refractometry\Data\Data_many_layers\2024.06.04_SPR+microscopy on InSb+Gr-6 layaer_197 um\02_FEL beam_InSb+Gr-up_InSb-down-along lines_gap-0 steps_IMO-9_0001.ascii.csv", sep=';', header=None)
plt.figure(figsize=(10, 10), dpi=1000)
plt.imshow(pic, vmin = 200)
coord = plt.ginput(1)
print(coord)
print(pic)

plt.show()