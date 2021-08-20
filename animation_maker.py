import matplotlib.image as mgimg
import matplotlib.pyplot as plt
from matplotlib import animation

## Read in graphs

p = 0
myimages = []

fig = plt.figure()
# ax1=fig.add_subplot(1,2,1)

for k in range(0, 512):

  try:
    fname = f"./images/image{k}.png"
        # read in pictures
    img = mgimg.imread(fname)
    imgplot = plt.imshow(img)
    # imgplot = plt.plot(0,0,12,22)
    myimages.append([imgplot])
  except:
    break

## Make animation

# plt.show()

animation = animation.ArtistAnimation(fig, myimages, interval=20, blit=True)

animation.save("animation.mp4", fps = 16)
