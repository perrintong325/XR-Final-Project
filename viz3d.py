import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
# plt.style.use('ggplot')
print(os.getcwd())

def viz(points_3ds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    for points_3d in points_3ds:
        x = points_3d[0,:]
        y = points_3d[1,:]
        z = points_3d[2,:]

        ax.plot(xs = [x[11], x[23]], ys = [y[11], y[23]], zs = [z[11], z[23]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[23], x[25]], ys = [y[23], y[25]], zs = [z[23], z[25]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[25], x[27]], ys = [y[25], y[27]], zs = [z[25], z[27]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[27], x[29]], ys = [y[27], y[29]], zs = [z[27], z[29]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[29], x[31]], ys = [y[29], y[31]], zs = [z[29], z[31]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[11], x[12]], ys = [y[11], y[12]], zs = [z[11], z[12]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[12], x[14]], ys = [y[12], y[14]], zs = [z[12], z[14]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[14], x[16]], ys = [y[14], y[16]], zs = [z[14], z[16]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[16], x[18]], ys = [y[16], y[18]], zs = [z[16], z[18]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[18], x[20]], ys = [y[18], y[20]], zs = [z[18], z[20]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[20], x[16]], ys = [y[20], y[16]], zs = [z[20], z[16]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[12], x[24]], ys = [y[12], y[24]], zs = [z[12], z[24]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[24], x[26]], ys = [y[24], y[26]], zs = [z[24], z[26]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[26], x[28]], ys = [y[26], y[28]], zs = [z[26], z[28]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[28], x[30]], ys = [y[28], y[30]], zs = [z[28], z[30]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[30], x[32]], ys = [y[30], y[32]], zs = [z[30], z[32]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[24], x[23]], ys = [y[24], y[23]], zs = [z[24], z[23]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[11], x[13]], ys = [y[11], y[13]], zs = [z[11], z[13]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[13], x[15]], ys = [y[13], y[15]], zs = [z[13], z[15]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[15], x[17]], ys = [y[15], y[17]], zs = [z[15], z[17]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[17], x[19]], ys = [y[17], y[19]], zs = [z[17], z[19]], linewidth = 4, c = 'r')
        ax.plot(xs = [x[19], x[15]], ys = [y[19], y[15]], zs = [z[19], z[15]], linewidth = 4, c = 'r')



        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-1, 3)
        ax.set_xlabel('x')
        ax.set_ylim3d(-8, 2)
        ax.set_ylabel('y')
        ax.set_zlim3d(-1.5, 1.5)
        ax.set_zlabel('z')
        plt.pause(0.1)
        ax.cla()


if __name__ == '__main__':
    points_3ds = np.load('points_3d_record.npy')
    # print(points_3ds)
    viz(points_3ds)
