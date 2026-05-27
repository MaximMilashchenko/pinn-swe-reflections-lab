import numpy as np

tmax = 9000
xmin = -1000
xmax = 1000

u_data = np.loadtxt("../data/raw/AH=5e+4/udata/u0000.dat")
for i in range(1, tmax):
    u_new_data = np.loadtxt("../data/raw/AH=5e+4/udata/u" + str(i).zfill(4) + ".dat")
    u_data = np.dstack((u_data, u_new_data))

    if int(i/10) == i/10.:
        print("u: " + str(i) + "/9000")

u_new_data = np.loadtxt("../data/raw/AH=5e+4/udata/u" + str(tmax).zfill(4) + ".dat")
u_data = np.dstack((u_data, u_new_data))
u_data = u_data[0].transpose()
np.save("../data/pinn_ready/dt=1s_dx=400m/AH=5e+4/Zonal_Velocity", u_data)

# Load and Plot Data for h
h_data = np.loadtxt("../data/raw/AH=5e+4/zdata/z0000.dat")
for i in range(1, tmax):
    h_new_data = np.loadtxt("../data/raw/AH=5e+4/zdata/z" + str(i).zfill(4) + ".dat")
    h_data = np.dstack((h_data, h_new_data))

    if int(i/10) == i/10.:
        print("h: " + str(i) + "/9000")

h_new_data = np.loadtxt("../data/raw/AH=5e+4/zdata/z" + str(tmax).zfill(4) + ".dat")
h_data = np.dstack((h_data, h_new_data))
h_data = h_data[0].transpose()
np.save("../data/pinn_ready/dt=1s_dx=400m/AH=5e+4/Sea_Level_Elevation", h_data)
