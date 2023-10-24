import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Load the data from the text file
data = np.loadtxt("data2.txt", delimiter="\t", skiprows=1)
datac = np.loadtxt("data_cell.txt", delimiter="\t", skiprows=1)
colors =  ["#F72585","#B5179E","#480CA8","#3F37C9","#4262F0","#4CC9F0"]

# Split the data into separate arrays for each time series
blchn = data[:, 0]
time_series2 = (data[:, 1]+data[:, 4])/2 #Press in [Pa]
time_series3 = (data[:, 2]+data[:, 5])/2 #Press out [Pa]

#DP_h = (data[:, 1]+data[:, 3])-(data[:, 2]-data[:, 4])
#DP_l = (data[:, 6]+data[:, 9])-(data[:, 7]-data[:, 10])

DP_h = (data[:, 1])-(data[:, 2])
DP_l = (data[:, 6])-(data[:, 7])


DP_P= (DP_h+DP_l)/2


########Data cell

blchnc = datac[:, 0]
P_ic = datac[:, 1]
P_oc = datac[:, 2]
DP_cell= P_ic-P_oc


#Pin_CLE = data[:, 6]-data[:, 9]
#Pout_CLE = data[:, 7]+data[:, 10]

# Apply a smoothing filter to the data
window_size = 5  # Window size for the filter
poly_degree = 4  # Polynomial degree for the filter
time_series2_smoothed = savgol_filter(time_series2, window_size, poly_degree)
time_series3_smoothed = savgol_filter(time_series3, window_size, poly_degree)


############ Diferencias ##############
DP = time_series2_smoothed-time_series3_smoothed
DP = DP_P
DP_ste= np.abs(DP-data[:, 6])
DP_ste= np.abs(DP_P-DP_h)
################################ NORMALIZAC ##################################
Q = 10*2.778*10**(-10) # 10 ml/hr to m^3/s
mu = 0.001 # Pa*s
wch = 2.5*10**(-6) # um to m

DPN_mean = (DP/(Q*mu*(wch**(-3)))) ## adimensional - factor de perdida de carga
DPN_ste=   (DP_ste/(Q*mu*(wch**(-3))))

DPN_cell = (DP_cell/(Q*mu*(wch**(-3))))

PP = np.array(0.04231610-0.03599830) # 1b
PP = np.append(PP, (0.04231610-0.03587662)) ## 2b
PP = np.append(PP, (0.04231610-0.03565308)) ## 3b
PP = np.append(PP, (0.04231610-0.03537536)) ## 4b
PP = np.append(PP, (0.04231610-0.03500298)) ## 5b
PP = np.append(PP, (0.04231610-0.03451058)) ## 6b
PP = np.append(PP, (0.04231610-0.03380165)) ## 7b
PP = np.append(PP, (0.04231610-0.03276386)) ## 8b
PP = np.append(PP, (0.04231610-0.03094590)) ## 9b
PP = np.append(PP, (0.04231610-0.02735630)) ## 10b
PP = np.append(PP, (0.04231610-0.01647367)) ## 11b
PP = PP*10**6
DPP = (DP/PP)*100
DPP_ste=(DP_ste/PP)*100


PPc = np.array(0.04231610-0.03505822) # 3.7b
PPc = np.append(PPc, (0.04231610-0.03338090)) ## 6.7b
PPc = np.append(PPc, (0.04231610-0.02782096)) ## 9.7b
PPc = PPc*10**6

DPPc = (DP_cell/PPc)*100


# Plot the two time series
#plt.style.use(['science','no-latex'])
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)


ax1.plot(blchn, DPN_mean, c='black', linestyle='dashed', alpha=0.2, linewidth = 2 , label="Mean pressure loss coeficient")
ax1.fill_between(blchn, DPN_mean-DPN_ste,DPN_mean+DPN_ste, alpha=0.1,color='#E65C4F')
ax1.scatter(blchnc, DPN_cell, marker="^", c="grey", edgecolors="black", label="Pressure loss coeficient on channels with cell")

ax1.set_ylabel("$(\Delta P  \cdot w_{ch}^{3})/ (\mu \cdot Q) $")
ax1.yaxis.set_tick_params(which='both', labelleft=True, labelright=False)
ax1.yaxis.set_ticks(np.arange(0, 0.141, 0.02))
#ax1.xaxis.set_ticks(np.arange(0, 13, 1))
ax1.yaxis.grid(True, alpha=0.3, linestyle='--')
ax1.xaxis.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc="upper left",frameon=True)



ax2.plot(blchn, DPP, c='black', linestyle='dashed', alpha=0.2, linewidth = 2 , label="Mean pressure loss")
ax2.fill_between(blchn, DPP-DPP_ste,DPP+DPP_ste, alpha=0.1,color='#E65C4F')

ax2.scatter(blchnc, DPPc, marker="^", c="grey", edgecolors="black", label="Pressure loss on channels with cell")

ax2.set_xlabel("Blocked channels")
ax2.set_ylabel("$\Delta P/(P_{in}-P_{out})$ (%)")
ax2.yaxis.set_tick_params(which='both', labelleft=True, labelright=False)
ax2.yaxis.set_ticks(np.arange(0, 101, 20))
ax2.xaxis.set_ticks(np.arange(0, 13, 1))
ax2.yaxis.grid(True, alpha=0.3,linestyle='--')
ax2.xaxis.grid(True, alpha=0.3,linestyle='--')
ax2.legend(loc="upper left",frameon=True)
#ax1.annotate("(a)", xy=(-0.15, 1.0), weight='bold', xycoords="axes fraction")
#ax2.annotate("(b)", xy=(-0.15, 1.0), weight='bold', xycoords="axes fraction")

plt.show()

# Save the plot to a file
#plt.savefig("time_series.pdf", bbox_inches='tight')
