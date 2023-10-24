import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Load the data from the text file
data = np.loadtxt("data.txt", delimiter="\t", skiprows=1)
colors =  ["#F72585","#B5179E","#480CA8","#3F37C9","#4262F0","#4CC9F0"]

# Split the data into separate arrays for each time series
blchn = data[:, 0]
time_series2 = (data[:, 1]+data[:, 4])/2 #Press in [Pa]
time_series3 = (data[:, 2]+data[:, 5])/2 #Press out [Pa]

# Apply a smoothing filter to the data
window_size = 5  # Window size for the filter
poly_degree = 4  # Polynomial degree for the filter
time_series2_smoothed = savgol_filter(time_series2, window_size, poly_degree)
time_series3_smoothed = savgol_filter(time_series3, window_size, poly_degree)


############ Diferencias ##############
DP = time_series2_smoothed-time_series3_smoothed
DP_ste= np.abs(DP-data[:, 6])

################################ NORMALIZAC ##################################
Q = 10*2.778*10**(-10) # 10 ml/hr to m^3/s
mu = 0.001 # Pa*s
wch = 2.5*10**(-6) # um to m

DPN_mean = (DP/(Q*mu*(wch**(-3)))) ## adimensional - factor de perdida de carga
DPN_ste=   (DP_ste/(Q*mu*(wch**(-3))))

PP = np.array(0.0423161-0.0362895) # 0b
PP = np.append(PP, (0.04223161-0.0361538)) ## 2b
PP = np.append(PP, (0.04223161-0.0355906)) ## 4b
PP = np.append(PP, (0.04223161-0.0348778)) ## 6b
PP = np.append(PP, (0.04223161-0.0332615)) ## 8b
PP = np.append(PP, (0.04223161-0.0289301)) ## 10b
PP = PP*10**6
DPP = (DP/PP)*100
DPP_ste=(DP_ste/PP)*100



# Plot the two time series
plt.style.use(['science','no-latex'])
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

#ax1.plot(blchn, time_series2 , label="Time Series 1")
ax1.plot(blchn, DPN_mean, c='#326789' , label="Time Series 1")
ax1.plot(blchn, DPN_mean , 'o', c='#326789', alpha=0.7, label="Time Series 1")
ax1.fill_between(blchn, DPN_mean-DPN_ste,DPN_mean+DPN_ste, alpha=0.1,color='#326789')



ax1.set_ylabel("$(\Delta P  \cdot w_{ch}^{3})/ (\mu \cdot Q) $")
ax1.yaxis.set_tick_params(which='both', labelleft=True, labelright=False)
#ax1.yaxis.set_ticks(np.arange(80, 105, 5))
#ax1.xaxis.set_ticks(np.arange(0, 13, 1))
ax1.yaxis.grid(True, alpha=0.3, linestyle='--')
ax1.xaxis.grid(True, alpha=0.3, linestyle='--')

#ax2.plot(blchn,time_series3, label="Time Series 2")
ax2.plot(blchn, DPP, c='#326789' , label="Time Series 1")
ax2.plot(blchn, DPP , 'o', c='#326789', alpha=0.7, label="Time Series 1")
ax2.fill_between(blchn, DPP-DPP_ste,DPP+DPP_ste, alpha=0.1,color='#326789')

ax2.set_xlabel("Blocked channels")
ax2.set_ylabel("$\Delta P/(P_{in}-P_{out})$ (%)")
ax2.yaxis.set_tick_params(which='both', labelleft=True, labelright=False)
#ax2.yaxis.set_ticks(np.arange(0, 11, 2))
#ax2.xaxis.set_ticks(np.arange(0, 13, 2))
ax2.yaxis.grid(True, alpha=0.3,linestyle='--')
ax2.xaxis.grid(True, alpha=0.3,linestyle='--')

#ax1.annotate("(a)", xy=(-0.15, 1.0), weight='bold', xycoords="axes fraction")
#ax2.annotate("(b)", xy=(-0.15, 1.0), weight='bold', xycoords="axes fraction")

plt.show()

# Save the plot to a file
#plt.savefig("time_series.pdf", bbox_inches='tight')
