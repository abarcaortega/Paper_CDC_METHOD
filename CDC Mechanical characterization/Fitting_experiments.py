#Analyses of aspiration curves for rectangular constrictions
#Aldo, Blanca, Cristina, María, Gustavo. 2022

import os
from statistics import mean
import numpy as np
from numpy.linalg import inv
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math

plt.style.use(['science','no-latex'])

########################################################
# Function for eQLV simulating the viscous entry process
########################################################
def Theoret_aspiration(lambda_vs_Pi_el, max_Pi_el,
                       P_exp, E0, gA, tau_fit):
    gB = 1. - gA
    tau = 1.
    #Relaxation function: gA + gB*math.exp(-t/tau)
    Pi_max = P_exp/E0 #maximum value (constant during the creep process) of the non-dimensional differential pressure
    if Pi_max >= max_Pi_el:
        print('E0 is too low in the computations, since the non-dimensional aspiration pressure exceeds the maximum value in que quasistatic curve')
        input("press any key to continue")
        #quit()
    def Relax_f(t):
            return gA+gB*math.exp(-t/tau)
    
    def Creep_f(t):
            return (1.+gB/gA)-(gB/gA)*math.exp(-t/(tau*(1.+gB/gA)))

    T_max = -tau*(1.+gB/gA)*math.log((gA/gB)*((gB/gA+1.)-max_Pi_el/Pi_max))
    nb_points_creep = 20
    T_increment = T_max/(nb_points_creep-1)
    
    lambda_vec = [] #Vector of the non-dimensional advance of the cell lambda=AL/Wc
    Pi_el_vec = [] #Vector of the non-dimensional differential pressure Pi=DP/E0
    time_T_vec = np.array([]) #Vector of non-dimensional time T


    for k in range(nb_points_creep): #k=0,1,...,nb_points_creep-1
        T_value = k*T_increment
        time_T_vec = np.append(time_T_vec, T_value)
        Pi_el_vec.append(Pi_max*Creep_f(T_value))
        lambda_vec.append(lambda_vs_Pi_el(Pi_el_vec[k]))

    time_t_vec = tau_fit*time_T_vec
    t_vs_lambda_teo = interp1d(lambda_vec, time_t_vec, fill_value='extrapolate')
    lambda_vs_t_teo = interp1d(time_t_vec, lambda_vec, fill_value='extrapolate')

    t_max_teo = max(time_t_vec)
    
    return lambda_vec, lambda_vs_t_teo, t_vs_lambda_teo, t_max_teo


########################################################
# Computing mechnical parameters by fitting
# non-linear least-squares method
# based on a script by G Guinea
########################################################
def fitting_nlls(t_exp_vec, lambda_exp_vec, P_exp, E0_ini, gA_ini, tau_ini,
                 lambda_vs_Pi_el, max_Pi_el, max_lambda_el):
    # This gA_ini not used, because a set of different values are tied as initial value...
    
    y = t_exp_vec
    x = lambda_exp_vec
    numDatos=len(x);
    bestError = 10000000 # best value of the fitting error, initially very large value 
    numIter=50; # Number of iterations in levemberg-Marquardt fitting process
    
    def delta(a,b):
        if a==b:
            g=1
        else:
            g=0
        return g
    
    numParam = 3;
    
    for gA_index in range (0,10):
        
        try:
            gA_ini = gA_index/10.
            #gA_ini = 0.5
            print('gA_ini:'+str(gA_ini)+'\n')
        
            Beta = np.array([E0_ini, gA_ini, tau_ini]);
            epsilonBeta = 1e-5*Beta; #Increments for approx derivatives in the fitting process
            
            levenberg=100.;
            levenberg_factor_mult=2;
            levenberg_factor_div=2;
        
            if Beta[0] <= P_exp/max_Pi_el:
                Beta[0] = 1.0001*P_exp/max_Pi_el
            max_gA_possible = (P_exp/(Beta[0]*max_Pi_el))*0.5
        
            if Beta[1] >= max_gA_possible:
                Beta[1] = max_gA_possible*0.95
            if Beta[1] < 1.e-2:
                Beta[1] = 1.e-2
            if Beta[2] < 1.e-3:
                Beta [2] = 1.e-3
        
            lambda_vec, lambda_vs_t_teo, t_vs_lambda_teo, t_max_teo = Theoret_aspiration(lambda_vs_Pi_el, max_Pi_el, P_exp,
                                                            Beta[0], Beta[1], Beta[2])
            
            R2=0
            y2=0
            R = np.zeros(numDatos)
            for i in range(0,numDatos):
                R[i]=t_vs_lambda_teo(x[i])-y[i];
                R2=R2+R[i]*R[i];
                y2=y2+y[i]*y[i];
                
            #for i in range(0,numDatos):
            #    R[i]=(t_vs_lambda_teo(x[i])-y[i])**2
            #
            #RMSE = math.sqrt((np.sum(R)/len(R)))*(1/(np.max(y)-np.min(y)))
            #print(RMSE)
            prevError=100*R2/y2
            #prevError=100*RMSE

            print('Fitting error: ' + str(prevError) + '; Beta: '+ str(Beta) + '\n')
        
            def transpose(m):
                zip(*m)
                
            J=[[0 for j in range(numParam)] for i in range(numDatos)]
            
            for j in range (numIter):
                for k in range(numParam):
                    epsilonBeta = Beta*1.e-3
                    E01=Beta[0]+delta(0,k)*epsilonBeta[0]
                    gA1=Beta[1]+delta(1,k)*epsilonBeta[1]
                    tau1=Beta[2]+delta(2,k)*epsilonBeta[2]
                    lambda_vec, lambda_vs_t_teo1, t_vs_lambda_teo1, t_max_teo1 = Theoret_aspiration(lambda_vs_Pi_el, max_Pi_el, P_exp,
                                                                                    E01, gA1, tau1)
                    E02=Beta[0]
                    gA2=Beta[1]
                    tau2=Beta[2]
                    lambda_vec, lambda_vs_t_teo2, t_vs_lambda_teo2, t_max_teo2 = Theoret_aspiration(lambda_vs_Pi_el, max_Pi_el, P_exp,
                                                                                    E02, gA2, tau2)
                    for i in range(numDatos):
                        J[i][k]=(t_vs_lambda_teo1(x[i]) - t_vs_lambda_teo2(x[i]))/epsilonBeta[k];
                deltaBeta = np.transpose(np.dot(-inv(np.dot(np.transpose(J),J)+levenberg*np.eye(numParam)),np.dot(np.transpose(J),R)));
        
                for k in range(numParam):
                    Beta[k]=Beta[k]+deltaBeta[k];
                if Beta[0] < P_exp/max_Pi_el:
                    Beta[0] = 1.001*P_exp/max_Pi_el
                max_gA_possible = P_exp/(Beta[0]*max_Pi_el)
                if Beta[1] > max_gA_possible:
                    Beta[1] = max_gA_possible*0.95
                if Beta[1] < 1.e-8:
                    #Beta[1] = 1.e-8
                    Beta[1] = gA1
                if Beta[2] < 1.e-8:
                    #Beta [2] = 1.e-8
                    Beta [2] = tau1
                R2=0
                y2=0
                
                E02=Beta[0]
                gA2=Beta[1]
                tau2=Beta[2]
                lambda_vec, lambda_vs_t_teo2, t_vs_lambda_teo2, t_max_teo2 = Theoret_aspiration(lambda_vs_Pi_el, max_Pi_el,
                                                                                                P_exp, E02, gA2, tau2)
                for i in range(0,numDatos):
                    R[i]=t_vs_lambda_teo2(x[i])-y[i];
                    R2=R2+R[i]*R[i];
                    y2=y2+y[i]*y[i];
                
                #for i in range(0,numDatos):
                #    R[i]=(t_vs_lambda_teo2(x[i])-y[i])**2
##
                #RMSE = math.sqrt((np.sum(R)/len(R)))*(1/(np.max(y)-np.min(y)))
#
                nowError=100*R2/y2;
                #nowError=100*RMSE         
                print(nowError)

                #nowError=100*R2/y2;
                if nowError > prevError and levenberg < 1.e7:
                    levenberg *= levenberg_factor_mult
                    print ('nowError > prevError')
                else:
                    levenberg /= levenberg_factor_div
                    if nowError<bestError:
                        bestBeta = Beta
                        bestlambda_vs_t_teo2 = lambda_vs_t_teo2
                        bestError = nowError

                prevError = nowError

                print('levenberg: ' + str(levenberg) + '; Fitting error: ' + str(nowError) + '; Beta: '+ str(Beta) + '\n')
                ####
                #plt.plot(t_exp_vec, lambda_exp_vec, "-b", label="Experimental curve")
                #plt.scatter(t_exp_vec, lambda_exp_vec)
                #plt.plot(t_exp_vec, lambda_vs_t_teo2(t_exp_vec), "-r", label="Fitted curve, $E_0$: %.3f, $g$: %.3f, $τ$: %.3f, Error: %.3f" % (E02, gA2, tau2, nowError))
                #plt.legend(loc="upper left")
                #plt.xlabel("Time [s]")
                #plt.ylabel("$A_L$/$W_{ch}$")
                #plt.grid(True)
                #plt.xlim([0, max(t_exp_vec)*1.2])
                #plt.ylim([0, max(lambda_exp_vec)*1.2])
                #plt.draw()
                #plt.pause(0.0001)
                #plt.clf()
            
        except:
            pass
            ####
                   

    #print(bestBeta)
    return bestBeta[0], bestBeta[1], bestBeta[2], bestlambda_vs_t_teo2, bestError
    

########################################################
# global data
########################################################
cwd = os.getcwd()
folder = cwd+"\\"+"Results\\"
folderqs = cwd+"\\"+"sim\\"


# Experiment data

num_exp=3
exp_file = "res_exp_"+str(num_exp)

dce = 27.741935483870968 
rce = dce/2
wc = 9.032258064516128 
wch=wc/2
blch = 5 #Number of blocked channels


d = ["2.0","4.0","8.0","16.0"]
rc = ["1.6", "2.0", "2.4", "3.2"] #List of values of Rc/Wch of the curves obtained by numerical analyses
rf = ["0.4", "0.8", "1.2", "1.6", "2.4", "4.0"]
wc = ["0.7", "0.8", "1.0", "1.2"]



#folder = 'D:/TESIS/TESIS EXPERIMENTOS/MICROF/2022-11-29/VIVAS/res/code_gus/'
InputQSFilepreName = folderqs+'rc_' #csv format, with ",". First column: lambda=AL/Wc, second column: Pi=DP/E0. No headings
List_of_QS_curves = ["1.6", "2.0", "2.4", "3.2"] #List of values of Rc/Wch of the curves obtained by numerical analyses
InputEXPFileName = folder+exp_file+'.txt' #csv format, with ",". First column: time (s), second column: lambda=AL/Wc. No headings
OutputFileName = folder+'Outf_'+exp_file+'.csv'
nb_points_QS_curves = 1000 #the numerical curves from Abaqus are discretized in this number of points
#Relaxation function: gA + gB*math.exp(-t/tau) with gA + gB = 1.



#Relation of P_exp to blocked channels, using Q=10ml/hr
#rblqch = [[0,7413.9], [2, 8014.5], [4, 8972.3], [6, 10610.3], [8, 14038.7], [10, 18644.0]]

def rPbch(x):
    # Relation of Pressure to number of blocked channels (x)
    return 0.5056*x**6-15.468*x**5+184.86*x**4-1076.9*x**3+3160.9*x**2-4062.7*x+3519.2

Rc_to_Wch = rce/wch #Ratio cell radius Rc to half of the width of the constriction Wch
P_exp = rPbch(blch) #Actual differential pressure in the experiment (Pa)


########################################################
# Import and compute numerical quasistatic-curve data
########################################################
List_of_QS_curves_float = [float(x) for x in List_of_QS_curves]
nb_integers = np.array([float(x) for x in range(0, nb_points_QS_curves)])
lambda_values = np.zeros((nb_points_QS_curves, len(List_of_QS_curves)))
Pi_values = np.zeros((nb_points_QS_curves, len(List_of_QS_curves)))

for i in range(len(List_of_QS_curves)): #i=0,1,...,List_of_QS_curves-1
    InputQSFileName = InputQSFilepreName + List_of_QS_curves[i] + '.txt'
    with open(InputQSFileName, 'r') as f:
        l_reading = [[float(num) for num in line.split('\t')] for line in f]
    A = np.array(l_reading)
    
    A_lambda_corr = []
    A_lambda_corr.append(A[0,0])
    A_Pi_corr = []
    A_Pi_corr.append(A[0,0])
    k = int(0)
    for j in range(len(A)):
        if A[j,0] > A_lambda_corr[k] and A[j,1] > A_Pi_corr[k]:
            A_lambda_corr.append(A[j,0])
            A_Pi_corr.append(A[j,1])
            k += 1
    
    Pi_vs_lambda_el = interp1d(A_lambda_corr, A_Pi_corr, fill_value='extrapolate')
    max_lambda_el = max(A_lambda_corr)
    lambda_values[:,i] = nb_integers*max_lambda_el/(nb_points_QS_curves-1)
    Pi_values[:,i] = Pi_vs_lambda_el(lambda_values[:,i])

lambda_Pi_matrix = np.zeros((nb_points_QS_curves,2))
for i in range(nb_points_QS_curves):
    interp_lambda = interp1d(List_of_QS_curves_float, lambda_values[i,:], fill_value='extrapolate')
    lambda_Pi_matrix[i,0] = interp_lambda(Rc_to_Wch)
    interp_Pi = interp1d(List_of_QS_curves_float, Pi_values[i,:], fill_value='extrapolate')
    lambda_Pi_matrix[i,1] = interp_Pi(Rc_to_Wch)

Pi_vs_lambda_el = interp1d(lambda_Pi_matrix[:,0], lambda_Pi_matrix[:,1], fill_value='extrapolate')
lambda_vs_Pi_el = interp1d(lambda_Pi_matrix[:,1], lambda_Pi_matrix[:,0], fill_value='extrapolate')
max_Pi_el = max(lambda_Pi_matrix[:,1])
max_lambda_el = max(lambda_Pi_matrix[:,0])

#print(List_of_QS_curves)
plt.figure(1)
for i in range(len(List_of_QS_curves)):
    plt.plot(lambda_values[:,i], Pi_values[:,i], label='Numerical curve - $R_c^*$: %.2f' % (float(List_of_QS_curves[i])))

plt.ylabel("$\Delta P/ E_0$")
plt.xlabel("$A_L$/$W_{ch}$")
plt.plot(lambda_Pi_matrix[:,0], lambda_Pi_matrix[:,1], '--', label='Extrapolated curve - $R_c^*$: %.2f' % (Rc_to_Wch))
plt.legend(loc="upper left")
plt.xlim([0, 3.0])
plt.ylim([0, 0.8])
plt.grid(True, alpha=0.5)
plt.legend(framealpha=1, frameon=True);
plt.show()


########################################################
# Import experiment data
########################################################
with open(InputEXPFileName, 'r') as f:
    l_reading = [[float(num) for num in line.split(' ')] for line in f]
B = np.array(l_reading)
#print(B[:,1])
#plt.plot(B[:,0], B[:,1])
lambda_exp_vec = B[:,1]/wch
t_exp_vec = B[:,0]
max_lambda_exp = max(B[:,1])
max_t_exp = max(B[:,0])




########################################################
# Run of the fitting process
########################################################
E0_ini = P_exp/(Pi_vs_lambda_el(mean(lambda_exp_vec[0:4]))) #We consider instant deformation equal to the average of the ___just the first measurement
gA_ini = 0.05
tau_ini = max_t_exp*1
####
## plt.figure(2)
## plt.plot(t_exp_vec, lambda_exp_vec)
## plt.show()

####
E0, gA, tau, lambda_vs_t_fit, bestError = fitting_nlls(t_exp_vec, lambda_exp_vec, P_exp,
                                            E0_ini, gA_ini, tau_ini,lambda_vs_Pi_el,
                                            max_Pi_el, max_lambda_el)

########################################################
# Export the results
########################################################
OutputFile = open(OutputFileName, 'w')
FirstLine = 'E0 = '+str(E0)+'; gA = '+str(gA)+'; tau = '+str(tau)+'\n'
OutputFile.write(FirstLine)
FirstLine = 'time (s)'+';'+'AL/Wc exp'+';'+'AL/Wc fit'+'\n'
OutputFile.write(FirstLine)
for i in range(len(lambda_exp_vec)):
    NewLine = str(t_exp_vec[i])+';'+str(lambda_exp_vec[i])+';'+str(lambda_vs_t_fit(t_exp_vec[i]))+'\n' #We assume Pi_max constant
    #print(NewLine)
    OutputFile.write(NewLine)
OutputFile.close()

print(E0, gA, tau, bestError)

with open("Visco_properties"+'.txt', 'a') as f:
    #save.append(pres_e[jj], item)
    f.write("%i %s %s %s %s\n" % (num_exp, E0, gA, tau, bestError))

plt.figure(3)

plt.plot(t_exp_vec, lambda_exp_vec, "-b", alpha=0.5, label="Experimental curve")
plt.scatter(t_exp_vec, lambda_exp_vec, alpha=0.5)
plt.plot(t_exp_vec, lambda_vs_t_fit(t_exp_vec), "-r", label="Fitted curve, $E_0$: %.2f [Pa], $g_\infty$: %.3f, $τ_C$: %.3f [s], Error: %.3f %%" % (E0, gA, tau, bestError))
#plt.legend(loc="upper left")
plt.xlabel("Time [s]")
plt.ylabel("$A_L$/$W_{ch}$")
plt.grid(True, alpha=0.5)

plt.xlim([0, max(t_exp_vec)*1.2])
plt.ylim([0, max(lambda_exp_vec)*1.2])


plt.legend(framealpha=1, frameon=True)
plt.xlabel("Time [s]")
plt.ylabel("$A_L$/$W_{ch}$")
plt.show()
