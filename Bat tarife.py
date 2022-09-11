from gekko import Gekko
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

par1 = pd.read_excel("C:/Users/GM/Downloads/excel/loads.xlsx")
Loads = np.array(par1["Load"])
print(Loads)
par2 = pd.read_excel("C:/Users/GM/Downloads/excel/PV.xlsx")
Pv_Toplam = np.array(par2["PV_NEW"])

par3 = pd.read_excel("C:/Users/GM/Downloads/excel/Tarifeler.xlsx")
Tarife_1 = np.array(par3["Tarife"])

np.linalg.norm(Pv_Toplam)
m = Gekko(remote=False)

# Tarife_1 = m.Const(0.7)   # Tamamı sabit sayılar
Pv_Sell = m.Const(0.25)  #Pv nin grid e sattığı birim fiyat

##Parametreler

# Loads = m.Param(loads)
# Pv_Toplam = m.Param(pv_toplam)
# print(Loads)

# variables

Grid_use = m.Array(m.Var, (48), value=0, lb=0)
Pv_Grid = m.Array(m.Var, (48), value=0, lb=0)
Pv_Loads = m.Array(m.Var, (48), value=0, lb=0)
SOE = m.Array(m.Var, (48), value=0, lb=0)
total_cost = m.Array(m.Var, (48))
P_dis = m.Array(m.Var, (48), value=0, lb=0)
P_ch = m.Array(m.Var, (48), value=0, lb=0)
DE = 0.85
CE = 0.93
Bat_max = m.Const(5.4)
U = m.Array(m.Var, (48), value=0, lb=0, ub=1, integer=True)
Q = m.Array(m.Var, (48), value=0, lb=0, ub=1, integer=True)
N = 1000
SOE_BAT_min = m.Const(1)

# Pv_Loads = m.if2(Pv_Toplam - Loads,Pv_Loads,Loads)

# Equation
m.Equations(Pv_Loads[t] + Pv_Grid[t] + P_ch[t] == Pv_Toplam[t] for t in range(48))

m.Equations(P_ch[t] <= Bat_max * U[t] for t in range(48))

m.Equations(P_dis[t] <= Bat_max * (1 - U[t]) for t in range(48))

m.Equation(SOE[0] == SOE_BAT_min)

m.Equations(Pv_Loads[t] + Grid_use[t] + P_dis[t] == Loads[t] for t in range(48))

m.Equations(SOE[t] >= SOE_BAT_min for t in range(48))
# m.Equations(Grid_use[t] == m.if2(Pv_Toplam[t] - Loads[t],Grid_use[t],0) for t in range(48))
m.Equations(SOE[t] <= Bat_max for t in range(48))

m.Equations(P_ch[t] <= N * (1 - Q[t]) for t in range(48))

m.Equations(P_dis[t] <= N * (Q[t]) for t in range(48))
# m.Equations(Pv_Grid[t] == m.if2(Pv_Toplam[t] - Loads[t],0,Pv_Grid[t]) for t in range(48))

m.Equations(total_cost[t] == Tarife_1[t] * Grid_use[t] - Pv_Sell * Pv_Grid[t] for t in range(48))
#####Battery
m.Equations(SOE[t] == SOE[t - 1] + P_ch[t] * CE - (P_dis[t] / DE) for t in range(48))

##Objectıve Funct.
# ♠for t in range(96)
# m.Obj(total_cost[t])

m.Minimize(sum(total_cost))

m.options.SOLVER = 1
m.options.MAX_ITER = 1000

m.solve(disp=True)

I = []
x = []
z = []
y = []
for i in range(48):
    I.append(total_cost[i].value)
    x.append(SOE[i].value)
    y.append(P_dis[i].value)
    z.append(P_ch[i].value)
# plt.plot(I)
# plt.plot(x)


plt.plot(y)
plt.plot(z)
plt.xlabel('Time')
plt.ylabel('SOE(kW)')
plt.show()
