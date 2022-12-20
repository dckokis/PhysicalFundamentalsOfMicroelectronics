import fompy.constants
import fompy.functions
import fompy.materials
import fompy.models
import fompy.units
import fompy.util
import numpy as np
import matplotlib.pyplot as plt



class Constants:
    # константы в СИ
    h = 6.6*1e-34 #СИ без черты
    h_tab = h/2/np.pi
    k = 1.38*1e-23
    m0 = 9.1*1e-31
    eps0 = 8.87*1e-12
    e = 1.6*1e-19

def convert_Joule_To_eV(J):
    return J*6.2*1e18

def get_N(m, T):
    # m - кг
    root = 2*np.pi*m*fompy.constants.k*T/(Constants.h)**2
    return 2*np.power(root, 3./2.)

def get_n(m, Ec, Ef, T):
    N = get_N(m, T)
    return N*np.exp(- (Ec - Ef)/(convert_Joule_To_eV(Constants.k*T)))

def get_p(m, Ef, Ev, T):
    N = get_N(m, T)
    return N*np.exp(-(Ef - Ev)/convert_Joule_To_eV(Constants.k*T))

def get_Nd_ion(Nd, Ef, Ed, T):
    return Nd/(1 + 0.5*np.exp((Ef - Ed)/convert_Joule_To_eV(Constants.k*T)))

def get_E0(me, mh, T, Ec, Ev):
    a = (Ec + Ev)/2. + 3./4.* convert_Joule_To_eV(fompy.constants.k*T)*np.log(me/mh)
    return a

def get_E1(Ec):
    return Ec

# ширины области обеднения
# в СИ
def get_W(Na, Nd, eps, delta_Ef):
    return np.sqrt(eps*Constants.eps0*delta_Ef/Constants.e*(Nd + Na)/Nd/Na)

# энергия связи в водородноподобной модели
# eV
# вроде считает не говно
def get_E_connective(m_eff, eps):
    return 1/eps**2*m_eff/Constants.m0*13.6

def convert_charge_to_sgs(charge):
    return charge*3*1e9

def erg_to_Ev(erg):
    return erg*6.24*1e11

def get_Wn(Na, Nd, eps, delta_Ef):
    return np.sqrt(2*eps*Constants.eps0*delta_Ef*Na/(fompy.constants.e*Nd*(Na + Nd)))

def get_Wp(Na, Nd, eps, delta_Ef):
    return np.sqrt(2*eps*Constants.eps0*delta_Ef*Nd/(fompy.constants.e*Na*(Na + Nd)))

def get_ni(Nc, Nv, Eg, T):
    return np.sqrt(Nc*Nv)*np.exp(-Eg/(2*convert_Joule_To_eV(fompy.constants.k*T)))

def get_pn0(Nd, ni):
    return ni**2/Nd

def get_np0(Na, ni):
    return ni**2/Na

def get_J0(Dp, pn0, alpha_p, Dn, np0, alpha_n):
    return Constants.e*(Dp*pn0/alpha_p + Dn*np0/alpha_n)

def get_D(diff_length, tau):
    return diff_length**2/tau

# Функции для 4 задачи

def get_Id(Ud, Ug, Vp, Ip):
    return Ip*(3*Ud/Vp-2*np.power((Ud + Ug)/Vp, 3./2.) - np.power(Ug/Vp, 3./2.))

def get_Vp(Nd, a, eps):
    return Constants.e*Nd*a**2/(2*eps*Constants.eps0)

def get_Ip(Nd, z, mobility, a, eps, L):
    return z*mobility*Constants.e**2*Nd**2*a**3/(6*eps*Constants.eps0*L)



def get_VAC_data(Vg, Vp, Ip):
    Ud = np.linspace(0,Vp, 1000)
    Id_arr = [get_Id(u, Vg, Vp, Ip) for u in Ud]

    return Ud, Id_arr

def plot_VAC_data(Ud, Id_arr):
    max_id = np.argmax(Id_arr)
    plt.grid()
    plt.plot(Ud[:max_id], Id_arr[:max_id])
    plt.xlabel("V")
    plt.ylabel("I")
    plt.show()

def plot_VAC(Vg, Vp, Ip):

    Ud, Id_arr = get_VAC_data(Vg, Vp, Ip)
    plot_VAC_data(Ud, Id_arr)

def flush_Id_arr(Id_arr):
    max = np.max(Id_arr)
    max_id = np.argmax(Id_arr)

    tmp =  np.ones_like(Id_arr)*max
    Id_arr[max_id:] =tmp[max_id:]


def plot_VAC_together(Ud, Id_arr_1, label1, Id_arr_2, label2):


    flush_Id_arr(Id_arr_1)
    flush_Id_arr(Id_arr_2)


    plt.grid()
    plt.plot(Ud, Id_arr_1, label = label1)
    plt.plot(Ud, Id_arr_2, label = label2)
    plt.plot()
    plt.xlabel("V")
    plt.ylabel("I")
    plt.legend()
    plt.show()