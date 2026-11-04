import numpy as np
from scipy.integrate import solve_ivp
MU=1.0
def acceleration(q): 
    """Accelerazione kepleriana a(q)= -mu q / |q|^3."""  
    return -MU * q / np.linalg.norm(q) **3
def rhs(t,y):
    """Lato destro del sistema del primo ordine y=(x,y,vx,vy)."""
    return np.concatenate([y[2:], acceleration(y[:2])])
def initial_state(e=0.7):
    """Perielio dell'orbita con semiasse a=1 ed eccentricità e."""
    return np.array([1.0-e,0.0,0.0, np.sqrt((1.0+e)/(1.0-e))])
def energy(y):
    return 0.5*np.dot(y[2:],y[2:])-MU/np.linalg.norm(y[:2])
def angular_momentum(y):
    return y[0]*y[3]-y[1]*y[2]

def euler_step(y,h):
    return y+h*rhs(0.0,y)

def symplectic_euler_step(y,h):
    q,p=y[:2],y[2:]
    p_new=p+h*acceleration(q)
    q_new=q+h*p_new
    return np.concatenate([q_new,p_new])

from scipy.optimize import fsolve
def midpoint_step_fsolve(y,h):
    def F(z):
        return z-y-h*rhs(0.0, 0.5*(y+z))
    z0=y+h*rhs(0.0,y)
    return fsolve(F,z0)

def verlet_step(y,h):
    q,p=y[:2],y[2:]
    p_half=p+0.5*h*acceleration(q)
    q_new=q+h*p_half
    p_new=p_half+0.5*h*acceleration(q_new)
    return np.concatenate([q_new,p_new])

def integrate(step, y0, t_end, n_steps):
    """Applica 'step' n_steps volte con passo h=t_end/n_steps."""
    h=t_end/n_steps
    ys=np.zeros((n_steps+1, len(y0)))
    ys[0]=y0
    for k in range(n_steps):
        ys[k+1]=step(ys[k],h)
    return ys

def reference(y0, t_end):
    """Soluzione di riferimento con DOP853 a tolleranza stretta."""
    sol=solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", rtol=1e-13, atol=1e-13)
    return sol.y[:,-1]