import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import imageio
from scipy.ndimage import gaussian_filter

def julia_basin_plot(
    c,
    res=100,
    max_iter=3000,
    xlim=2,
    ylim=2,
    eps=1e-6,
    N_quiver=1,   # set to 1 to plot ALL vectors
    save_path=None,
    show=False
):
    f = lambda x, c: x*x + c

    # --- grid ---
    x = np.linspace(-xlim, xlim, res)
    y = np.linspace(-ylim, ylim, res)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # --- critical orbit + cycle detection (unchanged) ---
    def critical_orbit(c, N=5000, burnin=1000):
        z = 0.0 + 0.0j
        orbit = []
        for i in range(N):
            z = z*z + c
            if i >= burnin:
                orbit.append(z)
        return np.array(orbit)

    def detect_cycle(orbit, tol=1e-8, max_period=20):
        for p in range(1, max_period + 1):
            if np.all(np.abs(orbit[p:] - orbit[:-p]) < tol):
                return p
        return None

    def get_attracting_cycle(c):
        orbit = critical_orbit(c)
        p = detect_cycle(orbit)
        if p is None:
            return None
        return orbit[-p:]

    cycle = get_attracting_cycle(c)
    has_cycle = cycle is not None

    # --- basin arrays ---
    bas0 = np.full(Z.shape,max_iter, dtype=int)
    bas02 = np.full(Z.shape,max_iter, dtype=int)
    basInfty = np.full(Z.shape, max_iter, dtype=int)
    basInfty2 = np.full(Z.shape, max_iter, dtype=int)

    Z20 = Z.copy()
    Z2I = Z.copy()

    # --- iteration loop (your logic) ---
    for i in range(max_iter):

        if has_cycle:
            dist = np.min(np.abs(Z20[..., None] - cycle), axis=-1)
            still_far = dist >= eps
            Z20[still_far] = f(Z20[still_far], c)

            just_escaped0 = (bas0 == max_iter) & (~still_far)
            bas0[just_escaped0] = i
            bas02[just_escaped0] = i + 1 - np.log(np.log(np.abs(Z20[just_escaped0])))/np.log(2)

        basInftyradius = np.abs(Z2I) <= 100 * max_iter
        Z2I[basInftyradius] = f(Z2I[basInftyradius], c)

        just_escapedI = (basInfty == max_iter) & (~basInftyradius)
        basInfty[just_escapedI] = i
        basInfty2[just_escapedI] = i + 1 - np.log(np.log(np.abs(Z2I[just_escapedI])))/np.log(2)

    # --- post-processing ---
    bas0Matrix = bas0.copy()
    bas0Matrix2 = bas02.copy()
    basInftyMatrix = basInfty.copy()
    basInftyMatrix[basInftyMatrix == max_iter] = 0
    basInftyMatrix2 = basInfty2.copy()
    basInftyMatrix2[basInftyMatrix2 == max_iter] = 0

    # --- gradients ---
    U0 = bas0Matrix.astype(float)
    dU0y, dU0x = np.gradient(U0)

    UI = basInftyMatrix.astype(float)
    dUIy, dUIx = np.gradient(UI)

    ### weighted distribution function ###
    k = 1.0 # sharpness
    T = np.percentile(basInftyMatrix2[basInftyMatrix2 > 0], 20)

    chi_inf = 1 / (1 + np.exp(-k*(basInftyMatrix2 - T)))
    chi_0 = 1 - chi_inf

    #basin of infinity
    Uinf = gaussian_filter(basInftyMatrix2.astype(float), sigma=1.5*res/40)
    Uy_inf, Ux_inf = np.gradient(Uinf)
    Fx_inf = Ux_inf
    Fy_inf = Uy_inf

    #basin of 0
    U0 = gaussian_filter(bas0Matrix.astype(float), sigma=1.5*res/40)
    Uy_0, Ux_0 = np.gradient(U0)
    Fx_0 = 0.1*Ux_0 # 0.1*
    Fy_0 = 0.1*Uy_0 # 0.1*

    #scaling to combat higher res images
    dx = 2 * xlim / (res - 1)
    dy = 2 * ylim / (res - 1)

    Fx_inf /= dx
    Fy_inf /= dy
    Fx_0   /= dx
    Fy_0   /= dy
    
    # blend
    FIx = chi_inf * Fx_inf + chi_0 * Fx_0
    FIy = chi_inf * Fy_inf + chi_0 * Fy_0

    pos = np.array([50.0,50.0])   # (x, y) initial particle position in array coordinates
    vel = np.array([0.0, 0.0])   # initial velocity
    m = 1.0
    dt = 0.5

    def basInftyforce(pos, FIx, FIy):
        x, y = pos
        
        ny, nx = FIx.shape

        i0 = int(np.floor(y))
        j0 = int(np.floor(x))

        i0 = np.clip(i0, 0, ny - 2)
        j0 = np.clip(j0, 0, nx - 2)

        i1 = i0 + 1
        j1 = j0 + 1

        wx = x - j0
        wy = y - i0

        fx = (
            (1-wx)*(1-wy)*FIx[i0, j0] +
            wx*(1-wy)*FIx[i0, j1] +
            (1-wx)*wy*FIx[i1, j0] +
            wx*wy*FIx[i1, j1]
        )

        fy = (
            (1-wx)*(1-wy)*FIy[i0, j0] +
            wx*(1-wy)*FIy[i0, j1] +
            (1-wx)*wy*FIy[i1, j0] +
            wx*wy*FIy[i1, j1]
        )

        return np.array([fx, fy])


    basInftytrajectory = []
    
    ny, nx = FIx.shape
    for step in range(9*res):
        F = basInftyforce(pos, FIx, FIy)
        vel += (F / m) * dt
        pos += vel * dt
        pos[0] = np.clip(pos[0], 0, nx - 1.001)
        pos[1] = np.clip(pos[1], 0, ny - 1.001)
        basInftytrajectory.append(pos.copy())
        
    Itraj = np.array(basInftytrajectory)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    imI = ax.imshow(
        np.ma.masked_where(basInftyMatrix == 0, basInftyMatrix),
        cmap=plt.cm.autumn,
        extent=[-xlim, xlim, -ylim, ylim],
        origin="lower"
    )

    im0 = ax.imshow(
        np.ma.masked_where(basInftyMatrix != 0, bas0Matrix),
        cmap=plt.cm.Blues_r,
        extent=[-xlim, xlim, -ylim, ylim],
        origin="lower"
    )

    # --- vector fields (FIXED COORDINATES) ---
    Xq, Yq = X, Y
    
    # Basin ∞ vectors
    maskI = (basInftyMatrix == 0)
    ax.quiver(
        Xq[::N_quiver, ::N_quiver],
        Yq[::N_quiver, ::N_quiver],
        np.ma.masked_where(maskI, dUIx)[::N_quiver, ::N_quiver],
        np.ma.masked_where(maskI, dUIy)[::N_quiver, ::N_quiver],
        color="red",
        scale=250,
        headwidth=6,
        headlength=6
    ) #red vecs
    
    # Basin 0 vectors
    mask0 = (basInftyMatrix != 0)
    ax.quiver(
        Xq[::N_quiver, ::N_quiver],
        Yq[::N_quiver, ::N_quiver],
        np.ma.masked_where(mask0, dU0x)[::N_quiver, ::N_quiver],
        np.ma.masked_where(mask0, dU0y)[::N_quiver, ::N_quiver],
        color="blue",
        scale=650,
        headwidth=6,
        headlength=6
    ) # blue vecs

     #trjectory plot
     
    traj_x = x[Itraj[:,0].astype(int)]
    traj_y = y[Itraj[:,1].astype(int)]

    ax.plot(traj_x, traj_y, color='black', linewidth=1)
    plt.scatter(traj_x[0], traj_y[0], c='white', lw=5)
    plt.scatter(traj_x[-1], traj_y[-1], c='black', lw=5)
    
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_title(f"c = {c}")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()


def julia_gif(
    output_dir,
    gif_name="julia_rotation.gif",
    n_frames=60,
    radius=0.7885,
    res=400
):
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    angles = np.linspace(0, 2*np.pi, n_frames, endpoint=False)

      
    for k, a in enumerate(angles):
        c = radius * np.exp(1j * a)
        fname = os.path.join(output_dir, f"frame_{k:03d}.png")

        julia_basin_plot(
            c=c,
            res=res,
            save_path=fname
        )

        frames.append(imageio.imread(fname))

    gif_path = os.path.join(output_dir, gif_name)
    imageio.mimsave(gif_path, frames, duration=0.1)

    print(f"Saved GIF to: {gif_path}")
    
output_dir = "/Users/g_dog/Desktop/PHYS 360 Code/FractalGravity/giftest"

julia_gif(
    output_dir=output_dir,
    n_frames=20,
    res= 6-
)
