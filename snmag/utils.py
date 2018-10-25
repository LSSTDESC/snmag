
def samplePatchOnSphericalCap(phimax, phimin, thetamax, thetamin,
                              size, rng, degrees=True):
    """
    Uniformly distributes samples on a patch on a sphere between
    phimin, phimax, thetamin, thetamax.

    This function is not equipped to handle wrap-around the ranges of theta
    phi and therefore does not work at the poles.
   
    Parameters
    ----------
    phimin: float, mandatory, radians
    phimax: float, mandatory, radians
    thetamin:float, mandatory, radians
    thetamax: float, mandatory, radians
    size: int, mandatory
            number of samples
    seed : int, optional, defaults to 1
            random Seed used for generating values
    Returns
    -------
    tuple of (phivals, thetavals) where phivals and thetavals are arrays of 
            size size in radians.
    """
    u = rng.uniform(size=size)
    v = rng.uniform(size=size)
    
    #phi = np.radians(phi)
    #theta = np.radians(theta)
    #delta = np.radians(delta)

    phivals = (phimax - phimin) * u + phimin
    phivals = np.where(phivals >= 0., phivals, phivals + 2. * np.pi)

    # use conventions in spherical coordinates
    # theta = np.pi/2.0 - theta
    #thetamax = np.pthetamax)
    #thetamin = np.pi/2.0 - np.radians(thetamin)
    #thetamin = theta - delta

    # if thetamax > np.pi or thetamin < 0. :
    #    raise ValueError('Function not implemented to cover wrap around poles')

    # Cumulative Density Function is cos(thetamin) - cos(theta) / cos(thetamin) - cos(thetamax)
    a = np.cos(thetamin) - np.cos(thetamax)
    thetavals = np.arccos(-v * a + np.cos(thetamin))

    return phivals, thetavals
