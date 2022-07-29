import math



def flux_to_luminosity(flux_density, distance, redshift):
    """

    :param flux_density: flux density
    :param dl: Luminosity Distance
    :param redshift: Photometric Redshift
    :return:
    """

    dl = distance*3.08567758128E+24
    phot_index = 1-2

    absolute_magnitude = 4*math.pi * (dl**2) * flux_density * ((1 + redshift) ** (phot_index))

    return absolute_magnitude
