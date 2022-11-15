# a collection of constants which many of the files here use
const G = 6.67e-11 # m²/kg²
const M = 1.989e+35 #kg
const AU = 1.496e11 # meters
const yr = 3.154e7 #seconds
# change units, use AU and years
const GM = Float32(G*M*yr^2 / AU^3) # AU^3/yr^2; Kepler III says this is 4π^2
const c = Float32(3e8 * (yr/AU)) # AU / yr
const r0 = 40.f0 # AU
const v0 = 500.f0 # AU / year
const ricci_r = 2*GM/c^2 # AU