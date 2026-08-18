20251209
    - Measurement set (AS7341 calibration) in front of the windows.
    - Measurement set 2
        ○ To what degree the colour differences/changes in the three sensors are well captured and match?
        ○ The three are perpendicular
        ○ EVERY 50 READS IS A SEPARATE COLOUR 
    - Measurement set 3, cosine correction
        1. 0-100: -55 (left)
        2. 55-100: 0
        3. 210-300: 55 (right)


# Experiments
## 20250912AS7341calibration
### Aims: 
1. Find the calibration parameters for AS7341
2. How close are the three sensors results. Daylight cloudy sky. 
    BasicCounts
    AGAIN (Gain)
    TINT (Integration Time)
    RawSensorValues(RawSensorValues)

    TINT (Integration time) selection can affect the counter for the sensor results. It means TINT directly determines the Full-Scale Range and saturation. 

#### Calibration equation

Channel-wise calibration (for channels 1–8):

    C_i = g_i * R_i + o_i

    Where:
    - R_i  = raw sensor value (BasicCounts)
    - g_i  = gain for channel i
    - o_i  = offset for channel i
    - C_i  = calibrated output


#### Vectorised form

Vector form for all 8 channels:

    C = R * g  +  o

Where:
- R = vector of 8 raw values
- g = vector of 8 gains
- o = vector of 8 offsets
- Operations are element-wise (no matrix multiplication)




## 20250912colourmatching
### Aim:
- How consistent are the measurements from the three sensors across different colours?

## 20250912cosine
### Aim:
- Find cosine function for each of the  three?

According to [Jill Fowler](https://internationallight.com/blog/what-cosine-correction-and-how-does-it-effect-light-measurement?srsltid=AfmBOoqhQaqYq9bhCfUmLqxoz93NMw2hUKd15sTyTu3VFrq0CVrqKVIS):
    - What is cosine correction? Cosine correction is the ability to take a recessed detector by using an input optic as the receiving surface.
    - How does it effect the light measurement? It allows you to accurately measure the same amount of light that is received by your product.

- *Cosine response*
The angular sensitivity of an irradiance sensor ideally proportional to cos θ, where θ is the angle between the incident radiation and the sensor normal. This ensures correct weighting of oblique radiation, matching the physical definition of irradiance (flux per projected area).

- *Cosine correction*
Design features or post-processing applied to make the sensor’s angular response approximate the ideal cosine law. Implemented via optical diffusers, domes, or calibration-based angular correction factors to reduce angular error, especially at high incidence angles.