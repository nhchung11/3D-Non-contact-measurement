# 3D Non-contact Measurement using Depth image

## Overall
### Inputs
1. .bin files generated from Lidar camera
2. .txt parameters files
### Outputs
1. 3D model of the animal to visualize
2. The animal belly's length

## How it works
1. Generate 3D model from .bin file and parameters file
2. Locate the cage 
3. Calculate the deviation angle to bring the 3D into correct form
4. Remove the cage
5. Remove the extra parts
6. Calculate the belly's length
