# VoxRS

VoxRS is a voxel-based ray sampling tool used to model canopy light transmittance probabilities from discrete return lidar point clouds. The analysis is done in two steps: 1) sampling and 2) resampling.

## Sampling
A cloud of intact lidar beam trajectories is constructed by merging the lidar point cloud of first returns with simultaneous sensor geolocation data. A voxel space is generated overlaying the region of interest. Beam trajectories are then point-sampled at regular intervals and a voxel-wise binomial model of lidar return probability is generated.

## Resampling
The probability of a lidar return along an arbitrary ray through the voxel space is calculated from the probabilistic voxel model, derived from sampled lidar observations. A given ray is pont-sampled at regular intervals, and the probability of a return for each sample is inherited from the parent voxel. The model returns the estimated total number of returns along the ray, along with the modeled uncertainty. Output returns can be scaled using validation data from hemispheric photography, and then transformed exponentially to model light transmittance. Two ray configurations for resampling are supported:
* **Hemispherical resampling**: a set of rays over the hemisphere (or a subsection of it) is generated for a set of points
* **Gridded resampling**: a set of rays with a fixed zenith/azimuth pair is generated across a given grid.

## Getting Started

### Download VoxRS
VoxRS is available for download on Github. The easiest way to get the files is to clone the repository with:

```
git clone https://github.com/jstaines/VoxRS.git
```

### Install Dependencies
 - Python >= 3.7
 - Install GDAL development files from package manager (ex. 'gdal-devel')

### Install VoxRS
With Conda:
```
cd VoxRS
conda env create -f requirements.yml
conda activate voxrs
```

With pip:
```
cd VoxRS
pip install -r "requirements.txt"
```

### Running VoxRS
Config files (.yml) are used to specify configuration variables. Sperate configuration files are used for the sampling (config_1_sampling.yml) and resampling (config_2_resampling.yml) steps. To run VoxRS, enter the desired configurations and save the files, then run the corresponding python files (step_1_sampling.py, then step_2_resampling.py) in order. More flexibility in configuration variables and algorythm selection can be found by reviewing and revising the two .py files.
