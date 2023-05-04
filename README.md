# VoxRS

VoxRS is a voxel-based ray sampling tool used to model canopy light transmittance probabilities from discrete return lidar point clouds. The analysis is done in two steps: 1) sampling and 2) resampling.

## Sampling
A cloud of intact lidar beam trajectories is constructed by merging the lidar point cloud of first returns with simultaneous sensor geolocation data. A voxel space is generated overlaying the region of interest. Beam trajectories are then point-sampled at regular intervals and a voxel-wise binomial model of lidar return probability is generated.

## Resampling
The probability of a lidar return along an arbitrary ray through the voxel space is calculated from the probabilistic voxel model, derived from sampled lidar observations. A given ray is pont-sampled at regular intervals, and the probability of a return for each sample is inherited from the parent voxel. The model returns the estimated total number of returns along the ray, along with the modeled uncertainty. Output returns can be scaled using validation data from hemispheric photography, and then transformed exponentially to model light transmittance. Two ray configurations for resampling are supported:
* **Hemispherical resampling**: a set of rays over the hemisphere (or a subsection of it) is generated for a set of points
* **Gridded resampling**: a set of rays with a fixed zenith/azimuth pair is generated across a given grid.

## Getting Started
Config files (.py) are used to point to all files, specify configuration variables, and call functions.
