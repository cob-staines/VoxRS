# VoxRS

VoxRS is a voxel-based discrete ray tracing ("ray sampling") model used to model canopy light transmittance probabilities from discrete return lidar point clouds. The analysis takes place in two steps: 1) Sampling and 2) Resampling.

# Sampling
Intact lidar beam trajectories are constructed by merging the point cloud of first returns with sensor trajectories. Beam trajectories are sampled and a voxel-wise binomial model of lidar return probability is generated.

# Resampling
The probability of a lidar return along an arbitrary rays through the voxelspace can be calculated using the model from sampled lidar observations. Two configurations for resampling are supported:
* Hemispherical sampling: a set of rays over the hemisphere (or a subsection of it) is generated for a set of points
* Gridded sampling: a set of rays with a fixed zenith and azimuth angles is generated across a given grid.

# Getting Started
Config files (.py) are used to point to all files, specify configuration variables, and call functions.
