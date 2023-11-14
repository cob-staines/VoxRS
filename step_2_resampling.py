def main():
    """
    Configuration file for generating synthetic hemispheres of expected lidar returns by voxel ray samping of lidar
    (eg. to estimate light transmittance across the hemisphere at a given point, Staines thesis figure 3.3).
        batch_dir: directory for all batch outputs
        pts_in: coordinates and elevations at which to calculate hemispheres

    :return:
    """

    import src.las_ray_sampling as lrs
    import numpy as np
    import pandas as pd
    import yaml
    import os

    # Read YAML config file 2
    config_in = "config_2_resampling.yml"

    with open(config_in, 'r') as stream:
        config = yaml.safe_load(stream)

    config_id = config["config_id"]

    # Initiate voxel obj
    vox = lrs.VoxelObj()
    vox.vox_hdf5 = ''  # (str) file path to vox file (.hdf5)

    # # LOAD VOX
    # load existing vox files, using vox file path
    vox = lrs.load_vox(vox.vox_hdf5, load_data=False)

    # # VOLUMETRIC RESAMPLING PARAMETERS
    resample_las = True   # resample_las (bool) - Do you want to generate a resampled point cloud?
    samps_per_vox = 50  # (int) volumetric resample rate [samples per voxel]
    sample_threshold = 0  # (int) noise filter will drop all returns from voxels where total returns <= sample_threshold
    las_out = vox.vox_hdf5.replace(".h5", "_resampled.las")  # (str) output .las file path

    # # GENERATE RESAMPLED POINT CLOUD
    if resample_las:
        lrs.vox_to_las(vox.vox_hdf5, las_out, samps_per_vox, sample_threshold)

    # # VoxRS HEMISPHERE PARAMETERS
    batch_dir = ''  # (str) directory for hemisphere batch outputs
    pts_in = ""  # (str) path to .csv file with coordinates and elevations at which to calculate hemispheres
    #            pts file should include the following header labels: id, easting_m, northing_m, elev_m
    pts = pd.read_csv(pts_in)

    # define VoxRS hemisphere metadata object
    rshmeta = lrs.RaySampleGridMetaObj()
    rshmeta.config_id = vox.config_id
    rshmeta.agg_sample_length = vox.sample_length  # (num) ray resample length (default to same as sample length)
    rshmeta.lookup_db = 'posterior'  # (str) db lookup, default 'posterior'
    rshmeta.agg_method = 'single_ray_agg'  # (str) aggregation method, default 'single_ray_agg'

    # HEMISPHERE RAY GEOMETRY PARAMETERS
    # phi_step = (np.pi / 2) / (180 * 2)
    rshmeta.img_size = 181  # angular resolution (square) ~ samples / pi
    # rshmeta.max_phi_rad = phi_step * rshmeta.img_size
    rshmeta.max_phi_rad = np.pi/2  # maximum zenith angle to sample  (pi/2 samples to horizon)
    hemi_m_above_ground = 0  # height [m] above ground points at which to generate hemispheres
    rshmeta.max_distance = 50  # maximum distance [m] to sample ray, balance comp. time with accuracy at distance
    rshmeta.min_distance = 0  # minimum distance [m] to sample ray, default 0, increase to avoid "lens occlusion"

    # PROCESSING PARAMETERS
    tile_count_1d = 5  # (int) number of square tiles along one side (total # of tiles = tile_count_1d^2)
    n_cores = 3  # (int) number of processing cores

    # create batch dir (with error handling)
    lrs.create_dir(batch_dir, desc='batch')

    # create output file dir
    rshmeta.file_dir = batch_dir + "outputs\\"
    if not os.path.exists(rshmeta.file_dir):
        os.makedirs(rshmeta.file_dir)


    rshmeta.id = pts.id
    rshmeta.origin = np.array([pts.easting_m,
                               pts.northing_m,
                               pts.elev_m + hemi_m_above_ground]).swapaxes(0, 1)

    rshmeta.file_name = [rshmeta.config_id + "_" + str(idd) + ".tif" for idd in pts.id]

    rshm = lrs.rs_hemigen(rshmeta, vox, tile_count_1d, n_cores)


# add config for sample from grid

if __name__ == "__main__":
    main()

# # preliminary visualization
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import tifffile as tif
# ii = 0
# peace = tif.imread(rshm.file_dir[ii] + rshm.file_name[ii])
# plt.imshow(peace[:, :, 2], interpolation='nearest')