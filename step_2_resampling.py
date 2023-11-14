def main():
    """
    Configuration file for generating synthetic hemispheres of expected lidar returns by voxel ray samping of lidar
    (eg. to estimate light transmittance across the hemisphere at a given point, Staines thesis figure 3.3).
        batch_dir: directory for all batch outputs
        pts_in: coordinates and elevations at which to calculate hemispheres

    :return:
    """

    import numpy as np
    import pandas as pd
    import yaml
    import os
    import src.las_ray_sampling as lrs

    # Read YAML config file 2
    config_in = "config_2_resampling.yml"

    with open(config_in, 'r') as stream:
        config = yaml.safe_load(stream)

    config_id = config["config_id"]
    working_dir = os.path.normpath(config["working_dir"])

    # Initiate voxel obj
    vox = lrs.VoxelObj()
    vox.vox_hdf5 = os.path.join(working_dir, "voxrs_" + config_id + '_vox.h5')  # (str) file path to vox file (.hdf5)

    # # LOAD VOX
    # load existing vox files, using vox file path
    vox = lrs.load_vox(vox.vox_hdf5, load_post=True)

    # create outputs folder if not exists
    outputs_dir = os.path.join(working_dir, "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    runtag = config["runtag"]
    if runtag is None:
        runtag = ''
    elif runtag != '':
        runtag = "_" + runtag + "_"

    # # VOLUMETRIC RESAMPLING
    if config["resample_las"]:

        # create volumetric_resampling folder if not exists
        vol_dir = os.path.join(outputs_dir, 'volumetric_resampling')
        if not os.path.exists(vol_dir):
            os.makedirs(vol_dir)     

        samps_per_vox = config["samps_per_vox"]  # (int) volumetric resample rate [samples per voxel]
        sample_threshold = config["sample_threshold"]  # (int) noise filter will drop all returns from voxels where total returns <= sample_threshold           
        las_out = os.path.join(vol_dir, config_id + runtag + "_resampled.las")  # (str) output .las file path

        # GENERATE RESAMPLED POINT CLOUD
        lrs.vox_to_las(vox.vox_hdf5, las_out, samps_per_vox, sample_threshold)

    # # HEMISPHERE RESAMPLING
    if config["resample_hemi"]:

        # create hemi_dir (with error handling)
        hemi_dir = os.path.join(outputs_dir, 'hemisphere_resampling', runtag)
        lrs.create_dir(hemi_dir, desc='hemi')

        # define VoxRS hemisphere metadata object
        rshmeta = lrs.RaySampleGridMetaObj()
        rshmeta.file_dir = hemi_dir
        rshmeta.config_id = config_id
        rshmeta.agg_sample_length = vox.sample_length  # (num) ray resample length (default to same as sample length)
        rshmeta.lookup_db = 'posterior'  # (str) db lookup, default 'posterior'
        rshmeta.agg_method = 'single_ray_agg'  # (str) aggregation method, default 'single_ray_agg'

        # HEMISPHERE RAY GEOMETRY PARAMETERS
        rshmeta.img_size = config["img_size"]  # angular resolution (square) ~ samples / pi

        # phi_step = (np.pi / 2) / (180 * 2)  # alternative way of defining, based on angular step
        # rshmeta.max_phi_rad = phi_step * rshmeta.img_size

        rshmeta.max_phi_rad = config["max_phi_deg"] * np.pi/180  # maximum zenith angle to sample  (pi/2 samples to horizon)
        hemi_m_above_ground = config["hemi_m_above_ground"]  # height [m] above ground points at which to generate hemispheres
        rshmeta.max_distance = config["max_distance"]  # maximum distance [m] to sample ray (balance computation time with accuracy at distance)
        rshmeta.min_distance = config["min_distance"]  # minimum distance [m] to sample ray (default 0, increase to avoid "lens occlusion" within dense voxels)

        # PROCESSING PARAMETERS
        tile_count_1d = 5  # (int) number of square tiles along one side (total # of tiles = tile_count_1d^2)
        n_cores = 3  # (int) number of processing cores
        
        # PTS CONFIGURATION
        pts_in = config["pts_in"]  # (str) path to .csv file with coordinates and elevations at which to calculate hemispheres
        #            pts file must include the following header labels: id, easting_m, northing_m, elev_m
        pts = pd.read_csv(pts_in)

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