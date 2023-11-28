def main():
    """
    Configuration file for building and sampling voxel space for voxel ray samping of lidar

    :return:
    """

    import numpy as np
    import yaml
    import os
    import src.las_ray_sampling as lrs

    # Read YAML config file 1
    config_in = "config_1_sampling.yml"

    with open(config_in, 'r') as stream:
        config = yaml.safe_load(stream)

    # # DATA INPUT PARAMETERS
    # config_id (str) - unique identifier for the configuration used for voxel space identification
    config_id = config["config_id"]
    working_dir = os.path.normpath(config["working_dir"])

    lrs.create_dir(working_dir, desc='working')

    # Initiate voxel obj
    vox = lrs.VoxelObj()
    vox.config_id = config_id
    
    # vox.las_in (str) - file path of .las file
    vox.las_in = config["las_in"]

    # vox.traj_in (str) - file path to trajectory file (.csv) corresponding to las file.
    #            Trajectory file should include the following header labels: Time[s], Easting[m], Northing[m], Height[m]
    vox.traj_in = config["traj_in"]

    # # VOXEL SPACE PARAMETERS
    vox.las_traj_hdf5 = os.path.join(working_dir, "voxrs_" + config_id + "_las_traj.h5")  # file path to las/trajectory file (.hdf5)
    vox.return_set = config["return_set"]  # (str) - 'first' (recommended), 'last', or 'all'
    vox.drop_class = config["drop_class"]  # (int) single class to be dropped from .las file prior to interpolation (-1 for None)
    if vox.drop_class is None or vox.drop_class == "":
        vox.drop_class = -1
    
    vox.sample_dtype = np.uint32  # data type for voxel sample array (smaller is faster, overflow throws no errors)
    vox.return_dtype = np.uint32  # data type for voxel return array (smaller is faster, overflow throws no errors)
    vox.cw_rotation = config["cw_rotation_deg"] * np.pi / 180  # (int) rotation of primary voxelspace axis in radians (default 0)
    voxel_length = config["voxel_length"]  # voxel dimension in meters
    vox.step = np.full(3, voxel_length)  # cubic voxels by default
    vox.sample_length = voxel_length / np.pi  # ray sample length (keep smaller than voxel length)
    vox.vox_hdf5 = os.path.join(working_dir, "voxrs_" + config_id + '_vox.h5')  # file path to vox file (.hdf5)

    # # PROCESSING PARAMETERS
    vox.las_traj_chunksize = 10000000  # (int) point cloud chunk size for interpolation with trajectory (default 10000000)
    z_slices = 4  # (int) - number of horizontal layers for chunking of ray sampling (increase for memory management)


    # # BUILD VOX
    # runs step 1) Ray Sampling and generates a vox file
    vox = lrs.las_to_vox(vox, z_slices, run_las_traj=True, posterior_calc=True, fail_overflow=False)

if __name__ == "__main__":
    main()