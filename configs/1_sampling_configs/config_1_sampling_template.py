def main():
    """
    Configuration file for building and sampling voxel space for voxel ray samping of lidar

    :return:
    """

    import numpy as np
    from src import las_ray_sampling as lrs


    # # DATA INPUT PARAMETERS
    # config_id (str) - unique identifier for the configuration used for voxel space identification
    config_id = ''

    # Initiate voxel obj
    vox = lrs.VoxelObj()

    # vox.las_in (str) - file path of .las file
    vox.las_in = ''

    # vox.traj_in (str) - file path to trajectory file (.csv) corresponding to las file.
    #            Trajectory file should include the following header labels: Time[s], Easting[m], Northing[m], Height[m]
    vox.traj_in = ''

    # vox.return_set (str) - 'first' (recommended), 'last', or 'all'
    vox.return_set = 'first'

    # vox.drop class (int) single class to be dropped from .las file prior to interpolation
    vox.drop_class = 7


    # # VOXEL SPACE PARAMETERS
    vox.las_traj_hdf5 = vox.las_in.replace('.las', '_ray_sampling_' + vox.return_set + '_returns_drop_' + str(vox.drop_class) + '_las_traj.h5')  # file path to las/trajectory file (.hdf5)
    vox.sample_dtype = np.uint32  # data type for voxel sample array (smaller is faster, overflow throws no errors)
    vox.return_dtype = np.uint32  # data type for voxel return array (smaller is faster, overflow throws no errors)
    vox.las_traj_chunksize = 10000000  # point cloud chunk size for interpolation with trajectory (default 10000000)
    vox.cw_rotation = 0 * np.pi / 180  # rotation of primary voxelspace axis in radians (default 0)
    voxel_length = .25  # voxel dimension in meters
    vox.step = np.full(3, voxel_length)  # cubic voxels by default
    vox.sample_length = voxel_length / np.pi  # ray sample length (keep smaller than voxel length)
    vox.vox_hdf5 = vox.las_in.replace('.las', config_id + '_r' + str(voxel_length) + '_vox.h5')  # file path to vox file (.hdf5)

    # # PROCESSING PARAMETERS
    # z_slices (int) - number of horizontal layers for chunking of ray sampling (increase for memory management)
    z_slices = 4


    # # BUILD VOX
    # runs step 1) Ray Sampling and generates a vox file
    vox = lrs.las_to_vox(vox, z_slices, run_las_traj=True, fail_overflow=False, posterior_calc=False)

if __name__ == "__main__":
    main()