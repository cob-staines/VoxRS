
import laspy
import numpy as np
import pandas as pd
import h5py
import warnings


def las_traj(las_in, traj_in, hdf5_path, chunksize=10000000, keep_return='all', drop_class=None):
    """
    For each point in las_in (after filtering for return and class), las_traj interpolates trajectory coordinates.
    distance_from_sensor_m, angle_from_nadir_deg, and angle_cw_from_north_deg are then calculated and stored in the
    hdf5 file along with the corresponding las and trajectory xyz data.

    :param las_in: path to las_file
    :param traj_in: path to trajectory file (.csv) corresponding to las file.
           Trajectory file should include the following header labels: Time[s], Easting[m], Northing[m], Height[m]
    :param hdf5_path: name of hdf5 file which will be created to store las_traj output
    :param chunksize: max size of chunks to be written into hdf5 file
    :param keep_return: speacify 'all', 'first' or 'last' to keep all returns, first returns or last returns only,
    respectively, prior to interpolation
    :param drop_class: integer of single class to be dropped from las prior to interpolation
    :return:
    """

    print('Loading LAS file... ', end='')
    # load las_in
    inFile = laspy.read(las_in)
    print('done')

    # filter points
    p0 = inFile.points

    # drop noise
    if drop_class >= 0:
        print('Filtering points by class... ', end='')
        if not isinstance(drop_class, int):
            raise Exception(
                'drop_class only set to handle single classes. More development required to handle multiple classes')
        p0 = p0[inFile.classification != drop_class]
        print('done')

    # keep returns
    print('Filtering points by return... ', end='')
    if keep_return == 'first':
        p0 = p0[p0.return_num == 1]
    elif keep_return == 'last':
        p0 = p0[p0.return_num == p0.num_returns]
    elif keep_return != 'all':
        raise Exception(
            'Return sets other than "all", "first" and "last" have yet to be programmed. Will you do the honors?')
    print('done')

    d0 = inFile
    d0.points = p0

    p0 = pd.DataFrame({"gps_time": np.array(d0.gps_time),
                       "x": np.array(d0.x),
                       "y": np.array(d0.y),
                       "z": np.array(d0.z)})

    print('Sorting returns... ', end='')
    p0 = p0.sort_values(by="gps_time")  # improve join time
    print('done')

    n_rows = len(p0)
    las_cols = p0.columns

    if chunksize is None:
        chunksize = n_rows
    elif n_rows < chunksize:
        chunksize = n_rows

    print('Writing LAS data to HDF5... ', end='')
    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('lasData', shape=p0.shape, data=p0.values, chunks=(chunksize, 1), compression='gzip')
        hf.create_dataset('lasData_cols', data=las_cols, dtype=h5py.string_dtype(encoding='utf-8'))
    inFile = None
    d0 = None
    p0 = None
    print('done')

    print('Loading trajectory... ', end='')
    # load trajectory from csv
    traj = pd.read_csv(traj_in)
    # rename columns for consistency
    traj = traj.rename(columns={'Time[s]': "gps_time",
                                'Easting[m]': "traj_x",
                                'Northing[m]': "traj_y",
                                'Height[m]': "traj_z"})
    # drop pitch, roll, yaw
    traj = traj[['gps_time', 'traj_x', 'traj_y', 'traj_z']]
    traj = traj.sort_values(by="gps_time").reset_index(drop=True)  # improve join time

    # add las key (False)
    traj = traj.assign(las=False)
    print('done')

    # preallocate output
    traj_interpolated = pd.DataFrame(columns=["gps_time", "traj_x", "traj_y", "traj_z", "distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"])

    n_chunks = np.ceil(n_rows / chunksize).astype(int)

    for ii in range(0, n_chunks):

        # chunk start and end
        las_start = ii * chunksize
        if ii != (n_chunks - 1):
            las_end = (ii + 1) * chunksize
        else:
            las_end = n_rows

        # take chunk of las data
        with h5py.File(hdf5_path, 'r') as hf:
            las_data = pd.DataFrame(hf['lasData'][las_start:las_end, 0:4], columns=['gps_time', 'x', 'y', 'z'])

        # add las key (True)
        las_data = las_data.assign(las=True)

        # only pull in relevant traj
        traj_start = np.max(traj.index[traj.gps_time < np.min(las_data.gps_time)])
        traj_end = np.min(traj.index[traj.gps_time > np.max(las_data.gps_time)])

        # append traj to las, keeping track of las index
        outer = las_data[['gps_time', 'las']]._append(traj.loc[traj_start:traj_end, :], sort=False)
        outer = outer.reset_index()
        outer = outer.rename(columns={"index": "index_las"})

        # order by gps time
        outer = outer.sort_values(by="gps_time")

        # QC: check first and last entries are traj
        if (outer.las.iloc[0] | outer.las.iloc[-1]):
            raise Exception('LAS data exists outside trajectory time frame -- Suspect LAS/trajectory file mismatch')

        # set index as gps_time
        outer = outer.set_index('gps_time')

        # forward fill nan values
        interpolated = outer.ffill()

        # drop traj entries
        interpolated = interpolated[interpolated['las']]
        # reset to las index
        interpolated = interpolated.set_index("index_las")
        # drop las key column
        interpolated = interpolated[['traj_x', 'traj_y', 'traj_z']]

        # concatenate with las_data horizontally by index
        merged = pd.concat([las_data, interpolated], axis=1, ignore_index=False)

        # distance from sensor
        p1 = np.array([merged.traj_x, merged.traj_y, merged.traj_z])
        p2 = np.array([merged.x, merged.y, merged.z])
        squared_dist = np.sum((p1 - p2) ** 2, axis=0)
        merged = merged.assign(distance_from_sensor_m=np.sqrt(squared_dist))

        # angle from nadir
        dp = p1 - p2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
            phi = np.arctan(np.sqrt(dp[0] ** 2 + dp[1] ** 2) / dp[2]) * 180 / np.pi  # in degrees
        merged = merged.assign(angle_from_nadir_deg=phi)

        # angle cw from north
        theta = np.arctan2(dp[0], (dp[1])) * 180 / np.pi
        merged = merged.assign(angle_cw_from_north_deg=theta)

        if traj_interpolated.empty:
            traj_interpolated = merged.loc[:, ["gps_time", "traj_x", "traj_y", "traj_z", "distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"]]
        else:
            traj_interpolated = pd.concat([traj_interpolated, merged.loc[:, ["gps_time", "traj_x", "traj_y", "traj_z", "distance_from_sensor_m", "angle_from_nadir_deg", "angle_cw_from_north_deg"]]])

        print('Interpolated ' + str(ii + 1) + ' of ' + str(n_chunks) + ' chunks')


    # save to hdf5 file
    print('Writing interpolated_traj data to HDF5... ', end='')
    with h5py.File(hdf5_path, 'r+') as hf:
        hf.create_dataset('trajData', traj_interpolated.shape, data=traj_interpolated.values, chunks=(chunksize, 1), compression='gzip')
        hf.create_dataset('trajData_cols', data=traj_interpolated.columns, dtype=h5py.string_dtype(encoding='utf-8'))
    traj_interpolated = None
    print('done')
