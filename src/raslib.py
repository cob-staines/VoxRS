import numpy as np
import pandas as pd
from osgeo import osr
from osgeo import gdal, gdalconst
from affine import Affine
from copy import deepcopy

class rasterObj(object):
    """
    Raster object used to pass master data and metadata through raslib functions
    """

    def __init__(self, raster):

        # load metadata
        self.gt = raster.GetGeoTransform()
        self.proj = raster.GetProjection()
        self.cols = raster.RasterXSize
        self.rows = raster.RasterYSize
        self.band_count = raster.RasterCount
        if self.band_count == 1:
            self.band = raster.GetRasterBand(1)
            self.data = self.band.ReadAsArray()
            self.no_data = self.band.GetNoDataValue()
        elif self.band_count > 1:
            self.band = []
            self.data = []
            for ii in range(1, self.band_count + 1):
                self.band.append(raster.GetRasterBand(ii))
                self.data.append(self.band[ii - 1].ReadAsArray())
            self.no_data = self.band[0].GetNoDataValue()
        # get affine transformation
        self.T0 = Affine.from_gdal(*raster.GetGeoTransform())
        # cell-centered affine transformation
        self.T1 = self.T0 * Affine.translation(0.5, 0.5)

    def copy(self):
        return deepcopy(self)


def raster_load(ras_in):
    # takes in path for georaster file, returns raster objects with following attributes:
        # ras.data -- numpy array of raster data
        # ras.gt -- geotransformation
        # ras.proj -- projection
        # ras.cols -- number of columns in raster
        # ras.rows -- number of rows in raster
        # ras.band -- raster band (1 only supported)
        # ras.T0 -- affine transformation for cell corners
        # ras.T1 -- affine transformation for cell centers

    # dependencies
    # import gdal
    from osgeo import gdal

    # open single band geo-raster file
    ras = gdal.Open(ras_in, gdal.GA_ReadOnly)

    # read data
    ras_out = rasterObj(ras)

    # close file
    ras = None

    return ras_out


# saves raster to file
def raster_save(ras_object, file_path, file_format="GTiff", data_format="float32"):
    # saves "ras_object" to "file_path" in "file_format"
    # file_format can be: "GTiff",
    # data_format can be: "float32", "float64", "byte", "int16", "int32", "uint16", "uint32"

    if data_format == "float32":
        gdal_data_format = gdal.GDT_Float32
    elif data_format == "float64":
        gdal_data_format = gdal.GDT_Float64
    elif data_format == "byte":
        gdal_data_format = gdal.GDT_Byte
    elif data_format == "int16":
        gdal_data_format = gdal.GDT_Int16
    elif data_format == "int32":
        gdal_data_format = gdal.GDT_Int32
    elif data_format == "uint16":
        gdal_data_format = gdal.GDT_UInt16
    elif data_format == "uint32":
        gdal_data_format = gdal.GDT_UInt32
    else:
        raise Exception(data_format, 'is not a valid data_format.')

    # confirm band count matches data length
    if isinstance(ras_object.data, list):
        if ras_object.data.__len__() != ras_object.band_count:
            raise Exception("ras_object.band_count and length of ras_object.data do not agree.")
    elif isinstance(ras_object.data, np.ndarray):
        if ras_object.band_count == 1:
            if ras_object.data.shape.__len__() == 2:
                # nest data in list for single band output
                ras_object.data = [ras_object.data]
            else:
                raise Exception("2D array expected for ras_object.data and ras_object.band_count == 1")
        else:
            raise Exception("multi-band output as 3D array not yet supported. Consider passing as list of 2D arrays.")

    outdriver = gdal.GetDriverByName(file_format)
    outdata = outdriver.Create(file_path, ras_object.cols, ras_object.rows, ras_object.band_count, gdal_data_format)
    # Set metadata
    outdata.SetGeoTransform(ras_object.gt)
    outdata.SetProjection(ras_object.proj)

    # Write data for each band
    for ii in range(0, ras_object.band_count):
        outdata.GetRasterBand(ii + 1).WriteArray(ras_object.data[ii])
        outdata.GetRasterBand(ii + 1).SetNoDataValue(ras_object.no_data)

    del outdata  # Flush


def clip_raster_to_valid_extent(ras):

    # test if ras is path or raster_object
    if isinstance(ras, str):
        ras_in = ras
        ras = raster_load(ras_in)
    elif not isinstance(ras, rasterObj):
        raise Exception('ras is not an instance of rasterObj or str (filepath), raster_to_pd() aborted.')

    if ras.band_count > 1:
        # unstack data
        data = np.full((ras.rows, ras.cols, ras.band_count), ras.no_data)
        for ii in range(0, ras.band_count):
            data[:, :, ii] = ras.data[ii]
        valid = np.where(np.any(data != ras.no_data, axis=2))
    else:
        # nest data in list
        ras.data = [ras.data]
        valid = np.where(ras.data != ras.no_data)

    yc_min, xc_min = np.min(valid, axis=1)
    yc_max, xc_max = np.max(valid, axis=1)

    x_min, y_min = ras.T0 * (xc_min, yc_min)

    ras.gt = (x_min, ras.gt[1], ras.gt[2], y_min, ras.gt[4], ras.gt[5])

    new_data = []
    for ii in range(0, ras.band_count):
        band = np.full((yc_max - yc_min + 1, xc_max - xc_min + 1), ras.no_data)
        band[(valid[0] - yc_min, valid[1] - xc_min)] = ras.data[ii][valid]
        new_data.append(band)

    ras.data = new_data

    ras.rows, ras.cols = ras.data[0].shape

    if ras.band_count == 1:
        ras.data = ras.data[0]

    ras.T0 = Affine.from_gdal(*ras.gt)
    # cell-centered affine transformation
    ras.T1 = ras.T0 * Affine.translation(0.5, 0.5)

    return ras

def raster_to_pd(ras, colnames, include_nans=False):

    # test if ras is path or raster_object
    if isinstance(ras, str):
        ras_in = ras
        ras = raster_load(ras_in)
    elif not isinstance(ras, rasterObj):
        raise Exception('ras is not an instance of rasterObj or str (filepath), raster_to_pd() aborted.')

    if isinstance(ras.data, np.ndarray):
        # nest data in list if not already
        ras.data = [ras.data]
    if ras.band_count != len(ras.data):
        raise Exception('data dimensions do not match band_count, raster_to_pd() aborted.')

    if isinstance(colnames, str):
        # nest colname in list if not already
        colnames = [colnames]
    if ras.band_count != len(colnames):
        raise Exception('length of colname does not match band_count, raster_to_pd() aborted.')

    all_vals = np.full([ras.rows, ras.cols], True)

    nan_vals = np.full([ras.rows, ras.cols, ras.band_count], False)
    for ii in range(0, ras.band_count):
        nan_vals[:, :, ii] = (ras.data[ii] == ras.no_data)
    nan_vals = np.all(nan_vals, axis=2)

    if include_nans:
        # inlcude all
        pts_index = np.where(all_vals)
    else:
        # only include non nans
        pts_index = np.where(~nan_vals)

    pts_flop = (pts_index[1], pts_index[0])
    pts_coords = ras.T1 * pts_flop  # flop from numpy (y, x) to affine (x, y)
    pts = pd.DataFrame({'x_coord': pts_coords[0],  # affine transform output returns [x, y]
                        'y_coord': pts_coords[1],
                        'x_index': pts_index[1],  # numpy output from np.where() returns [y, x]
                        'y_index': pts_index[0]})

    for ii in range(0, ras.band_count):
        pts.loc[:, colnames[ii]] = ras.data[ii][pts.y_index, pts.x_index]
        # sub in np.nan for no_data values
        pts.loc[pts.loc[:, colnames[ii]] == ras.no_data, colnames[ii]] = np.nan

    return pts


def pd_to_raster(df, colname, ids_in, ras_out=None):
    # take dataframe df with rows corresponding to raster coords, create raster from column colname, using ids_in as template

    if isinstance(df, str):
        df_in = df
        df = pd.read_csv(df_in)
    elif not isinstance(df, pd.core.frame.DataFrame):
        raise Exception('df is not an instance of pd.core.frame.DataFrame or str(filepath), pd_to_raster() aborted.')

    # ids = raster_load(ids_in)
    ras = raster_load(ids_in)  # use as template

    # ras to pd
    ids_pd = raster_to_pd(ids_in, colnames="id")

    # merge df with ids
    merged = pd.merge(df, ids_pd, how="left", on="id")

    # wipe ras data
    ras.data = np.full((ras.rows, ras.cols), ras.no_data)
    # assign ras data
    ras.data[(merged.y_index.values, merged.x_index.values)] = merged.loc[:, colname]

    if ras_out is not None:
        # write to file
        raster_save(ras, ras_out)

    return ras


def gdal_raster_reproject(src, match, nodatavalue=np.nan, mode="nearest"):

    # Source
    if isinstance(src, str):
        src_filename = src
        src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    elif ~isinstance(src, gdal.Dataset):
        raise Exception('src is not either a file path or osgeo.gdal.Dataset')

    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    band_count = src.RasterCount

    # Match
    if isinstance(match, str):
        match_filename = match
        match = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    elif ~isinstance(match, gdal.Dataset):
        raise Exception('match is not either a file path or osgeo.gdal.Dataset')

    # We want a section of source that matches this:
    match_proj = match.GetProjection()
    match_geotrans = match.GetGeoTransform()
    wide = match.RasterXSize
    high = match.RasterYSize

    # create memory destination
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', wide, high, band_count, gdal.GDT_Float32)
    # pad with nodatavalue
    for ii in range(1, band_count + 1):
        dest.GetRasterBand(ii).WriteArray(np.full((high, wide), nodatavalue), 0, 0)

    # Set mode
    if mode == "nearest":
        gdal_mode = gdal.GRA_NearestNeighbour
    elif mode == "mean":
        gdal_mode = gdal.GRA_Average
    elif mode == "median":
        gdal_mode = gdal.GRA_Med
    elif mode == "cubic":
        gdal_mode = gdal.GRA_Cubic
    elif mode == "cubic_spline":
        gdal_mode = gdal.GRA_CubicSpline
    elif mode == "lanczos":
        gdal_mode = gdal.GRA_Lanczos
    elif mode == "bilinear":
        gdal_mode = gdal.GRA_Bilinear
    else:
        raise Exception("Mode not yet defined. Will you do the honors_")

    # Set the geotransform
    dest.SetGeoTransform(match_geotrans)
    dest.SetProjection(match_proj)
    # Perform projection/resampling
    # res = gdal.ReprojectImage(src, dest, src_proj, match_proj, gdal.GRA_Bilinear)
    res = gdal.ReprojectImage(src, dest, src_proj, match_proj, gdal_mode)

    rp_array = np.full((high, wide, band_count), nodatavalue)
    for ii in range(1, band_count + 1):
        rp_array[:, :, ii - 1] = np.array(dest.GetRasterBand(ii).ReadAsArray())

    del dest  # Flush

    return rp_array


def ras_reproject(ras_data, ras_proj, ras_reproj_out, mode="nearest"):

    samp = raster_load(ras_data)

    ras_data_out = gdal_raster_reproject(ras_data, ras_proj, nodatavalue=samp.no_data, mode=mode)

    ras = raster_load(ras_proj)
    ras.data = ras_data_out[:, :, 0]
    raster_save(ras, ras_reproj_out)


def pd_sample_raster_gdal(data_dict, include_nans=False, nodatavalue=np.nan, mode="nearest"):
    files = list(data_dict.values())
    colnames = list(data_dict.keys())

    # take first item in files as parent
    print('Loading ' + str(colnames[0]) + "... ", end='')
    df = raster_to_pd(files[0], colnames[0], include_nans=include_nans)
    print('done')

    # for remaining items in files
    for ii in range(1, len(files)):
        print('Loading ' + str(colnames[ii]) + "... ", end='')
        rs_array = gdal_raster_reproject(files[ii], files[0], nodatavalue=nodatavalue, mode=mode)

        band_count = rs_array.shape[2]
        if band_count == 1:
            df.loc[:, colnames[ii]] = rs_array[df.y_index, df.x_index, 0]
        elif band_count > 1:
            if len(colnames[ii]) != band_count:
                raise Exception('colnames key ' + str(colnames[ii]) + ' does not agree with number of bands in image.')
            for jj in range(0, band_count):
                df.loc[:, colnames[ii][jj]] = rs_array[df.y_index, df.x_index, jj]

        print('done')
    return df
