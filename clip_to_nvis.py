# %%
import os
from glob import glob
import geopandas as gpd
from multiprocessing import Pool
import multiprocessing as mp

from joblib import Parallel, delayed
import joblib

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import rasterio
from rasterio.windows import from_bounds
from pyproj import CRS
import sys
import time

# %%
def init_qgis():
    #Imports specifically for QGIS api
    import sys
    import os
    from pathlib import Path
    # Get conda environment path
    conda_prefix = Path(sys.executable).parents[1]
    # Set environment variables
    os.environ['QGIS_PREFIX_PATH'] = str(conda_prefix)
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    # Add QGIS Python paths
    qgis_python_path = conda_prefix / "share" / "qgis" / "python"
    plugins_path = conda_prefix / "share" / "qgis" / "python" / "plugins"
    for path in [str(plugins_path), str(qgis_python_path)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    from qgis.core import QgsApplication
    from qgis.analysis import QgsNativeAlgorithms

    from qgis.analysis import QgsNativeAlgorithms
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    from qgistools import QGISTools

    tools = QGISTools()

    return tools

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2020/nvis_polygon_7853.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2020/merged/NVIS_8L_fm_mape_r1_KI2020_merged.tif'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2020/'
# output_name = 'NVIS_8L_fm_mape_r1_KI2020'

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_WASouth/NVIS_WASouth_7850.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_WASouth/merged/WASouth_merged.tif'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_WASouth/'
# output_name = 'NVIS_8L_fm_mape_r1_WASouth'

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2025/nvis_polygon_7853.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2025/merged/NVIS_8L_fm_mape_r1_KI2025_merged.tif'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2025/'
# output_name = 'NVIS_8L_fm_mape_r1_KI2025'

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Billabong/NVIS_Billabong_7850.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Billabong/merged/NVIS_8L_fm_mape_r1_Billabong_merged.tif'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Billabong/'
# output_name = 'NVIS_8L_fm_mape_r1_Billabong'

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_MulgaLands/NVIS_Mulga_NSW.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_MulgaLands/merged/NVIS_8L_fm_mape_r1_MulgaLands_merged.tif'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_MulgaLands/'
# output_name = 'NVIS_8L_fm_mape_r1_MulgaLands'

nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Tenterfield/NVIS_Tenterfield_QLD.gpkg'
nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Tenterfield/merged/NVIS_8L_fm_mape_r1_Tenterfield_merged.tif'
outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Tenterfield/'
output_name = 'NVIS_8L_fm_mape_r1_Tenterfield_QLD'

# %%
os.environ['GDAL_CACHEMAX'] = '2048'  # 2GB cache (adjust based on your RAM)
os.environ['GDAL_NUM_THREADS'] = 'ALL_CPUS'  # Use all CPU cores
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['VSI_CACHE'] = 'TRUE'
os.environ['VSI_CACHE_SIZE'] = '25000000'  # 25MB cache per file
os.environ['GDAL_MAX_DATASET_POOL_SIZE'] = '450'

from shapely import wkt

def create_batch_grid(nvis_polygon, grid_size, temp_path):
    nvis_gdf = gpd.read_file(nvis_polygon)
    tools = init_qgis()
    # nvis_bounds = nvis_gdf.unary_union.bounds
    nvis_bounds = nvis_gdf.union_all().bounds
    nvis_bounds_rounded = (
        ((nvis_bounds[0] // grid_size)) * grid_size, # minx
        ((nvis_bounds[1] // grid_size)) * grid_size, # miny
        ((nvis_bounds[2] // grid_size) + 1) * grid_size, # maxx
        ((nvis_bounds[3] // grid_size) + 1) * grid_size  # maxy
    )
    print(nvis_bounds_rounded)

    batch_grid_extent = str(nvis_bounds_rounded[0] - 1000) + ',' + str(nvis_bounds_rounded[2] + 1000) + ',' + str(nvis_bounds_rounded[1] - 1000) + ',' + str(nvis_bounds_rounded[3] + 1000) + f' [{nvis_gdf.crs.srs}]'

    print(batch_grid_extent)

    output_path = os.path.join(temp_path, f'batch_grid_{grid_size}m.gpkg')

    tools.creategrid(
            nvis_gdf.crs.srs,
            batch_grid_extent,
            grid_size,
            2,
            output_path
        )
    
    return output_path

def process_batch(row_dict,nvis_gdf_dissolved):
    tools = init_qgis()
    row_dict['geometry'] = wkt.loads(row_dict['geometry_wkt'])
    batch_grid = row_dict
    batch_grid_bounds = batch_grid['geometry'].bounds
    
    # check if batch grid overlaps nvis_gdf
    if batch_grid['geometry'].intersects(nvis_gdf_dissolved['geometry']).any():
        batch_id = str(int(batch_grid_bounds[0])) + '_' + str(int(batch_grid_bounds[1]))
        extent = str(batch_grid_bounds[0] - 10) + ',' + str(batch_grid_bounds[2] + 10) + ',' + str(batch_grid_bounds[1] - 10) + ',' + str(batch_grid_bounds[3] + 10) + f' [{batch_grid_gdf.crs.srs}]'
        
        start_time = time.time()
        tools.creategrid(
            batch_grid_gdf.crs.srs,
            extent,
            10,
            2,
            os.path.join(temp_path, f'{batch_id}_batch_grid_10m.gpkg')
        )
        # print(f"Batch grid created in {time.time() - start_time} seconds for batch {batch_id}")

        clipped_nvis = os.path.join(temp_path, f'{batch_id}_clipped_nvis.gpkg')
        clipped_nvis_gdf = gpd.read_file(clipped_nvis)

        start_time = time.time()
        batch_10m_grid = gpd.read_file(os.path.join(temp_path, f'{batch_id}_batch_grid_10m.gpkg'))
        # overlay_grid = batch_10m_grid.overlay(nvis_gdf)
        overlay_grid = batch_10m_grid.overlay(clipped_nvis_gdf)
        overlay_grid.to_file(os.path.join(temp_path, f'{batch_id}_overlay_grid.gpkg'))
        # print(f"Overlay grid created in {time.time() - start_time} seconds for batch {batch_id}")

        start_time = time.time()
        null_values = ['', '-9999.0']
        upper_classes = list(overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_U_HEIGHT_CLASS.unique())
        upper_classes = [c for c in upper_classes if c not in null_values]
        middle_classes = list(overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_M_HEIGHT_CLASS.unique())
        middle_classes = [c for c in middle_classes if c not in null_values]
        ground_classes = list(overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_G_HEIGHT_CLASS.unique())
        ground_classes = [c for c in ground_classes if c not in null_values]
        # print(f"Classes extracted in {time.time() - start_time} seconds for batch {batch_id}")

        upper_files = []
        middle_files = []
        ground_files = []

        upper_zonals_files = []
        middle_zonals_files = []
        ground_zonals_files = []

        start_time = time.time()
        for upper_class in upper_classes:
            overlay_grid_upper = overlay_grid[overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_U_HEIGHT_CLASS == upper_class]
            if not overlay_grid_upper.empty:
                overlay_grid_upper_filename = os.path.join(temp_path, f'{batch_id}_overlay_grid_upper_{str(int(float(upper_class)))}.gpkg')
                overlay_grid_upper.to_file(overlay_grid_upper_filename)
                upper_files.append(overlay_grid_upper_filename)
        null_upper = overlay_grid[overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_U_HEIGHT_CLASS.isin(null_values)]
        if len(null_upper) > 0:
            null_upper.to_file(os.path.join(temp_path, f'{batch_id}_overlay_grid_upper_null.gpkg'))
            upper_zonals_files.append(os.path.join(temp_path, f'{batch_id}_overlay_grid_upper_null.gpkg'))
        # print(f"Upper classes extracted in {time.time() - start_time} seconds for batch {batch_id}")
        
        start_time = time.time()
        for middle_class in middle_classes:
            overlay_grid_middle = overlay_grid[overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_M_HEIGHT_CLASS == middle_class]
            if not overlay_grid_middle.empty:
                overlay_grid_middle_filename = os.path.join(temp_path, f'{batch_id}_overlay_grid_middle_{str(int(float(middle_class)))}.gpkg')
                overlay_grid_middle.to_file(overlay_grid_middle_filename)
                middle_files.append(overlay_grid_middle_filename)
        null_middle = overlay_grid[overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_M_HEIGHT_CLASS.isin(null_values)]
        if len(null_middle) > 0:
            null_middle.to_file(os.path.join(temp_path, f'{batch_id}_overlay_grid_middle_null.gpkg'))
            middle_zonals_files.append(os.path.join(temp_path, f'{batch_id}_overlay_grid_middle_null.gpkg'))
        # print(f"Middle classes extracted in {time.time() - start_time} seconds for batch {batch_id}")
        
        start_time = time.time()
        for ground_class in ground_classes:
            overlay_grid_ground = overlay_grid[overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_G_HEIGHT_CLASS == ground_class]
            if not overlay_grid_ground.empty:
                overlay_grid_ground_filename = os.path.join(temp_path, f'{batch_id}_overlay_grid_ground_{str(int(float(ground_class)))}.gpkg')
                overlay_grid_ground.to_file(overlay_grid_ground_filename)
                ground_files.append(overlay_grid_ground_filename)
        null_ground = overlay_grid[overlay_grid.NVIS_V7_0_LUT_AUST_FLAT_G_HEIGHT_CLASS.isin(null_values)]
        if len(null_ground) > 0:
            null_ground.to_file(os.path.join(temp_path, f'{batch_id}_overlay_grid_ground_null.gpkg'))
            ground_zonals_files.append(os.path.join(temp_path, f'{batch_id}_overlay_grid_ground_null.gpkg'))
        # print(f"Ground classes extracted in {time.time() - start_time} seconds for batch {batch_id}")
        
        nn_output_tile = os.path.join(temp_path, f'{batch_id}_nn_tile.tif')

        for upper_file in upper_files:
            start_time
            try:
                out_zonals_path = os.path.join(temp_path, f'{batch_id}_zonal_stats_upper_{upper_file.split("_")[-1].split(".")[0]}.gpkg')
                # (column_prefix, input_vector_layer,input_raster_layer, raster_band , statistics, output_path, verbose=True)
                tools.calculate_zonal_statitics(
                    'zs_',
                    upper_file,
                    nn_output_tile,
                    int(upper_file.split("_")[-1].split(".")[0]),
                    [2],
                    out_zonals_path
                )
                upper_zonals_files.append(out_zonals_path)
            except Exception as e:
                print(f"Error processing zonals for upper class {upper_file}: {e}")
            # print(f"Zonal statistics for upper class processed in {time.time() - start_time} seconds for batch {batch_id}")
        
        for middle_file in middle_files:
            start_time = time.time()
            try:
                out_zonals_path = os.path.join(temp_path, f'{batch_id}_zonal_stats_middle_{middle_file.split("_")[-1].split(".")[0]}.gpkg')
                # (column_prefix, input_vector_layer,input_raster_layer, raster_band , statistics, output_path, verbose=True)
                tools.calculate_zonal_statitics(
                    'zs_',
                    middle_file,
                    nn_output_tile,
                    int(middle_file.split("_")[-1].split(".")[0]),
                    [2],
                    out_zonals_path
                )
                middle_zonals_files.append(out_zonals_path)
            except Exception as e:
                print(f"Error processing zonals for middle class {middle_file}: {e}")
            # print(f"Zonal statistics for middle class processed in {time.time() - start_time} seconds for batch {batch_id}")

        for ground_file in ground_files:
            start_time = time.time()
            try:
                out_zonals_path = os.path.join(temp_path, f'{batch_id}_zonal_stats_ground_{ground_file.split("_")[-1].split(".")[0]}.gpkg')
                # (column_prefix, input_vector_layer,input_raster_layer, raster_band , statistics, output_path, verbose=True)
                tools.calculate_zonal_statitics(
                    'zs_',
                    ground_file,
                    nn_output_tile,
                    int(ground_file.split("_")[-1].split(".")[0]),
                    [2],
                    out_zonals_path
                )
                ground_zonals_files.append(out_zonals_path)
            except Exception as e:
                print(f"Error processing zonals for ground class {ground_file}: {e}")
            # print(f"Zonal statistics for ground class processed in {time.time() - start_time} seconds for batch {batch_id}")

        merged_upper_zonals_path = os.path.join(temp_path, f'{batch_id}_merged_zonal_stats_upper.gpkg')
        merged_middle_zonals_path = os.path.join(temp_path, f'{batch_id}_merged_zonal_stats_middle.gpkg')
        merged_ground_zonals_path = os.path.join(temp_path, f'{batch_id}_merged_zonal_stats_ground.gpkg')

        start_time = time.time()
        tools.merge_vector_layers(upper_zonals_files, merged_upper_zonals_path)
        tools.merge_vector_layers(middle_zonals_files, merged_middle_zonals_path)
        tools.merge_vector_layers(ground_zonals_files, merged_ground_zonals_path)
        # print(f"Merged zonal statistics in {time.time() - start_time} seconds for batch {batch_id}")

        upper_raster_path = os.path.join(temp_path, f'{batch_id}_upper_raster.tif')
        middle_raster_path = os.path.join(temp_path, f'{batch_id}_middle_raster.tif')
        ground_raster_path = os.path.join(temp_path, f'{batch_id}_ground_raster.tif')

        start_time = time.time()
        upper_rasterise_cmd = f"""
            gdal_rasterize \
            -l {os.path.basename(merged_upper_zonals_path).split('.')[0]} \
            -a zs_mean \
            -tr 10.0 10.0 \
            -a_nodata -9999.0 \
            -te {batch_grid_bounds[0] - 10} {batch_grid_bounds[1] - 10} {batch_grid_bounds[2] + 10} {batch_grid_bounds[3] + 10} \
            -ot Float32 \
            -of GTiff \
            -co COMPRESS=ZSTD \
            -co ZSTD_LEVEL=9 \
            -co TILED=YES \
            -co BIGTIFF=YES \
            -co PREDICTOR=1 \
            {merged_upper_zonals_path} \
            {upper_raster_path} \
        """
        # print(upper_rasterise_cmd)
        os.system(upper_rasterise_cmd)
        # print(f"Rasterised upper zonals in {time.time() - start_time} seconds for batch {batch_id}")

        start_time = time.time()
        middle_rasterise_cmd = f"""
            gdal_rasterize \
            -l {os.path.basename(merged_middle_zonals_path).split('.')[0]} \
            -a zs_mean \
            -tr 10.0 10.0 \
            -a_nodata -9999.0 \
            -te {batch_grid_bounds[0] - 10} {batch_grid_bounds[1] - 10} {batch_grid_bounds[2] + 10} {batch_grid_bounds[3] + 10} \
            -ot Float32 \
            -of GTiff \
            -co COMPRESS=ZSTD \
            -co ZSTD_LEVEL=9 \
            -co TILED=YES \
            -co BIGTIFF=YES \
            -co PREDICTOR=1 \
            {merged_middle_zonals_path} \
            {middle_raster_path} \
        """
        # print(middle_rasterise_cmd)
        os.system(middle_rasterise_cmd)
        # print(f"Rasterised middle zonals in {time.time() - start_time} seconds for batch {batch_id}")

        start_time = time.time()
        ground_rasterise_cmd = f"""
            gdal_rasterize \
            -l {os.path.basename(merged_ground_zonals_path).split('.')[0]} \
            -a zs_mean \
            -tr 10.0 10.0 \
            -a_nodata -9999.0 \
            -te {batch_grid_bounds[0] - 10} {batch_grid_bounds[1] - 10} {batch_grid_bounds[2] + 10} {batch_grid_bounds[3] + 10} \
            -ot Float32 \
            -of GTiff \
            -co COMPRESS=ZSTD \
            -co ZSTD_LEVEL=9 \
            -co TILED=YES \
            -co BIGTIFF=YES \
            -co PREDICTOR=1 \
            {merged_ground_zonals_path} \
            {ground_raster_path} \
        """
        # print(ground_rasterise_cmd)
        os.system(ground_rasterise_cmd)
        # print(f"Rasterised ground zonals in {time.time() - start_time} seconds for batch {batch_id}")

        # delete temporary files that are not tifs
        temp_files = glob(os.path.join(temp_path, f'{batch_id}*'))
        for temp_file in temp_files:
            if not temp_file.endswith('.tif'):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Error deleting temporary file {temp_file}: {e}")

def split_nn_outputs(nn_outputs, batch_grid_gdf, temp_path):
    """
    Split the NN outputs into batches based on the batch grid.
    """
    grid_gdf = gpd.read_file(batch_grid_gdf)

    with rasterio.open(nn_outputs) as src:
        raster_crs = src.crs

        reprojected_grid_gdf = grid_gdf.to_crs(3857)

        for idx, row in tqdm(reprojected_grid_gdf.iterrows(), total=len(reprojected_grid_gdf), desc='Splitting NN Outputs'):
            bounds = row.geometry.buffer(20).bounds
            batch_id_bounds = grid_gdf.iloc[idx].geometry.bounds
            batch_id = str(int(batch_id_bounds[0])) + '_' + str(int(batch_id_bounds[1]))
            if batch_id == '754000_6036000':
                print(batch_id)
            out_name = os.path.join(temp_path, f"{batch_id}_nn_tile.tif")
            if os.path.exists(out_name):
                continue
            try:
                window = from_bounds(*bounds, transform=src.transform)
                window = window.intersection(src.window(*src.bounds))

                if window.width == 0 or window.height == 0:
                    print(f"Skipping empty window for bounds {bounds}")
                    continue

                out_transform = src.window_transform(window)
                out_image = src.read(window=window)

                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": CRS.from_epsg(3857),
                    "driver": "GTiff",
                    "compress": "ZSTD",
                    "zstd_level": 9,
                    "tiled": True,
                    "BIGTIFF": "YES",
                    "predictor": 1
                })

                with rasterio.open(out_name, "w", **out_meta) as dest:
                    dest.write(out_image)
            except Exception as e:
                print(f"Error processing bounds {bounds} for batch {batch_id}: {e}")

def split_nvis_polygons(nvis_polygon, batch_grid_gdf, temp_path):
    """
    Split the NVIS polygons based on the batch grid.
    """
    nvis_gdf = gpd.read_file(nvis_polygon)
    batch_grid_gdf = gpd.read_file(batch_grid_gdf)

    # Reproject to a common CRS if necessary
    if nvis_gdf.crs != batch_grid_gdf.crs:
        nvis_gdf = nvis_gdf.to_crs(batch_grid_gdf.crs)

    for idx, row in tqdm(batch_grid_gdf.iterrows(), total=len(batch_grid_gdf), desc='Splitting NVIS Polygons'):
        bounds = row.geometry.bounds
        batch_id_bounds = batch_grid_gdf.iloc[idx].geometry.bounds
        batch_id = str(int(batch_id_bounds[0])) + '_' + str(int(batch_id_bounds[1]))
        overlay_polygons_filename = os.path.join(temp_path, f'{batch_id}_clipped_nvis.gpkg')
        if os.path.exists(overlay_polygons_filename):
            continue
        
        try:
            # overlay_polygons = nvis_gdf.overlay(row)
            # overlay_polygons = nvis_gdf[nvis_gdf.intersects(row.geometry)]
            overlay_polygons = nvis_gdf.clip(row.geometry.buffer(20))
            if not overlay_polygons.empty:
                overlay_polygons.to_file(overlay_polygons_filename, driver='GPKG')
            else:
                print(f"No NVIS polygons found for batch {batch_id}")
        except Exception as e:
            print(f"Error processing bounds {bounds} for batch {batch_id}: {e}")

def merge_outputs(temp_path, output_path, output_name):
    os.makedirs(os.path.join(output_path, 'clipped_nvis'), exist_ok=True)
    # gdal_merge.py -o /mnt/datapool1/datapool1/datasets/nvis_outputs/KI2020/KI2020_merged.tif -co COMPRESS=ZSTD -co ZSTD_LEVEL=9 -co TILED=YES -co BIGTIFF=YES -co PREDICTOR=1 /mnt/datapool1/datapool1/datasets/nvis_outputs/KI2020/clipped_mosaic/*.tif
    merge_cmd = f"""
        gdal_merge.py \
        -o {output_path}/clipped_nvis/{output_name}_upper_merged.tif \
        -co COMPRESS=ZSTD \
        -co ZSTD_LEVEL=9 \
        -co TILED=YES \
        -co BIGTIFF=YES \
        -co PREDICTOR=1 \
        {temp_path}/*_upper_raster.tif
    """
    print(merge_cmd)
    os.system(merge_cmd)
    merge_cmd = f"""
        gdal_merge.py \
        -o {output_path}/clipped_nvis/{output_name}_middle_merged.tif \
        -co COMPRESS=ZSTD \
        -co ZSTD_LEVEL=9 \
        -co TILED=YES \
        -co BIGTIFF=YES \
        -co PREDICTOR=1 \
        {temp_path}/*_middle_raster.tif
    """
    print(merge_cmd)
    os.system(merge_cmd)
    merge_cmd = f"""
        gdal_merge.py \
        -o {output_path}/clipped_nvis/{output_name}_ground_merged.tif \
        -co COMPRESS=ZSTD \
        -co ZSTD_LEVEL=9 \
        -co TILED=YES \
        -co BIGTIFF=YES \
        -co PREDICTOR=1 \
        {temp_path}/*_ground_raster.tif
    """
    print(merge_cmd)
    os.system(merge_cmd)

    # gdaladdo -r average --config COMPRESS_OVERVIEW ZSTD --config ZSTD_LEVEL_OVERVIEW 9 --config TILED_OVERVIEW YES --config PREDICTOR_OVERVIEW 1 --config BIGTIFF_OVERVIEW YES /mnt/datapool1/datapool1/datasets/nvis_outputs/KI2020/KI2020_merged_pyramids.tif 2 4 8 16 32 64 128 256
    pyramids_cmd = f"""
        gdaladdo \
        -r average \
        --config COMPRESS_OVERVIEW ZSTD \
        --config ZSTD_LEVEL_OVERVIEW 9 \
        --config TILED_OVERVIEW YES \
        --config PREDICTOR_OVERVIEW 1 \
        --config BIGTIFF_OVERVIEW YES \
        {output_path}/clipped_nvis/{output_name}_upper_merged.tif \
        2 4 8 16 32 64 128 256
    """
    print(pyramids_cmd)
    os.system(pyramids_cmd)
    pyramids_cmd = f"""
        gdaladdo \
        -r average \
        --config COMPRESS_OVERVIEW ZSTD \
        --config ZSTD_LEVEL_OVERVIEW 9 \
        --config TILED_OVERVIEW YES \
        --config PREDICTOR_OVERVIEW 1 \
        --config BIGTIFF_OVERVIEW YES \
        {output_path}/clipped_nvis/{output_name}_middle_merged.tif \
        2 4 8 16 32 64 128 256
    """
    print(pyramids_cmd)
    os.system(pyramids_cmd)
    pyramids_cmd = f"""
        gdaladdo \
        -r average \
        --config COMPRESS_OVERVIEW ZSTD \
        --config ZSTD_LEVEL_OVERVIEW 9 \
        --config TILED_OVERVIEW YES \
        --config PREDICTOR_OVERVIEW 1 \
        --config BIGTIFF_OVERVIEW YES \
        {output_path}/clipped_nvis/{output_name}_ground_merged.tif \
        2 4 8 16 32 64 128 256
    """
    print(pyramids_cmd)
    os.system(pyramids_cmd)

# temp_name = 'temp'
temp_name = 'temp_QLD'
temp_path = os.path.join(outputs_path, temp_name)
os.makedirs(temp_path, exist_ok=True)

grid_size = 5000
batch_grid_file = create_batch_grid(nvis_polygon, grid_size, temp_path)
# batch_grid_file = os.path.join(temp_path, 'batch_grid_5000m.gpkg')

split_nn_outputs(nn_outputs, batch_grid_file, temp_path)

split_nvis_polygons(nvis_polygon, batch_grid_file, temp_path)

nvis_gdf = gpd.read_file(nvis_polygon)
nvis_gdf_dissolved = nvis_gdf.dissolve()
# # Convert iterator to list for parallel processing
batch_grid_gdf = gpd.read_file(batch_grid_file)
batch_list = list(batch_grid_gdf.itertuples())
results = []
with Pool(mp.cpu_count()) as p:
# with Pool(1) as p:
    with tqdm(total=len(batch_list), desc='batch_list', colour='blue', dynamic_ncols=True) as pbar_chips:
        for batch in batch_list:
            batch_grid_bounds = batch.geometry.bounds
            batch_id = str(int(batch_grid_bounds[0])) + '_' + str(int(batch_grid_bounds[1]))
            
            if not os.path.exists(os.path.join(temp_path, f'{batch_id}_upper_raster.tif')) or \
               not os.path.exists(os.path.join(temp_path, f'{batch_id}_middle_raster.tif')) or \
               not os.path.exists(os.path.join(temp_path, f'{batch_id}_ground_raster.tif')):
                # print(batch_id)
                row_dict = batch._asdict() if hasattr(batch, '_asdict') else dict(batch)
                row_dict['geometry_wkt'] = row_dict['geometry'].wkt
                del row_dict['geometry']
                result = p.apply_async(process_batch, args=(row_dict,nvis_gdf_dissolved,), callback=lambda _: pbar_chips.update(1))
                results.append(result)
            # break
        for result in results:
            try:
                result.get()  # This will raise any exception that occurred
            except Exception as e:
                print(f"Error in process_batch: {e}")
                import traceback
                traceback.print_exc()

    p.close()
    p.join()

merge_outputs(temp_path, outputs_path, output_name)

# sys.exit(0)


