# %%
import os
from glob import glob
import geopandas as gpd
from multiprocessing import Pool
import multiprocessing as mp

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
# delivery_tile = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2020/KI2020_delivery_tile.geojson'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2020/'
# output_name = 'NVIS_8L_fm_mape_r1_KI2020'

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_WASouth/NVIS_WASouth_7850.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_WASouth/merged/WASouth_merged.tif'
# delivery_tile = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_WASouth/WASouth_delivery_tile.geojson'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_WASouth/'
# output_name = 'NVIS_8L_fm_mape_r1_WASouth'

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2025/nvis_polygon_7853.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2025/merged/NVIS_8L_fm_mape_r1_KI2025_merged.tif'
# delivery_tile = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2025/KI2025_delivery_tile.geojson'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_KI2025/'
# output_name = 'NVIS_8L_fm_mape_r1_KI2025'

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Billabong/NVIS_Billabong_7850.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Billabong/merged/NVIS_8L_fm_mape_r1_Billabong_merged.tif'
# delivery_tile = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Billabong/Billabong_delivery_tile.geojson'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Billabong/'
# output_name = 'NVIS_8L_fm_mape_r1_Billabong'

# nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_MulgaLands/NVIS_Mulga_dissolved.gpkg'
# nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_MulgaLands/merged/NVIS_8L_fm_mape_r1_MulgaLands_merged.tif'
# delivery_tile = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_MulgaLands/MulgaLands_delivery_tile.geojson'
# outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_MulgaLands/'
# output_name = 'NVIS_8L_fm_mape_r1_MulgaLands'

nvis_polygon = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Tenterfield/NVIS_Tenterfield_dissolved.gpkg'
nn_outputs = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Tenterfield/merged/NVIS_8L_fm_mape_r1_Tenterfield_merged.tif'
delivery_tile = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Tenterfield/Tenterfield_delivery_tile.geojson'
outputs_path = '/mnt/datapool1/datapool1/datasets/nvis_outputs/NVIS_8L_fm_mape_r1_Tenterfield/'
output_name = 'NVIS_8L_fm_mape_r1_Tenterfield_QLD'

# %%
os.environ['GDAL_CACHEMAX'] = '2048'  # 2GB cache (adjust based on your RAM)
os.environ['GDAL_NUM_THREADS'] = 'ALL_CPUS'  # Use all CPU cores
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['VSI_CACHE'] = 'TRUE'
os.environ['VSI_CACHE_SIZE'] = '25000000'  # 25MB cache per file
os.environ['GDAL_MAX_DATASET_POOL_SIZE'] = '450'
os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'

from shapely import wkt

height_ranges = {1: '<0.5',
                 2: '0.5-1',
                 3: '1-2',
                 4: '>2',
                 5: '<3',
                 6: '<10',
                 7: '10-30',
                 8: '>30'}

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

def process_batch(row_dict):
    tools = init_qgis()
    row_dict['geometry'] = wkt.loads(row_dict['geometry_wkt'])
    batch_grid = row_dict
    batch_grid_bounds = batch_grid['geometry'].bounds
    
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
    print(f"Batch grid created in {time.time() - start_time} seconds for batch {batch_id}")

    start_time = time.time()
    batch_10m_grid_file = os.path.join(temp_path, f'{batch_id}_batch_grid_10m.gpkg')

    nn_output_tile = os.path.join(temp_path, f'{batch_id}_nn_tile.tif')

    if not os.path.exists(nn_output_tile):
        print(f"NN output tile {nn_output_tile} does not exist, skipping.")
        return None

    for band, height_range in height_ranges.items():
        out_zonals_path = os.path.join(temp_path, f'{batch_id}_zonal_stats_band{band}.gpkg')
        try:
            tools.calculate_zonal_statitics(
                'zs_',
                batch_10m_grid_file,
                nn_output_tile,
                band,
                [2],
                out_zonals_path
            )
        except Exception as e:
            print(f"Error processing zonals for height range {height_range}: {e}")
            continue

        clamped_zonals_path = os.path.join(temp_path, f'{batch_id}_zonal_stats_band{band}_clamped.gpkg')
        out_zonals_gdf = gpd.read_file(out_zonals_path)
        out_zonals_gdf['zs_mean'] = out_zonals_gdf['zs_mean'].clip(0, 1)
        out_zonals_gdf.to_file(clamped_zonals_path)

        band_raster_path = os.path.join(temp_path, f'{batch_id}_band{band}_raster.tif')

        band_rasterise_cmd = f"""
            gdal_rasterize \
            -l {os.path.basename(clamped_zonals_path).split('.')[0]} \
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
            {clamped_zonals_path} \
            {band_raster_path} \
        """
        os.system(band_rasterise_cmd)

    # Merge bands into single tif
    merge_cmd = f"""
        gdal_merge.py \
        -separate \
        -o {temp_path}/{batch_id}_merged.tif \
        -co COMPRESS=ZSTD \
        -co ZSTD_LEVEL=9 \
        -co TILED=YES \
        -co BIGTIFF=YES \
        -co PREDICTOR=1 \
        {temp_path}/{batch_id}_band*_raster.tif
    """
    print(merge_cmd)
    os.system(merge_cmd)

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

def merge_outputs(temp_path, output_path, output_name):
    os.makedirs(os.path.join(output_path, 'separated'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'merged'), exist_ok=True)
    for band, height_range in height_ranges.items():
        merge_cmd = f"""
            gdal_merge.py \
            -o {output_path}/separated/{output_name}_band{band}_merged.tif \
            -co COMPRESS=ZSTD \
            -co ZSTD_LEVEL=9 \
            -co TILED=YES \
            -co BIGTIFF=YES \
            -co PREDICTOR=1 \
            {temp_path}/*_band{band}_raster.tif
        """
        print(merge_cmd)
        os.system(merge_cmd)
        pyramids_cmd = f"""
            gdaladdo \
            -r average \
            --config COMPRESS_OVERVIEW ZSTD \
            --config ZSTD_LEVEL_OVERVIEW 9 \
            --config TILED_OVERVIEW YES \
            --config PREDICTOR_OVERVIEW 1 \
            --config BIGTIFF_OVERVIEW YES \
            {output_path}/separated/{output_name}_band{band}_merged.tif \
            2 4 8 16 32 64 128 256
        """
        print(pyramids_cmd)
        os.system(pyramids_cmd)

    merge_cmd = f"""
        gdal_merge.py \
        -o {output_path}/merged/{output_name}_all_merged.tif \
        -co COMPRESS=ZSTD \
        -co ZSTD_LEVEL=9 \
        -co TILED=YES \
        -co BIGTIFF=YES \
        -co PREDICTOR=1 \
        {temp_path}/*_merged.tif
    """
    print(merge_cmd)
    os.system(merge_cmd)
    pyramids_cmd = f"""
        gdaladdo \
        -r average \
        --config COMPRESS_OVERVIEW ZSTD \
        --config ZSTD_LEVEL_OVERVIEW 9 \
        --config TILED_OVERVIEW YES \
        --config PREDICTOR_OVERVIEW 1 \
        --config BIGTIFF_OVERVIEW YES \
        {output_path}/merged/{output_name}_all_merged.tif \
        2 4 8 16 32 64 128 256
    """
    print(pyramids_cmd)
    os.system(pyramids_cmd)

def clip_outputs(temp_path, output_path, clipping_polygon, epsg=9473):
    os.makedirs(os.path.join(output_path, 'merged_clipped'), exist_ok=True)
    clip_cmd = f"""
        gdalwarp \
        -cutline {clipping_polygon} \
        -crop_to_cutline \
        -dstnodata -9999 \
        -co COMPRESS=ZSTD \
        -co ZSTD_LEVEL=9 \
        -co TILED=YES \
        -co BIGTIFF=YES \
        -co PREDICTOR=1 \
        {output_path}/merged/{output_name}_all_merged.tif \
        {output_path}/merged/{output_name}_all_merged_clipped.tif
    """
    print(clip_cmd)
    os.system(clip_cmd)
    # Reproject to epsg
    # reproject_cmd = f"""
    #     gdalwarp \
    #     -t_srs EPSG:{epsg} \
    #     -r bilinear \
    #     -co COMPRESS=ZSTD \
    #     -co ZSTD_LEVEL=9 \
    #     -co TILED=YES \
    #     -co BIGTIFF=YES \
    #     -co PREDICTOR=1 \
    #     {output_path}/merged/{output_name}_all_merged_clipped.tif \
    #     {output_path}/merged/{output_name}_all_merged_clipped_epsg{epsg}.tif
    # """
    # print(reproject_cmd)
    # os.system(reproject_cmd)
    pyramids_cmd = f"""
        gdaladdo \
        -r average \
        --config COMPRESS_OVERVIEW ZSTD \
        --config ZSTD_LEVEL_OVERVIEW 9 \
        --config TILED_OVERVIEW YES \
        --config PREDICTOR_OVERVIEW 1 \
        --config BIGTIFF_OVERVIEW YES \
        {output_path}/merged/{output_name}_all_merged_clipped.tif \
        2 4 8 16 32 64 128 256
    """
    print(pyramids_cmd)
    os.system(pyramids_cmd)

temp_name = 'temp'
# temp_name = 'temp_QLD'
temp_path = os.path.join(outputs_path, temp_name)
os.makedirs(temp_path, exist_ok=True)

grid_size = 5000
batch_grid_file = create_batch_grid(nvis_polygon, grid_size, temp_path)

split_nn_outputs(nn_outputs, batch_grid_file, temp_path)

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
                result = p.apply_async(process_batch, args=(row_dict,), callback=lambda _: pbar_chips.update(1))
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

clip_outputs(temp_path, outputs_path, delivery_tile)