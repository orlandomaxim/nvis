import os
from glob import glob
import geopandas as gpd

def merge_clipped_tiles(input_folder, output_file, pyramids=False):
    """
    Merges all clipped tiles in the input folder into a single GeoTIFF file.
    """
    # Get all clipped files
    clipped_files = glob(os.path.join(input_folder, '*_clipped.tif'))
    
    if not clipped_files:
        print("No clipped files found.")
        return
    
    # Use gdal_merge.py to merge the files
    cmd = f'gdal_merge.py -o {output_file} -co COMPRESS=ZSTD -co ZSTD_LEVEL=9 -co TILED=YES -co BIGTIFF=YES -co PREDICTOR=1 ' + ' '.join(clipped_files)
    print(f'Merging {len(clipped_files)} files into {output_file}...')
    os.system(cmd)

    if pyramids:
        # Create overviews for the merged file
        overview_cmd = f'gdaladdo -r average --config COMPRESS_OVERVIEW ZSTD --config ZSTD_LEVEL_OVERVIEW 9 --config TILED_OVERVIEW YES --config PREDICTOR_OVERVIEW 1 --config BIGTIFF_OVERVIEW YES {output_file} 2 4 8 16 32 64 128 256'
        print(f'Creating overviews for {output_file}...')
        os.system(overview_cmd)

if __name__=='__main__':
    # outputs = 'KI2020_mape_v2'
    # outputs = 'WA_2025'
    # delivery_name = 'WA_2025'

    delivery_name = 'Kangaroo_Island2020'
    # delivery_name = 'KangarooIsland_AdditionalScene'
    outputs = 'NVIS_8L_fm_mape_r1_KI2020'

    # delivery_name = 'KangarooIsland_2025'
    # outputs = 'NVIS_8L_fm_mape_r1_KI2025'

    # delivery_name = 'Billabong'
    # outputs = 'NVIS_8L_fm_mape_r1_Billabong'

    # delivery_name = 'MulgaLands'
    # outputs = 'NVIS_8L_fm_mape_r1_MulgaLands'

    # delivery_name = 'Tenterfield'
    # outputs = 'NVIS_8L_fm_mape_r1_Tenterfield'
    
    # outputs = 'WASouth'
    # delivery_name = 'WA_2025'
    nn_outputs_path = f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/raw'
    polygon_path = f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/clipping_polygons'
    output_path = f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/clipped'
    processing_summary = glob(f'/mnt/datapool2/Archive/EO_IMAGERY/raw/aoi/{delivery_name}/**/01_Raw/processing_summary.csv', recursive=True)[0]
    # processing_summary = glob(f'/mnt/datapool2/Archive/EO_IMAGERY/raw/aoi/{delivery_name}/01_Raw/processing_summary.csv', recursive=True)[0]
    print(processing_summary)

    name_map = {}
    with open(processing_summary, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            raw_name, ortho_name = line.strip().split(',')
            ortho_name = ortho_name.strip().replace('.tif', '')
            raw_name = raw_name.strip().replace('.tif', '')
            name_map[ortho_name] = raw_name

    print(name_map)

    os.makedirs(output_path, exist_ok=True)

    nn_output_files = glob(os.path.join(nn_outputs_path, '*.tif'))
    polygon_files = glob(os.path.join(polygon_path, '*.geojson'))

    temp_path = os.path.join(output_path, 'temp')
    os.makedirs(temp_path, exist_ok=True)

    # print(output_files)
    print(polygon_files)
    for nn_output_file in nn_output_files:
        print(nn_output_file)

        image_name = os.path.basename(nn_output_file).split('.')[0].replace(delivery_name + '_', '').replace('_Merged', '').replace('COMBINED', 'MSS')
        # print(image_name)
        # if image_name in name_map:
        #     image_name = name_map[image_name]
        poly_path = os.path.join(polygon_path, delivery_name + '_' + image_name + '.geojson')
        # poly_path = os.path.join(polygon_path, image_name + '.geojson')
        if not os.path.exists(poly_path):
            print(f'Polygon file {poly_path} does not exist, skipping.')
            continue
        print(f'Processing {nn_output_file} with polygon {poly_path}')
        # Clop raster using gdal
        output_basename = os.path.basename(nn_output_file).split('.')[0]
        output_clipped_path = os.path.join(output_path, f'{output_basename}_clipped.tif')
        temp_clipped_path = os.path.join(temp_path, f'{output_basename}_clipped.tif')
        if os.path.exists(output_clipped_path):
            print(f'Output file {output_clipped_path} already exists, skipping clipping.')
            continue

        if os.path.exists(temp_clipped_path):
            print(f'Temporary file {temp_clipped_path} already exists, removing it.')
            os.remove(temp_clipped_path)

        print(f'Clipping {nn_output_file} to {output_clipped_path} using polygon {poly_path}')

        # gdalwarp -overwrite -of GTiff -cutline //192.168.11.30/datapool1/datasets/cgsat/polygons/KangarooIsland_2020_JL1KF01A_0034_20201211_1325657_MSS_Ortho.geojson -cl None -crop_to_cutline -dstnodata -9999.0 -co COMPRESS=ZSTD -co ZSTD_LEVEL=9 -co TILED=YES -co BIGTIFF=YES -co PREDICTOR=1 C:/Users/AM/Documents/misc/KangarooIsland_2020_JL1GF02A_0006_20201126_1325396_COMBINED_Ortho_Merged_Clipped.tif C:/Users/AM/Documents/misc/KangarooIsland_2020_JL1GF02A_0006_20201126_1325396_COMBINED_Ortho_Merged_Clipped.tif
        cmd = f'gdalwarp -overwrite -of GTiff -cutline {poly_path} -crop_to_cutline -dstnodata -9999.0 -co COMPRESS=ZSTD -co ZSTD_LEVEL=9 -co TILED=YES -co BIGTIFF=YES -co PREDICTOR=1 {nn_output_file} {temp_clipped_path}'
        # print(f'Executing command: {cmd}')
        os.system(cmd)
        os.rename(temp_clipped_path, output_clipped_path)

    merged_output_path = f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/merged'
    os.makedirs(merged_output_path, exist_ok=True)
    merge_clipped_tiles(output_path, os.path.join(merged_output_path, f'{outputs}_merged.tif'), pyramids=True)


