# %%
from glob import glob
import rasterio
from rasterio.crs import CRS
import os
import pandas as pd
pd.set_option('display.max_colwidth', None)
import geopandas as gpd
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import re

# %%
delivery_name = 'Kangaroo_Island2020'
# # delivery_name = 'KangarooIsland_AdditionalScene'
outputs = 'NVIS_8L_fm_mape_r1_KI2020'
id = 'KI2020'

# delivery_name = 'KangarooIsland_2025'
# outputs = 'NVIS_8L_fm_mape_r1_KI2025'
# id = 'KI2025'

# delivery_name = 'WA_2025'
# outputs = 'NVIS_8L_fm_mape_r1_WASouth'
# id = 'WASouth'

# delivery_name = 'Billabong'
# outputs = 'NVIS_8L_fm_mape_r1_Billabong'
# id = 'Billabong'

# delivery_name = 'MulgaLands'
# outputs = 'NVIS_8L_fm_mape_r1_MulgaLands'
# id = 'MulgaLands'

# delivery_name = 'Tenterfield'
# outputs = 'NVIS_8L_fm_mape_r1_Tenterfield'
# id = 'Tenterfield'

nn_outputs_path = f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/raw'
polygon_path = f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/clipping_polygons'
output_path = f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/clipped'
merged_path = f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/merged'
processing_summary = glob(f'/mnt/datapool2/Archive/EO_IMAGERY/raw/aoi/{delivery_name}/**/01_Raw/processing_summary.csv', recursive=True)[0]

# polygons = glob(f'/mnt/datapool1/datapool1/datasets/nn_datasets/polygons/{id}/*.geojson')
# nn_datasets = glob('/mnt/datapool1/datapool1/datasets/nn_datasets/NVIS/**/*.hdf5', recursive=True)

def get_geospatial_metadata(merged_tif):
    metadata = {}

    with rasterio.open(merged_tif) as src:
        metadata['driver'] = src.driver
        metadata['count'] = src.count
        metadata['width'] = src.width
        metadata['height'] = src.height
        metadata['dtype'] = src.dtypes[0]
        metadata['nodata'] = src.nodata

        metadata['crs'] = src.crs.to_string()

        metadata['bounds'] = {
            'left': src.bounds.left,
            'bottom': src.bounds.bottom,
            'right': src.bounds.right,
            'top': src.bounds.top
        }

        metadata['resolution'] = {
            'x': abs(src.res[0]),
            'y': abs(src.res[1])
        }

        metadata['transform'] = list(src.transform)

        width_meters = (src.bounds.right - src.bounds.left)
        height_meters = (src.bounds.top - src.bounds.bottom)
        metadata['spatial_extent'] = {
            'width_meters': width_meters,
            'height_meters': height_meters,
            'area_square_meters': width_meters * height_meters
        }

        metadata['bands'] = []
        for band_num in range(1, src.count + 1):
            band_data = src.read(band_num, masked=True)
            
            # Calculate statistics, ignoring nodata values
            band_info = {
                'band_number': band_num,
                'height_range': {
                    1: 'below_0.5m',
                    2: '0.5-1m', 
                    3: '1-2m',
                    4: 'above_2m',
                    5: 'below_3m',
                    6: 'below_10m',
                    7: '10-30m',
                    8: 'above_30m'
                }.get(band_num, f'band_{band_num}'),
                'min': float(band_data.min()) if not band_data.mask.all() else None,
                'max': float(band_data.max()) if not band_data.mask.all() else None,
                'mean': float(band_data.mean()) if not band_data.mask.all() else None,
                'std': float(band_data.std()) if not band_data.mask.all() else None,
                'dtype': str(band_data.dtype),
                'valid_pixels': int((~band_data.mask).sum()),
                'total_pixels': int(band_data.size)
            }
            metadata['bands'].append(band_info)

        metadata['units'] = 'na'

        # Additional technical metadata
        metadata['tiled'] = src.is_tiled
        if src.is_tiled:
            metadata['block_size'] = src.block_shapes[0]  # (height, width) of blocks
        
        # Get compression info from profile if available
        try:
            metadata['compression'] = src.compression.name if src.compression else None
        except:
            metadata['compression'] = None

    return metadata

def get_height_geospatial_metadata(merged_tif):
    metadata = {}

    with rasterio.open(merged_tif) as src:
        metadata['driver'] = src.driver
        metadata['count'] = src.count
        metadata['width'] = src.width
        metadata['height'] = src.height
        metadata['dtype'] = src.dtypes[0]
        metadata['nodata'] = src.nodata

        metadata['crs'] = src.crs.to_string()

        metadata['bounds'] = {
            'left': src.bounds.left,
            'bottom': src.bounds.bottom,
            'right': src.bounds.right,
            'top': src.bounds.top
        }

        metadata['resolution'] = {
            'x': abs(src.res[0]),
            'y': abs(src.res[1])
        }

        metadata['transform'] = list(src.transform)

        width_meters = (src.bounds.right - src.bounds.left)
        height_meters = (src.bounds.top - src.bounds.bottom)
        metadata['spatial_extent'] = {
            'width_meters': width_meters,
            'height_meters': height_meters,
            'area_square_meters': width_meters * height_meters
        }

        metadata['bands'] = []
        for band_num in range(1, src.count + 1):
            band_data = src.read(band_num, masked=True)
            
            # Calculate statistics, ignoring nodata values
            band_info = {
                'band_number': band_num,
                'height_range': {
                    0: '0m',
                    1: 'below_0.5m', 
                    2: '0.5m-1m',
                    3: '1m-2m',
                    4: '2m-10m',
                    5: '3m-10m',
                    6: '10m-30m',
                    7: 'above_30m'
                },
                'dtype': str(band_data.dtype),
                'total_pixels': int(band_data.size)
            }
            metadata['bands'].append(band_info)

        metadata['units'] = 'na'

        # Additional technical metadata
        metadata['tiled'] = src.is_tiled
        if src.is_tiled:
            metadata['block_size'] = src.block_shapes[0]  # (height, width) of blocks
        
        # Get compression info from profile if available
        try:
            metadata['compression'] = src.compression.name if src.compression else None
        except:
            metadata['compression'] = None

    return metadata

def get_polygon_area(file, decimals=6, clip_to=None, dissolve=False):
    """Calculate area of polygons in hectares, optionally clipped to a boundary
    
    Args:
        file (str): Path to vector file or geodataframe
        decimals (int): Number of decimal places to round to
        clip_to (GeoDataFrame, optional): GeoDataFrame to clip geometries to
        
    Returns:
        str: Area in hectares as string
    """
    if isinstance(file, str):
        # Check if file exists
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist.")
        gdf = gpd.read_file(file)
    elif isinstance(file, gpd.GeoDataFrame):
        gdf = file
    else:
        raise ValueError("Input must be a file path or a GeoDataFrame.")
    # gdf['geometry'] = gdf['geometry'].apply(lambda geom: geometrycollection_to_multipolygon(geom))
    if clip_to is not None:
        # Handle clip_to as either string file path or GeoDataFrame
        if isinstance(clip_to, str):
            clip_gdf = gpd.read_file(clip_to)
        else:
            clip_gdf = clip_to

        # Ensure same CRS
        if gdf.crs != clip_gdf.crs:
            clip_gdf = clip_gdf.to_crs(gdf.crs)
        
        # Perform clip
        gdf = gpd.clip(gdf, clip_gdf)

    if dissolve:
        gdf = gdf.dissolve()
    return round(gdf.geometry.area.sum() / 10000, decimals)

def get_train_test_areas(train_test_polygons):
    """
    Get the train and test areas from the train_test_polygons files
    """
    train_test_areas = {'train_area': 0, 'test_area': 0}
    for file in train_test_polygons:
        file_gdf = gpd.read_file(file)
        train_gdf = file_gdf[file_gdf['train'] == 1]
        test_gdf = file_gdf[file_gdf['test'] == 1]
        train_test_areas['train_area'] += get_polygon_area(train_gdf)
        train_test_areas['test_area'] += get_polygon_area(test_gdf)
    print(f"Train area: {train_test_areas['train_area']} ha")
    print(f"Test area: {train_test_areas['test_area']} ha")
    print(f"Train percentage: {train_test_areas['train_area'] / (train_test_areas['train_area'] + train_test_areas['test_area']) * 100:.2f}%")
    print(f"Test percentage: {train_test_areas['test_area'] / (train_test_areas['train_area'] + train_test_areas['test_area']) * 100:.2f}%")
    return train_test_areas

als_areas = {'NVIS_8L_fm_mape_r1_KI2020':       [{'date': '30/01/2020', 'area': '2536', 'sensor': 'RIEGL Q680i-S'},
                                                 {'date': '07/02/2020', 'area': '2048', 'sensor': 'RIEGL Q680i-S'},
                                                 {'date': '23/02/2020', 'area': '599', 'sensor': 'RIEGL Q680i-S'},
                                                 {'date': '13/04/2020', 'area': '1985', 'sensor': 'RIEGL Q680i-S'},
                                                 {'date': '14/04/2020', 'area': '3782', 'sensor': 'RIEGL Q680i-S'},
                                                 {'date': '10/09/2020', 'area': '166', 'sensor': 'RIEGL Q680i-S'},
                                                 {'date': '12/09/2020', 'area': '2822', 'sensor': 'RIEGL Q680i-S'}],
             'NVIS_8L_fm_mape_r1_KI2025':       [{'date': '18/06/2025', 'area': '20903', 'sensor': 'RIEGL VUX 160-23'}],
             'NVIS_8L_fm_mape_r1_WASouth':      [{'date': '20/04/2024', 'area': '18659', 'sensor': 'RIEGL VUX 160-23'},
                                                 {'date': '21/04/2024', 'area': '5409', 'sensor': 'RIEGL VUX 160-23'}],
             'NVIS_8L_fm_mape_r1_Billabong':    [{'date': '15/04/2024', 'area': '16153', 'sensor': 'RIEGL VUX 160-23'}],
             'NVIS_8L_fm_mape_r1_MulgaLands':   [{'date': '25/04/2023', 'area': '9518', 'sensor': 'RIEGL Q680i-S'}],
             'NVIS_8L_fm_mape_r1_Tenterfield':  [{'date': '22/05/2025', 'area': '21576', 'sensor': 'RIEGL VUX 160-23'},
                                                 {'date': '23/05/2025', 'area': '21770', 'sensor': 'RIEGL VUX 160-23'}]
}

# %%
def get_als_metadata(outputs, polygons):
    als_metadata = get_train_test_areas(polygons)
    print(als_metadata)
    als_metadata['captures'] = []
    for als_area in als_areas[outputs]:
        als_metadata['captures'].append(als_area)
        # print(als_area)
    # print(als_areas[outputs])
    return als_metadata

# merged_file = glob(os.path.join(merged_path, '*_all_merged_clipped.tif'))[0]
# print(merged_file)

# als_metadata_dict = get_als_metadata(outputs, polygons)
# print(als_metadata_dict)

# geospatial_metadata_dict = get_geospatial_metadata(merged_file)
# print(geospatial_metadata_dict)

# clipping_polygons = glob(polygon_path + '/*.geojson')
# clipping_polygon_names = [os.path.splitext(os.path.basename(f))[0].replace(f'{delivery_name}_', '').replace('KangarooIsland_AdditionalScene_', '') for f in clipping_polygons]
# imagery_metadata_dict = {}
# imagery_metadata_dict['mosaic_tiles'] = clipping_polygon_names
# print(imagery_metadata_dict)

# metadata_dict = {
#     'geospatial_metadata': geospatial_metadata_dict,
#     'als_metadata': als_metadata_dict,
#     'imagery_metadata': imagery_metadata_dict
# }
# xml_bytes = dicttoxml(metadata_dict, custom_root='MetaData', attr_type=False)
# xml_string = parseString(xml_bytes).toprettyxml()
# print(xml_string)
# with open(f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/{id}_metadata.xml', 'w') as f:
#     f.write(xml_string)

# merged_files = glob(os.path.join(merged_path, '*_height_merged_clipped.tif'))
merged_files = glob(os.path.join(merged_path, '*_10m_height_sample_merged.tif'))
print(merged_files)

# als_metadata_dict = get_als_metadata(outputs, polygons)
# print(als_metadata_dict)

for merged_file in merged_files:
    geospatial_metadata_dict = get_height_geospatial_metadata(merged_file)
    print(geospatial_metadata_dict)
    # cover_cutoff = os.path.basename(merged_file).replace(outputs + '_', '').replace('_height_merged_clipped.tif', '')
    cover_cutoff = os.path.basename(merged_file).replace(outputs + '_', '').replace('_10m_height_sample_merged.tif', '')

    clipping_polygons = glob(polygon_path + '/*.geojson')
    clipping_polygon_names = [os.path.splitext(os.path.basename(f))[0].replace(f'{delivery_name}_', '').replace('KangarooIsland_AdditionalScene_', '') for f in clipping_polygons]
    # imagery_metadata_dict = {}
    # imagery_metadata_dict['mosaic_tiles'] = clipping_polygon_names
    # print(imagery_metadata_dict)

    metadata_dict = {
        'cover_cutoff': cover_cutoff,
        'geospatial_metadata': geospatial_metadata_dict,
        # 'als_metadata': als_metadata_dict,
        # 'imagery_metadata': imagery_metadata_dict
    }
    xml_bytes = dicttoxml(metadata_dict, custom_root='MetaData', attr_type=False)
    xml_string = parseString(xml_bytes).toprettyxml()
    print(xml_string)
    with open(f'/mnt/datapool1/datapool1/datasets/nvis_outputs/{outputs}/{os.path.splitext(os.path.basename(merged_file))[0]}_metadata.xml', 'w') as f:
        f.write(xml_string)