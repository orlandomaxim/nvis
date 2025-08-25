import geopandas as gpd

working_dir = '/mnt/BAMspace3/2_project_specific/client_work/NVIS/NVIS_V7_0_VECTOR_STATE_FILES_EXT_ALL'

nvis_detail_lut_gdf = gpd.read_file(working_dir + '/NVIS_V7_0_VECTOR_STATE_FILES_EXT.gdb', layer='NVIS7_0_LUT_AUST_DETAIL')
nvis_flat_lut_gdf = gpd.read_file(working_dir + '/NVIS_V7_0_VECTOR_STATE_FILES_EXT.gdb', layer='NVIS7_0_LUT_AUST_FLAT')

print(nvis_detail_lut_gdf)
print(nvis_flat_lut_gdf)

# save as csv
nvis_detail_lut_gdf.to_csv(working_dir + '/NVIS_V7_0_LUT_AUST_DETAIL.csv', index=False)
nvis_flat_lut_gdf.to_csv(working_dir + '/NVIS_V7_0_LUT_AUST_FLAT.csv', index=False)