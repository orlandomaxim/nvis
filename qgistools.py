import os
import sys
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

from qgis.core import *
import sys
# from processing.core.Processing import Processing
from osgeo import gdal, ogr
import os
from typing import  List
# sys.path.append('/usr/share/qgis/python')
# sys.path.append('/usr/share/qgis/python/plugins')
import processing


# from processing.core.Processing import Processing



gdal.UseExceptions()
ogr.UseExceptions()

class QGISTools:
    """
    A class that provides wrappers for common QGIS processing tools.
    """

    @staticmethod
    def gdal_cliprasterbymasklayer(input_raster, mask_layer, output_raster, no_data_value=None, crop_to_cutline=True, target_crs=None, **args):
        """
        Clips a raster by a mask layer using the GDAL 'cliprasterbymasklayer' function in QGIS processing.

        :param input_raster: Path to the input raster file.
        :param mask_layer: Path to the mask layer (vector) file.
        :param output_raster: Path to the output clipped raster file.
        :param crop_to_cutline: Boolean, whether to crop the raster to the exact extent of the mask layer.
        :param no_data_value: Optional, specify a NoData value to fill areas outside the mask.
        """
        # Parameters for the GDAL clip raster by mask layer
        params = {
            'INPUT': input_raster,
            'MASK': mask_layer,
            'OUTPUT': output_raster,
            'CROP_TO_CUTLINE': crop_to_cutline,
            'NODATA': no_data_value
        }
        print(output_raster)
        # Run the processing algorithm
        results = processing.run('gdal:cliprasterbymasklayer', params)
        return results

    @staticmethod
    def polygonize(input_raster, output_vector):
        """
        Get the extent of a raster as a polygon using QGIS 'Extract Layer Extent' function.
        
        :param input_raster: Path to the input raster file.
        :param output_vector: Path to save the output polygon vector file.
        """
        params = {
            'INPUT': input_raster,
            'OUTPUT': output_vector
        }

        # Run the processing algorithm
        processing.run('native:polygonize', params)

    @staticmethod
    def reproject_raster_layer(input_raster, output_raster,target_crs):
        params = {
            'INPUT': input_raster,
            'TARGET_CRS': target_crs,  # Reproject to WGS 84 as a test
            'OUTPUT': output_raster
        }
        processing.run('gdal:warpreproject', params)

    @staticmethod
    def extract_raster_extent(input_raster, output_geojson):
        """
        Extracts the extent of a raster as a polygon using QGIS 'Polygon from Layer Extent' tool.
        
        :param input_raster: Path to the input raster file.
        :param output_geojson: Path to save the output vector (GeoJSON) file.
        """
        params = {
            'INPUT': input_raster,
            'OUTPUT': output_geojson
        }

        try:
            # Run the processing algorithm
            processing.run('native:polygonfromlayerextent', params)
            print(f"Extent polygon saved to: {output_geojson}")
        except Exception as e:
            print(f"Error during extract layer extent: {e}")

    @staticmethod
    def vector_difference(input_layer, difference_layer, output_path):
        """
        Perform a difference operation between two vector layers using QGIS native:difference.
        
        :param input_layer: Path to the input vector layer.
        :param difference_layer: Path to the vector layer to subtract from the input layer.
        :param output_path: Path to save the resulting difference layer.
        """
        params = {
            'INPUT': input_layer,
            'OVERLAY': difference_layer,
            'OUTPUT': output_path
        }
        print(params)

        try:
            # Run the native:difference algorithm
            result = processing.run('native:difference', params)
            print(f"Difference operation completed. Output saved to: {output_path}")
            return result
        except Exception as e:
            print(f"Error during difference operation: {e}")

    @staticmethod
    def clip_raster_by_mask(input_raster, mask_layer, output_raster, crop_to_cutline=True, no_data_value=None, **args):
        """
        Clip a raster by a mask layer using GDAL's 'gdal:cliprasterbymasklayer' tool.
        
        :param input_raster: Path to the input raster file.
        :param mask_layer: Path to the vector mask layer (e.g., a shapefile or GeoPackage).
        :param output_raster: Path to save the output clipped raster file.
        :param crop_to_cutline: Boolean indicating whether to crop the raster to the exact mask extent.
        :param no_data_value: Optional NoData value to apply to areas outside the mask.
        """
        params = {
            'INPUT': input_raster,         # Input raster file
            'MASK': mask_layer,            # Vector layer to use as the mask
            'OUTPUT': output_raster,       # Output raster file
            'CROP_TO_CUTLINE': crop_to_cutline,  # Whether to crop the raster to the mask extent
            'NO_DATA': no_data_value if no_data_value is not None else -9999# NoData value
        }

        try: 
            print(params)
            # Run the GDAL algorithm for clipping raster by mask
            processing.run('gdal:cliprasterbymasklayer', params)
            # gdalwarp = f"gdalwarp -overwrite -of GTiff -cutline {mask_layer} -crop_to_cutline -dstnodata {no_data_value} {input_raster} {output_raster}"
            # os.system(gdalwarp)
            print(f"Raster clipped successfully. Output saved to: {output_raster}")
        except Exception as e:
            print(f"Error during raster clipping: {e}")

    @staticmethod
    def raster_calculator(expression, input_rasters, output_raster, extent=None, crs=None):
        """
        Perform raster calculations using the QGIS native raster calculator ('native:rastercalc').

        :param expression: A string containing the mathematical expression for the raster calculation.
                        Use 'A', 'B', 'C', etc., to refer to the input rasters in the expression.
        :param input_rasters: A list of paths to the input raster files. The order should match the variables used in the expression.
        :param output_raster: Path to save the resulting raster file.
        :param extent: Optional. Define the extent for the output raster (e.g., 'xmin, xmax, ymin, ymax').
        :param crs: Optional. Define the CRS (coordinate reference system) for the output raster (e.g., 'EPSG:4326').
        """
        params = {
            'EXPRESSION': expression,  # The calculation expression
            'LAYERS': [{'layer': r, 'name': chr(65 + idx)} for idx, r in enumerate(input_rasters)],  # 'A', 'B', 'C', etc.
            'CELLSIZE': 0,             # Optional. Set to 0 to use the default resolution of the input rasters.
            'EXTENT': extent if extent else None,  # Optional. If not provided, the extent of the first layer is used.
            'CRS': crs if crs else None,  # Optional. If not provided, the CRS of the first raster is used.
            'OUTPUT': output_raster    # Path to the output raster file
        }

        try:
            # Run the QGIS native raster calculator
            processing.run('native:rastercalc', params)
            print(f"Raster calculation completed. Output saved to: {output_raster}")
        except Exception as e:
            print(f"Error during raster calculation: {e}")

    @staticmethod
    def weighted_average_raster(input_rasters, weights, output_raster):
        """
        Merge multiple rasters into a single output raster using a weighted average.

        :param input_rasters: List of paths to the input raster files (5 rasters in this case).
        :param weights: List of weights corresponding to the input rasters (must be the same length as input_rasters).
        :param output_raster: Path to save the resulting weighted average raster.
        """
        if len(input_rasters) != len(weights):
            raise ValueError("The number of input rasters must match the number of weights.")
        
        # Check if all raster files exist
        for idx, raster in enumerate(input_rasters):
            if not os.path.exists(raster):
                raise FileNotFoundError(f"Raster file not found: {raster}")
            else:
                print(f"Raster {chr(65 + idx)} loaded: {raster}")

        # Construct the expression for the weighted average, e.g., '0.2*A + 0.3*B + ...'
        # expression = " + ".join([f"{weights[i]} * A{i + 1}" for i in range(len(input_rasters))])
        expression = " + ".join([f"{weights[i]} * {chr(65 + i)}" for i in range(len(input_rasters))])
        print(expression)

        # Create the layer mapping for each input raster (A, B, C, etc.)
        # layers = [{'layer': r, 'name': f"A{i + 1}"} for i, r in enumerate(input_rasters)]
        layers = [{'layer': r, 'name': chr(65 + idx)} for idx, r in enumerate(input_rasters)]
        print(layers)

        # Set up parameters for the raster calculator
        # params = {
        #     'EXPRESSION': expression,  # The weighted average expression
        #     'LAYERS': layers,          # The input rasters with their corresponding labels
        #     'CELLSIZE': 0,             # Use 0 to keep the original cell size
        #     'EXTENT': None,            # Use the extent of the input rasters
        #     'CRS': None,               # Use the CRS of the input rasters
        #     'OUTPUT': output_raster    # Path to the output raster
        # }

        params = {
            'EXPRESSION': expression,  # The weighted average expression
            'OUTPUT': output_raster,   # Path to the output raster file
        }

        # Dynamically add the input rasters with correct variable names
        for idx, raster in enumerate(input_rasters):
            variable = chr(65 + idx)  # A, B, C, ...
            params[f'INPUT_{chr(65 + idx)}'] = raster  # Assign rasters to INPUT_A, INPUT_B, ...
            params[f'BAND_{variable}'] = 1 

        # Set up the rest of the raster calculator parameters
        params['CELLSIZE'] = 0   # Use 0 to keep the original cell size
        params['EXTENT'] = None  # Use the extent of the input rasters
        params['CRS'] = None     # Use the CRS of the input rasters

        print(params)

        try:
            # Run the GDAL raster calculator
            processing.run('gdal:rastercalculator', params)
            print(f"Weighted average raster created successfully. Output saved to: {output_raster}")
        except Exception as e:
            print(f"Error during raster calculation: {e}")
      
    @staticmethod
    def clip_vector_layers(input_layer, overlay_layer, output_layer, **args):
        # Clip
        clip_params = {
            'INPUT': input_layer,
            'OVERLAY': overlay_layer,
            'OUTPUT': output_layer,
            'OUTPUT_GEOMETRY_TYPE': 1
        }
        try:
            results = processing.run('native:clip', clip_params)
            print("clip sucessfully completed")
            return results
        except Exception as e:
            print(f"Error during Clipping: {e}")    
    
    @staticmethod
    def fill_no_data_values(input_layer, output_layer, band, fill_value):
        # QGIS fill NoData cells not GDAL Fill NoDATA
        no_fill_params = {
            'BAND': band,
            'FILL_VALUE': fill_value,
            'INPUT':  input_layer,
            'OUTPUT': output_layer
        }
        try:
            results = processing.run('native:fillnodata', no_fill_params)
            print("fill no data completed sucessfully")
            return results
        except Exception as e:
            print(f"Error nofill cells algorithm: {e}")

    @staticmethod
    def reproject_vector_layer(input_layer, target_crs, output):
        params = {
            'INPUT': input_layer,               
            'TARGET_CRS': target_crs,  
            'OUTPUT': output            
        }

        try:
            results = processing.run("native:reprojectlayer", params)

            print("reprojected completed sucessfully")
            return results
        except Exception as e:
            print(f"Error during reprojection: {e}")
    @staticmethod
    def calculate_zonal_statitics(column_prefix, input_vector_layer,input_raster_layer, raster_band , statistics, output_path, verbose=True):
        params = {
            'COLUMN_PREFIX': column_prefix,
            'INPUT': input_vector_layer,
            'INPUT_RASTER': input_raster_layer,
            'RASTER_BAND': raster_band,
            'STATISTICS': statistics, # [0,1,2], 
            'OUTPUT': output_path,
            'NODATA': -9999,
        }                          
        try:
            result = processing.run("native:zonalstatisticsfb", params)
            if verbose:
                print("zonal statistics calculated successfully")
            return result
        except Exception as e:
            print(f" Error when calculating zonal statistics: {e}")
   
             
    @staticmethod
    def clip_rasters_and_vectors(raster_layers: [List[str]], vector_layers:  [List[str]], output_layers: [List[str]],merged_raster_output):
        for raster_path, vector_path, output_path in zip(raster_layers, vector_layers, output_layers):
     
            raster_layer = QgsRasterLayer(raster_path, "raster")
            mask_layer = QgsRasterLayer(vector_path, "ogr")
            params = {
                "INPUT": raster_path,
                "MASK": vector_path,
                "OUTPUT": output_path
            }
            print(params)
                            
            processing.run("gdal:cliprasterbymasklayer", params)
            print(output_layers)

        # Merge all clipped raster layers
        merge_params = {
            'INPUT': output_layers,   
            'DATA_TYPE': 5,               # Set to 0 for Byte, 1 for Int16, etc.
            'OUTPUT': merged_raster_output 
                }

        result = processing.run("gdal:merge", merge_params)
        return result 


    @staticmethod
    def fix_geometries(input,output, method=None):
 
        
        alg_params = {
            'INPUT':input,
            'METHOD': method,
            'OUTPUT': output
            }
        print(alg_params)    
                            
        result =processing.run('native:fixgeometries', alg_params)
        return result

        
    @staticmethod
    def dissolve_alg(field,input,separate_disjoint,output):
        params = {
            'FIELD': field,
            'INPUT': input,
            'SEPARATE_DISJOINT': separate_disjoint,
            'OUTPUT': output,
            'OUTPUT_GEOMETRY_TYPE': 1
        } 
        result = processing.run('native:dissolve', params)  
        return result
    
    @staticmethod
    def rasterize_polygon(field, input_vector, layer_name, target_resolution, 
                        extent, output, nodata_value=-9999.0, 
                        creation_options=None):
        """
        Rasterize a polygon vector layer using GDAL rasterize
        
        Args:
            field: Attribute field to burn (e.g., 'zs_mean')
            input_vector: Path to input vector file
            layer_name: Name of the layer to rasterize
            target_resolution: Target resolution as tuple (x_res, y_res) or single value
            extent: Target extent as tuple (xmin, ymin, xmax, ymax)
            output: Path to output raster file
            nodata_value: NoData value (default: -9999.0)
            creation_options: Dict of creation options (default: None)
        """
        
        # Handle target resolution
        if isinstance(target_resolution, (int, float)):
            x_res = y_res = target_resolution
        else:
            x_res, y_res = target_resolution

        # Default creation options matching your gdal command
        if creation_options is None:
            creation_options = {
                'COMPRESS': 'ZSTD',
                'ZSTD_LEVEL': '9',
                'TILED': 'YES',
                'BIGTIFF': 'YES',
                'PREDICTOR': '1'
            }
        
        # Convert creation options to GDAL format
        extra_options = []
        for key, value in creation_options.items():
            extra_options.append(f'-co {key}={value}')
        extra_string = ' '.join(extra_options) if extra_options else ''
        
        params = {
            'INPUT': input_vector,
            'FIELD': field,  # The attribute field to burn
            'BURN': None,    # Don't use burn value, use field instead
            'USE_Z': False,
            'UNITS': 1,      # Georeferenced units
            'WIDTH': None,   # Don't specify width, use resolution instead
            'HEIGHT': None,  # Don't specify height, use resolution instead
            'EXTENT': extent,  # Use the specific extent provided
            'NODATA': nodata_value,
            'DATA_TYPE': 5,  # Float32
            'INVERT': False,
            'EXTRA': f'-l {layer_name} -tr {x_res} {y_res} -of GTiff {extra_string}',
            'OUTPUT': output
        }
        
        print("Rasterize parameters:")
        print(params)
        
        try:
            results = processing.run('gdal:rasterize', params)
            return results
        except Exception as e:
            print(f"Error during rasterization: {e}")
            raise

    # @staticmethod
    # def rasterize_polygon(field, input_vector, burn_value, width, height, output):
    #     vector_layer = QgsVectorLayer(input_vector)
    #     vector_extent = vector_layer.extent()
    
        
    #     params = {
    #         'BURN': burn_value, # attribute field to burn 
    #         'DATA_TYPE': 5,  # Float32
    #         'EXTENT': vector_extent,
    #         'EXTRA': None,
    #         'FIELD': field,
    #         'WIDTH': width,  # Raster width in pixels
    #         'HEIGHT': height,  # Raster height in pixels
    #         'INPUT': input_vector,
    #         'INVERT': False,
    #         'NODATA': -9999,
    #         'UNITS': 1,  # Georeferenced units
    #         'USE_Z': False,
    #         'OUTPUT': output
    #     }
    #     print(params)
    #     results = processing.run('gdal:rasterize', params)
    #     return results
   
    @staticmethod
    def merge_rasters(raster_paths: [List[str]], output):
        raster_layers = [QgsRasterLayer(path, f"raster_{i}") for i, path in enumerate(raster_paths)]

        # Default creation options matching your gdal command
        if creation_options is None:
            creation_options = {
                'COMPRESS': 'ZSTD',
                'ZSTD_LEVEL': '9',
                'TILED': 'YES',
                'BIGTIFF': 'YES',
                'PREDICTOR': '1'
            }
        
        # Convert creation options to GDAL format
        extra_options = []
        for key, value in creation_options.items():
            extra_options.append(f'-co {key}={value}')
        extra_string = ' '.join(extra_options) if extra_options else ''

        merge_params = {
            'INPUT': [layer.dataProvider().dataSourceUri() for layer in raster_layers],
            'PCT': False,                # Merge percentage raster
            'SEPARATE': False,           # False merges as a single layer; True keeps layers separate in bands
            'NODATA_INPUT': -9999,        # Set nodata value for input layers if necessary
            'NODATA_OUTPUT': -9999,       # Set nodata value for output if necessary
            'OPTIONS': '',               # Additional GDAL options
            'DATA_TYPE': 5,              # Specify data type, e.g., Float32
            'EXTRA': extra_string,       # Additional options for GDAL
            'OUTPUT': output
                }

        result = processing.run("gdal:merge", merge_params)
        if result:
            print("Raster merge completed successfully.")
            return result 
        else:
            print("Error: Raster merge failed.")
        
     

    @staticmethod    
    def save_raster(raster_path, output_path,target_crs):
        " This method is useful to save a raster in a specific crs equivalent to exporting a file in QGIS"
        raster_layer = QgsRasterLayer(raster_path, "input_raster")
        writer = QgsRasterFileWriter(output_path)
       
        pipe = QgsRasterPipe()
        provider = raster_layer.dataProvider()
        pipe.set(provider)
        # if not pipe.set(provider=raster_layer.dataProvider()):
        #     print("Failed to set data provider in pipe!")

        # writer.setNoDataValue(1, -9999)  # Optional: Set NoData value

        width = raster_layer.width()
        height = raster_layer.height()
        extent = raster_layer.extent()

        success = writer.writeRaster(pipe, width, height, extent, target_crs)
        if success == QgsRasterFileWriter.NoError:
            print("Raster saved successfully.")
        else:
            print("Error saving raster!")

    @staticmethod    
    def assign_projection(target_crs,input):
        alg_params = {
            'CRS':  target_crs,
            'INPUT': input,
           
        }
        
        result = processing.run('gdal:assignprojection', alg_params)
        print(result)

    @staticmethod 
    def raster_calculator(output_path, expression, **kwargs):

        """ this func takes the following args:
                    output_path: path to save the final output
                    expression: expression to pass in
                    input path as arguments provided as kwargs 
        """

        params = {
            'BAND_A': None, 'BAND_B': None, 'BAND_C': None, 'BAND_D': None,
            'BAND_E': None, 'BAND_F': None,
            'FORMULA': expression,            
            'OUTPUT': output_path,
            'NO_DATA': -9999,  
            }

        for i, (key, raster_layer) in enumerate(kwargs.items(), start=1):
            if not raster_layer.isValid():
                raise ValueError(f"The raster layer '{key}' is not valid.")
            params[f'INPUT_{chr(64 + i)}'] = raster_layer.source()
            params[f'BAND_{chr(64 + i)}'] = 1  

   

        processing.run("gdal:rastercalculator", params)    

        print("Raster calculation completed successfully!")

    @staticmethod
    def extractbyattribute(field, input_path, operator, value , output):
        """ args
            input = input vector
            output = path to save the output
            field = attribute_field
            operator =     0: Equal to (=)
                        1: Not equal to (!=)
                        2: Less than (<)
                        3: Less than or equal to (<=)
                        4: Greater than (>)
                        5: Greater than or equal to (>=)
                        6: Like (for string matching)
                        7: Is NULL 
            value = desired value             
        """
        params = {
            'FIELD': field,
            'INPUT': input_path,
            'OPERATOR': operator, 
            'VALUE': value,
            'OUTPUT': output
            }

        print(params)
        
        result = processing.run('native:extractbyattribute', params)
        
        

    @staticmethod
    def creategrid(project_crs, extent, grid_size, shape_type, output):
        params = {
            'CRS':project_crs,
            'EXTENT': extent,
            'HOVERLAY':0,
            'HSPACING': grid_size,
            'TYPE': shape_type,  # 2 =Rectangle (Polygon)
            'VOVERLAY': 0,
            'VSPACING': grid_size,
            'OUTPUT': output
        }
        result = processing.run('native:creategrid', params)
        if result:
            print("creating grid completed successfully.")
            return result 
        else:
                print("creating grid failed.")


    @staticmethod
    def pixels_to_polygons(field_name,input,  band, output):
        params = {
            'FIELD_NAME': field_name,
            'INPUT_RASTER': input,
            'RASTER_BAND': band,
            'OUTPUT': output
        }
        result = processing.run('native:pixelstopolygons', params)
        if result:
            print("pixels to polygons completed successfully.")
            return result 
        else:
            print("pixels to polygons  failed.")

    @staticmethod
    def merge_vector_layers(input_layers: [List[str]], output_layer_path):
        params = {
            'LAYERS': input_layers,         
            'OUTPUT': output_layer_path        
        }

      
        result = processing.run(
            "native:mergevectorlayers",        # Merge vector layers algorithm
            params
        )
        return result

    @staticmethod
    def clip_raster_by_extent(input_raster, extent,no_data_value, output_raster):    
        params = {
            'INPUT': input_raster,     
            'PROJWIN':extent,    
            'NODATA': no_data_value,            
            'OPTIONS': '',              
            'DATA_TYPE': 0,              
            'OUTPUT': output_raster     
        }


        result = processing.run('gdal:cliprasterbyextent', params)
        return result
        
    @staticmethod
    def extract_by_attribute(input_layer, field, operator, value, output):
        params = {
            'INPUT': input_layer,        # Input vector layer
            'FIELD': field,    # The field to filter by (replace with actual field name)
            'OPERATOR': operator,                # Operator (0 = Equal to, see list below for options)
            'VALUE': value,     # The value to filter by (replace with actual value)
            'OUTPUT': output           # Output to temporary memory
        }

        # Run the processing algorithm
        result = processing.run('native:extractbyattribute', params)
        return result


    @staticmethod
    def delete_holes_vector(input_layer, output_layer): 
        result = processing.run("native:deleteholes", {
            'INPUT': input_layer,              
            'OUTPUT': output_layer            
            })
        return result    