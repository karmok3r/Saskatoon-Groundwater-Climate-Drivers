/******************************************************************************
 * Groundwater Climate Drivers — WSA Observation Wells, Saskatchewan
 * Google Earth Engine (Code Editor, JavaScript)
 *
 * What it does
 *   1. Extracts monthly climate drivers at each well: ERA5-Land (snowpack,
 *      snowmelt, precipitation, evaporation, soil moisture, runoff) and
 *      TerraClimate (precipitation, AET/PET, deficit, SWE, PDSI), 1960–present
 *   2. Extracts GRACE/GRACE-FO terrestrial water storage anomalies (JPL
 *      mascons), 2002–present
 *   3. Extracts built-up surface around each well (GHSL, 1975–2025), to test
 *      whether urban growth affects the record (e.g. WSA Saskatoon)
 *   4. Exports long-format CSV tables to Google Drive, to be joined with the
 *      WSA monthly water levels in Python
 *
 * Well coordinates are approximate (±1–2 km), converted from WSA's Dominion
 * Land Survey locations by wsa_wells.py.
 *
 * Author: Uday Karmoker
 ******************************************************************************/

// ============================ USER SETTINGS =================================

// All WSA observation-well locations with a recovered hydrograph
// (from wells_province.py; wells sharing a location are listed once).
var WELLS = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([-107.0945, 52.0346]), {name: 'WSA Agrium 43'}),
  ee.Feature(ee.Geometry.Point([-103.9406, 53.0675]), {name: 'WSA Armley'}),
  ee.Feature(ee.Geometry.Point([-108.866, 52.8266]), {name: 'WSA Atton’s Lake'}),
  ee.Feature(ee.Geometry.Point([-105.4939, 50.3011]), {name: 'WSA Baildon 59'}),
  ee.Feature(ee.Geometry.Point([-105.4743, 50.2682]), {name: 'WSA Baildon 60'}),
  ee.Feature(ee.Geometry.Point([-102.2805, 50.8814]), {name: 'WSA Bangor A and B'}),
  ee.Feature(ee.Geometry.Point([-107.74, 55.1404]), {name: 'WSA Beauval'}),
  ee.Feature(ee.Geometry.Point([-106.2106, 52.0383]), {name: 'WSA Blucher No. 3 and No. 4'}),
  ee.Feature(ee.Geometry.Point([-105.5216, 52.2682]), {name: 'WSA Bruno'}),
  ee.Feature(ee.Geometry.Point([-104.905, 50.7609]), {name: 'WSA Buena Vista 09-93'}),
  ee.Feature(ee.Geometry.Point([-104.8928, 50.75]), {name: 'WSA Buena Vista 18-94'}),
  ee.Feature(ee.Geometry.Point([-104.8934, 50.7609]), {name: 'WSA Buena Vista 20-95'}),
  ee.Feature(ee.Geometry.Point([-104.8932, 50.7573]), {name: 'WSA Buena Vista 21-95'}),
  ee.Feature(ee.Geometry.Point([-107.1717, 51.5821]), {name: 'WSA Conquest No.500'}),
  ee.Feature(ee.Geometry.Point([-107.3128, 51.5857]), {name: 'WSA Conquest No.501'}),
  ee.Feature(ee.Geometry.Point([-107.3126, 51.5784]), {name: 'WSA Conquest No.502'}),
  ee.Feature(ee.Geometry.Point([-105.678, 49.1259]), {name: 'WSA Coronach 1'}),
  ee.Feature(ee.Geometry.Point([-102.443, 50.9325]), {name: 'WSA Crater Lake'}),
  ee.Feature(ee.Geometry.Point([-106.7187, 52.2718]), {name: 'WSA Dalmeny'}),
  ee.Feature(ee.Geometry.Point([-109.3051, 49.5949]), {name: 'WSA Davis Creek 1A and 1B'}),
  ee.Feature(ee.Geometry.Point([-106.221, 52.9288]), {name: 'WSA Duck Lake No. 1'}),
  ee.Feature(ee.Geometry.Point([-103.1773, 49.2682]), {name: 'WSA Estevan No. 1 and No. 2'}),
  ee.Feature(ee.Geometry.Point([-105.8345, 49.2026]), {name: 'WSA Fife Lake 002'}),
  ee.Feature(ee.Geometry.Point([-102.855, 49.7098]), {name: 'WSA Forget'}),
  ee.Feature(ee.Geometry.Point([-108.528, 49.7573]), {name: 'WSA Garden Head'}),
  ee.Feature(ee.Geometry.Point([-106.5017, 52.0675]), {name: 'WSA Goodale Farm 009'}),
  ee.Feature(ee.Geometry.Point([-106.2668, 52.5091]), {name: 'WSA Hague'}),
  ee.Feature(ee.Geometry.Point([-109.5552, 52.093]), {name: 'WSA Hearts Hill'}),
  ee.Feature(ee.Geometry.Point([-108.3305, 49.7609]), {name: 'WSA Instow'}),
  ee.Feature(ee.Geometry.Point([-107.9091, 52.7682]), {name: 'WSA Lilac'}),
  ee.Feature(ee.Geometry.Point([-108.3419, 54.1842]), {name: 'WSA Meadow Lake'}),
  ee.Feature(ee.Geometry.Point([-104.4508, 52.9616]), {name: 'WSA Melfort'}),
  ee.Feature(ee.Geometry.Point([-105.0641, 51.5127]), {name: 'WSA Nokomis'}),
  ee.Feature(ee.Geometry.Point([-103.2524, 49.1405]), {name: 'WSA Outram'}),
  ee.Feature(ee.Geometry.Point([-104.4296, 50.4945]), {name: 'WSA Pilot Butte South'}),
  ee.Feature(ee.Geometry.Point([-104.4319, 50.5383]), {name: 'WSA Regina 2010-04(LF)'}),
  ee.Feature(ee.Geometry.Point([-104.6839, 50.5273]), {name: 'WSA Regina 530'}),
  ee.Feature(ee.Geometry.Point([-104.4759, 50.5018]), {name: 'WSA Regina 96-01'}),
  ee.Feature(ee.Geometry.Point([-104.4757, 50.4982]), {name: 'WSA Regina 96-02'}),
  ee.Feature(ee.Geometry.Point([-104.4618, 50.5073]), {name: 'WSA Regina D1-86'}),
  ee.Feature(ee.Geometry.Point([-104.4622, 50.5146]), {name: 'WSA Regina TH81-2'}),
  ee.Feature(ee.Geometry.Point([-104.3162, 50.1697]), {name: 'WSA Riceton'}),
  ee.Feature(ee.Geometry.Point([-106.5267, 52.1806]), {name: 'WSA Saskatoon'}),
  ee.Feature(ee.Geometry.Point([-108.5024, 49.6989]), {name: 'WSA Shaunavon'}),
  ee.Feature(ee.Geometry.Point([-105.1839, 51.4653]), {name: 'WSA Simpson 13-04'}),
  ee.Feature(ee.Geometry.Point([-105.1898, 51.4653]), {name: 'WSA Simpson 16-05'}),
  ee.Feature(ee.Geometry.Point([-103.0493, 53.3813]), {name: 'WSA Smoky Burn A and B'}),
  ee.Feature(ee.Geometry.Point([-102.4045, 51.8302]), {name: 'WSA Stenen'}),
  ee.Feature(ee.Geometry.Point([-107.056, 51.6587]), {name: 'WSA Swanson'}),
  ee.Feature(ee.Geometry.Point([-107.5108, 51.8886]), {name: 'WSA Tessier'}),
  ee.Feature(ee.Geometry.Point([-108.417, 51.031]), {name: 'WSA Tyner'}),
  ee.Feature(ee.Geometry.Point([-108.951, 52.4762]), {name: 'WSA Unity'}),
  ee.Feature(ee.Geometry.Point([-107.0407, 52.0164]), {name: 'WSA Vanscoy'}),
  ee.Feature(ee.Geometry.Point([-108.8872, 50.3777]), {name: 'WSA Verlo'}),
  ee.Feature(ee.Geometry.Point([-102.4974, 51.1478]), {name: 'WSA Yorkton 517 and 519'}),
  ee.Feature(ee.Geometry.Point([-102.4512, 51.177]), {name: 'WSA Yorkton 93-07'}),
  ee.Feature(ee.Geometry.Point([-102.4105, 51.1879]), {name: 'WSA Yorkton 95-04B'})
]);

// Record period. The oldest wells start in 1966; TerraClimate starts in 1958.
var START = '1960-01-01';
var END   = '2026-11-01';

// Radius for built-up surface around each well (m)
var URBAN_RADIUS_M = 2000;

// Scales (m), roughly the native grid of each dataset
var ERA5_SCALE  = 11132;   // 0.1 degree
var TERRA_SCALE = 4638.3;  // 1/24 degree
var GRACE_SCALE = 55660;   // 0.5 degree grid (mascons are ~3 degrees)

// Export settings
var EXPORT_FOLDER = 'gee_sask_groundwater_province';

// Well used for the Console preview charts
var PREVIEW_WELL = 'WSA Dalmeny';

// ============================ HELPERS =======================================

// Samples every image of a monthly collection at every well and returns one
// row per well and month, with the band names as columns.
function sampleMonthly(collection, scale) {
  return collection.map(function (img) {
    var date = img.date();
    return img.reduceRegions({
      collection: WELLS,
      reducer: ee.Reducer.first().forEach(img.bandNames()),
      scale: scale
    }).map(function (f) {
      return f.set({date: date.format('YYYY-MM'), year: date.get('year'),
                    month: date.get('month')})
              .setGeometry(null);
    });
  }).flatten();
}

// ============================ ERA5-LAND =====================================

// Monthly aggregates. Temperature K -> degC; water fluxes and stores m -> mm.
// ECMWF signs evaporation negative (upward), so evaporation is flipped to
// positive here.
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterDate(START, END)
  .map(function (img) {
    var water = img.select([
      'total_precipitation_sum', 'snowfall_sum', 'snowmelt_sum',
      'snow_depth_water_equivalent', 'surface_runoff_sum',
      'sub_surface_runoff_sum'
    ]).multiply(1000);
    var evap = img.select(['total_evaporation_sum', 'potential_evaporation_sum'])
      .multiply(-1000);
    var soil = img.select([
      'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
      'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4'
    ]);
    var temp = img.select('temperature_2m').subtract(273.15);
    return ee.Image(water.addBands(evap).addBands(soil).addBands(temp)
      .copyProperties(img, ['system:time_start']));
  });

print('ERA5-Land months', era5.size());

// ============================ TERRACLIMATE ==================================

// Applies the TerraClimate scale factors (pr, swe and ro are already in mm)
var terra = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
  .filterDate(START, END)
  .map(function (img) {
    var mm = img.select(['pr', 'swe', 'ro']);
    var tenths = img.select(['aet', 'pet', 'def', 'soil', 'tmmx', 'tmmn'])
      .multiply(0.1);
    var pdsi = img.select('pdsi').multiply(0.01);
    return ee.Image(mm.addBands(tenths).addBands(pdsi)
      .copyProperties(img, ['system:time_start']));
  });

print('TerraClimate months', terra.size());

// ============================ GRACE / GRACE-FO ==============================

// JPL mascons with Coastline Resolution Improvement; liquid-water-equivalent
// thickness anomaly (cm). Includes the 2017–2018 gap between missions.
var grace = ee.ImageCollection('NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI')
  .filterDate(START, END)
  .select(['lwe_thickness']);

print('GRACE months', grace.size());
print('GRACE first and last month',
  ee.Date(grace.aggregate_min('system:time_start')).format('YYYY-MM'),
  ee.Date(grace.aggregate_max('system:time_start')).format('YYYY-MM'));

// ============================ BUILT-UP SURFACE ==============================

// GHSL built-up surface (m2 per 100 m cell), 5-yearly epochs. Epochs after
// 2020 are model projections.
var ghsl = ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S')
  .filter(ee.Filter.lte('system:index', '2025'))
  .select('built_surface');

var builtTable = ghsl.map(function (img) {
  var epoch = ee.Number.parse(img.get('system:index'));
  return img.reduceRegions({
    collection: WELLS.map(function (f) { return f.buffer(URBAN_RADIUS_M); }),
    reducer: ee.Reducer.sum().setOutputs(['built_m2']),
    scale: 100
  }).map(function (f) {
    var areaM2 = Math.PI * URBAN_RADIUS_M * URBAN_RADIUS_M;
    return f.set({epoch: epoch,
                  built_pct: ee.Number(f.get('built_m2')).divide(areaM2).multiply(100)})
            .setGeometry(null);
  });
}).flatten();

// ============================ TABLES ========================================

var era5Table  = sampleMonthly(era5, ERA5_SCALE);
var terraTable = sampleMonthly(terra, TERRA_SCALE);
var graceTable = sampleMonthly(grace, GRACE_SCALE);

// Well locations, for checking on the map and in the CSVs
var wellTable = WELLS.map(function (f) {
  var c = f.geometry().coordinates();
  return f.set({lon: c.get(0), lat: c.get(1)});
});

// ============================ PREVIEW =======================================

Map.centerObject(WELLS, 6);
Map.addLayer(WELLS, {color: 'red'}, 'WSA observation wells');

var previewWell = WELLS.filter(ee.Filter.eq('name', PREVIEW_WELL)).first().geometry();

print(ui.Chart.image.series({
  imageCollection: era5.filterDate('2000-01-01', END)
    .select(['snow_depth_water_equivalent', 'snowmelt_sum', 'total_precipitation_sum']),
  region: previewWell,
  reducer: ee.Reducer.first(),
  scale: ERA5_SCALE
}).setOptions({
  title: 'ERA5-Land at ' + PREVIEW_WELL + ', 2000–present (mm)',
  vAxis: {title: 'mm'},
  lineWidth: 1
}));

print(ui.Chart.image.series({
  imageCollection: grace,
  region: previewWell,
  reducer: ee.Reducer.first(),
  scale: GRACE_SCALE
}).setOptions({
  title: 'GRACE/GRACE-FO water storage anomaly at ' + PREVIEW_WELL + ' (cm)',
  vAxis: {title: 'cm LWE'},
  lineWidth: 1, pointSize: 2
}));

print('Built-up surface within ' + URBAN_RADIUS_M + ' m (%)',
  ui.Chart.feature.groups(builtTable, 'epoch', 'built_pct', 'name')
    .setChartType('LineChart')
    .setOptions({title: 'Built-up share around each well (GHSL)',
                 hAxis: {title: 'Epoch', format: '####'}, vAxis: {title: '%'}}));

// ============================ EXPORTS =======================================

Export.table.toDrive({
  collection: era5Table,
  description: 'era5land_monthly_wells',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV',
  selectors: ['name', 'date', 'year', 'month',
              'total_precipitation_sum', 'snowfall_sum', 'snowmelt_sum',
              'snow_depth_water_equivalent', 'surface_runoff_sum',
              'sub_surface_runoff_sum', 'total_evaporation_sum',
              'potential_evaporation_sum', 'volumetric_soil_water_layer_1',
              'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
              'volumetric_soil_water_layer_4', 'temperature_2m']
});

Export.table.toDrive({
  collection: terraTable,
  description: 'terraclimate_monthly_wells',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV',
  selectors: ['name', 'date', 'year', 'month', 'pr', 'swe', 'ro', 'aet', 'pet',
              'def', 'soil', 'tmmx', 'tmmn', 'pdsi']
});

Export.table.toDrive({
  collection: graceTable,
  description: 'grace_monthly_wells',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV',
  selectors: ['name', 'date', 'year', 'month', 'lwe_thickness']
});

Export.table.toDrive({
  collection: builtTable,
  description: 'ghsl_builtup_wells',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV',
  selectors: ['name', 'epoch', 'built_m2', 'built_pct']
});

Export.table.toDrive({
  collection: wellTable,
  description: 'wells',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV',
  selectors: ['name', 'lat', 'lon']
});
