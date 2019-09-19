# mlflowrate: Software Explanation

## How to use:
Oilwell data can arrive via excel spreadsheets or mixed tagged csv formats. It's important to be able to properly integrate these styles into consistent dataframes for machine learning!

The main modules revolve around the data integration and data science pipeline, the modules are:

1. Integrate
2. DataSets
3. Explore

### WorkFlow Module
The data management parent module: **WorkFlow**, controls the flow of data between these three modules.  
Users can use:    `WorkFlow.status()`    to check the stage in the pipeline they're at.  
The    `WorkFlow.nextphase()`    method enables user progression onto the next stage.

Users must import the data first as a Spark DataFrame: The original file types should be converted to .csv files, then use the Spark API to import the csv file as a Spark dataframe.

We assume timeseries data given the nature of predicting flow rates: all date columns must be renamed as **"datetime"** for the software to work!

After importing the csv files as Spark DataFrames, to start using mlflowrate, one must name each data and collect them into a dictionary. This is then passed into an instantiation of a WorkFlow object:
    
    datas = {data1: Spark DataFrame, data2: Spark DataFrame}
    pipeline = WorkFlow(datas)

Finally, the user can now progress onto the 3 data science stages, for example:

    pipeline.integrate.cleandata()
    pipeline.integrate.organise()
    
    pipeline.nextphase()
    
    pipeline.datasets.makeset()
    pipeline.datasets.correlation()

**Check the user documentations for the features and descriptions!**

### Important things to note:

1. **Data Formats**  
  Two types of common data formats are targeted for data integration:
  
    1.1 **Tagname Format**  
      Oil well characteristic data given in the columnar form: 

          | datetime | tag | value |    

      The tag name may contain one or two pieces of information for us to sort through.  
      In our case, the data provided contained information on measurement origins and what the measurement was.

          E.g. OILWELL-TEMPXXYY   

      The function    `organise_data()`    organises data into their unique oil well origins and sensor measurement types. See the documentations for further information.
    
    1.2 **Standard Format**      
      Oil well characteristic data given as multiple features against the date column.

          | datetime | temp | pres | choke | etc. |    

      The function   `organise_data()`    also organises this data for integrating into the mlflowrate pipeline.

2. **Understanding how to integrate and clean the data**  
  A problem with oil well data given in the "Tagname Format" is that we can have incontiguous features.
  
      E.g: Temperature has 365 samples spanning from 2016-2017, but pressure has 720 samples spanning from 2016-2018.  

      This means when the two features are combined into a DataFrame, temperature will contain an extra 365 nulls to fill up the empty space to keep the data symmetry.  

      mlflowrate organises these incontiguous data into two formats for cleaning and integration:  
        1. **Spark DataFrame Format**  
        2. **Dictionary Format**  

      The Dictionary Format separates out the individual features and collected them into their own key-value pairs in a dictionary.  
      For example:

          Dictionary = {temperature: | datetime | value |,  pressure: | datetime | value |}

      Every data passed into the software will have a Spark DataFrame Format, and a Dictionary Format.
      Using any of the cleaning methods in the package will change both of these formats.

      It is the users job to produce a contiguous and clean Spark DataFrame format for the next stage. The    `.status()`    method enables users to check for duplicates, nulls and sample sizes for assistance.
