## mlflowrate: Software Explanation

### How to use:
Oilwell data can arrive via excel spreadsheets or mixed tagged csv formats. It's important to be able to properly integrate these styles into consistent dataframes for machine learning!

1. Data Formats  
  All data is in timeseries given the nature of predicting flow rates.  
  Two types of common data integration formats have been developed:
  
  - Tagnames Format  
      Oil well characteristic data given in the columnar form: 

      | date | tag | value |

      The tag name may contain one or two pieces of information for us to sort through.  
      In our case, the data provided contained information on measurement origins and what the measurement was.  
      E.g. OILWELL-TEMPXX  
      We provide the feature    organise_data()    to be able to organise data into their unique Oilwell origins and sensor measurement types. See the documentations for further information.

