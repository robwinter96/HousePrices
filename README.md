# UK House Prices 
 <i>By Rob Winter, 12/10/2022</i>

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The satndard Anaconda distribution of Python should be sufficient to run the project files

The code was created and run using Python versions 3.* and higher

## Project Motivations <a name="motivation"></a>

### Introduction 

Across the UK, first time buyers have seen houses become further and further out of reach, despite incetntives such as Help To Buy and Lifetime ISA's and no stamp duty for first time buyers (up to a £425k). 

What I want to look at in this project is how we might be able to find the best opportunity for first time buyers to get on the property ladder.

### Potential reasons for un-attainable house prices

It's been suggested that this has been driven by multiple factors which include:
- Low interest rates
- Low housing supply
- Low unemployment
- House prices rising faster than wage growth.


### First time buyer questions to answer 
The following project aims to answer questions in relation to first time buyers problems. They are as follows:

1. Are prices actually rising due to the aforementioned reason?
2. Can we predict what mean house prices in certain countys will be in the future given these "Big factors" to then be able to seek out buying opportnities?
3. When's the best time to buy in your area? (Seasonally and economically)

## File Descriptions <a name="files"></a>

There is a single notebooks available here to showcase work related to the above questions.  The notebook contains multiple sections which guide the reader through the wrangling, cleaning, exploring and analysis of the data. Markdown cells were used throughout to assist in walking through the thought process for individual steps.  

There is an additional `.py` file that runs the necessary code to obtain the final model used to predict salary.

## Results <a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@robwinter96/this-is-how-you-can-get-on-the-property-ladder-f285d127f2c2).

One thing to note is that the lack of data surrounding the size of the properties/number of bedrooms limited the conclusions that could be made.

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>


The data source for this project was found at the locations linked below. These datasets were then merged together to make conclusions from. Ackknowledgements must therefore go to **HMRC Land Registry**, **Official National Statistics (ONS) UK**, ?, ?.

- **UK Interest Rate data** <i>(from 1975 to 2022)</i>. **Source**: https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp#
- **UK Individual House Sales Data** <i>(from 1995 to 2017)</i>. **Source**: https://landregistry.data.gov.uk/app/ppd
- **UK Unemployment Data** <i>(from 1971 to 2022)</i>. **Source**: https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms
- **UK GDP Data** <i>(from 1997 to 2022)</i>. **Source**: https://www.ons.gov.uk/economy/grossdomesticproductgdp/datasets/monthlygrossdomesticproductbygrossvalueadded
- **UK New Build House Data** <i>(from 1995 to 2015)</i>. **Source**: https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/numberofresidentialpropertysalesforsubnationalgeographiesnewlybuiltdwellingshpssadataset22


Contains HM Land Registry data © Crown copyright and database right 2021. This data is licensed under the Open Government Licence v3.0.
