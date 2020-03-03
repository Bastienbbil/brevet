# Student challenge

As part of the Data Camp course, within the <a href="https://datascience-x-master-paris-saclay.fr/le-master/structure-des-enseignements/">University of Paris-Saclay Datascience
Masters</a>, students were required to build a data challenge addressing
some social/science/business problem using data obtained from external sources.
This is one example of a student challenge, showcased on the ramp.studio server.

# Prediction "Brevet des collèges" success rate

Every year in France, more than 800 000 students aged around 15 take a compulsory national exam, the "Brevet des collèges" which marks the end of four years of studies. The national average is 86.5% of success.  This project aims at predicting the average success rate at the brevet des colleges for each french public college and  identifying factors of the success of the best colleges and the degrees of flexibility of the government to reduce inequalities.

## Set up

Open a terminal and

- install the ramp-workflow library (if not already done)

```
pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
```
- Follow the ramp-kits instructions from the <a href='https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit'>wiki</a>

## Local notebook

- Get started on this RAMP with the <a href="https://github.com/CedricAllainEnsae/success-rates-brevet/blob/master/Starting_kit_taux-de-succes-brevet.ipynb"> starting kit notebook </a>

- The notebook data_camp_get_data gives the scrapping code for the target.

- WARNING: The maps are not visible in the notebook on the github because they are dynamic. So you need to run the notebook locally and  make sure you have an internet connection because they need to access some url on <a href='https://france-geojson.gregoiredavid.fr/' >FranceGeoJson</a> where some dynamic maps are held.
To see the notebook correctly compiled with the dynamics maps, please click <a href='https://nbviewer.jupyter.org/github/CedricAllainEnsae/success-rates-brevet/blob/master/Starting_kit_taux-de-succes-brevet.ipynb'>here</a>.

- To test the starting-kit, please run the following command

```
ramp_test_submission --starting_kit
```
