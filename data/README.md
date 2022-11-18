VEP (Visually Evoked Potential) Experimental Data 
=================================================

Contains the data from a visually evoked potential experiment conducted by W. Dave Hairston at the Army Research Laboratory in Aberdeen Maryland as part of a larger headset comparision study. The study has 18 subjects. The collection is presented in two forms: a compressed archive of the raw data and a zip file of labeled power features. This data archive is being contributed in support of a data-in-brief paper:  

> Robbins, K., Su, K-M, and Hairson, W.D.  
> An 18-subject EEG data collection using a visual-oddball task, designed for benchmarking algorithms and headset performance comparisons. 
> Data-in-brief, submitted

The data is being released as a supporting data collection for:  
The PREP pipeline is freely available under the GNU General Public License. 
Please cite the following publication if using:  

> Su K-M, Hairston, W.D. and Robbins KA  (submitted)    
> EEG-Annotate: Automated identification and labeling of events in continuous signals with applications to EEG  
> Front. Neuroinform. 9:16. doi: 10.3389/fninf.2015.00016   

References to the original publications describing this data are:  

> Hairston, W.D., Whitaker, K.W., Ries, A.J., Vettel, J.M., Bradford, J.C., Kerick, S.E., McDowell, K., 2014  
> Usability of four commercially-oriented EEG systems  
> J. Neural Eng. 11, 046018 doi:10.1166/jnsne.2014.1092  
> http://www.ingentaconnect.com/content/asp/jnsne/2014/00000003/00000001/art00003  

> Ries, A.J., Touryan, J., Vettel, J., McDowell, K., Hairston, W.D., 2014 
> A comparison of electroencephalography signals acquired from conventional and mobile systems  
> J. Neurosci. Neuroengineering  3, 10–20  


### Raw data

The raw data is presented as EEGLAB .set files for processing in MATLAB. It has not been processed, other than to insert events and channel locations. The data directory structure is organized as an EEG Study (ESS) and contains an XML manifest describing the data. The structure and the data form a container for easy automated processing. The events have also been identified with Hierarchical Event Descriptors (HED) tags.  

For additional information about ESS see http://www.eegstudy.org/ and the paper:  
> Bigdely-Shamlo, N. Makeig, S. and Robbins KA (2016)  
> Preparing laboratory and real-world EEG data for large-scale analysis: A containerized approach  
> Front. Neuroinform., 08 March 2016. http://journal.frontiersin.org/article/10.3389/fninf.2016.00007  		 

For additional information about HED tags see http://www.hedtags.org/ and the paper:  
> Bigdely-Shamlo N, Cockfield, J. Makeig, S. Rognon, T. La Valle, C. Miyakoshi, M. and Robbins KA (2016)  
> Hierarchical Event Descriptors (HED): Semi-Structured Tagging for Real-World Events in Large-Scale EEG  
> Front. Neuroinform., 17 October 2016. http://journal.frontiersin.org/article/10.3389/fninf.2016.00042  

## Feature data 

Each dataset in the feature data is a .mat file containing two variables: 

* labels - cell array containing the labels for the overlapping features windows. The labels are '34', '35' or empty. The label 34 corresponds to display of a 'friend' or non-target image. 
* samples - a featureSize x numberFeatures array with containing the actual feature vectors.

The feature vectors have been extracted from the VEP datasets after preprocessing with the PREP pipeline and then cleaning with the MARA automated pipeline. Both of these are freely available as EEGLAB plugins. The computation of feature vectors is available in the freely available toolbox EEG-Annotate (https://github.com/VisLab/EEG-Annotate).  

PREP:  
> Bigdely-Shamlo N, Mullen T, Kothe C, Su K-M and Robbins KA (2015)  
> The PREP pipeline: standardized preprocessing for large-scale EEG analysis  
> Front. Neuroinform. 9:16. doi: 10.3389/fninf.2015.00016  
> https://github.com/VisLab/EEG-Clean-Tools

MARA:  
> Winkler, I., Brandl, .,  Horn, F., Waldburger, E., Allefeld, C. and M. Tangermann (2014)  
> Robust artifactual independent component classification for BCI practitioners  
> J. Neural Eng., vol. 11, no. 3, p. 035013, 2014.

EEGLAB:  
> Delorme, A. and S. Makeig (2004)  
> EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis  
> J. Neurosci. Methods, vol. 134, no. 1, pp. 9–21, Mar. 2004.
	

While the 		 	 
and as a supporting data collection for:  
The PREP pipeline is freely available under the GNU General Public License. 
Please cite the following publication if using:  
> Bigdely-Shamlo N, Mullen T, Kothe C, Su K-M and Robbins KA (2015)  
> The PREP pipeline: standardized preprocessing for large-scale EEG analysis  
> Front. Neuroinform. 9:16. doi: 10.3389/fninf.2015.00016  		 