# Second Assignment - Data Mining
***Deadline: June 15 - 2025***

Data for this project consists of two tables in a tab-separated columns format. Each row in those files corresponds to an abstract of a scientific article from ACM Digital Library, which was assigned to one or more topics from the  ACM Computing Classification System (the old one from 1998).

There are two data sets for this task, the training and the testing sample, respectively. They are text (TSV - tab separated values) files compressed using 7zip.

The training data (DM2023_training_docs_and_labels.tsv) has three columns: the first one is an identifier of a document, the second one stores the text of the abstract, and the third one contains a list of comma-separated topic labels.

The test data (DM2023_test_docs.tsv) has a similar format, but the labels in the third column are missing.

**The task and the format of submissions:** the task for you is to predict the labels of documents from the test data and submit them to the moodle using the link below. A correctly formatted submission should be a text file with exactly 100000 lines plus the report. Each line should correspond to a document from the test data set (the order matters!) and contain a list of one or more predicted labels, separated by commas. The report can be in the form of R/Python notebook (with code and computation outcomes). Please remember about explanations and visualizations – make this report as interesting for a reader as you can.

You may make several submissions (up to 20), so please remember to clearly mark the final version of your answer in case there is more than one.

Practical note: The submission size in moodle is limited to 512MB. In case your files are larger please use compression (7z,gz, ...) other than Zip. Moodle does not like .zip files. 

Evaluation: the quality of submissions will be evaluated using a script to compute the average F1-score measure, i.e., for each test document, the F1-score between the predicted and true labels will be computed, and the values obtained for all test cases will be averaged.

Good luck!

***Data links:*** 
* [Train text & labels](https://drive.google.com/file/d/1xtVxkhUN__lA3m3cWU5Z6UCI7PJuJ1F3/view)
* [Only text](https://drive.google.com/file/d/1TM-L7fEYdNHOlCFWHcw8ksMNrkoodMfk/view)