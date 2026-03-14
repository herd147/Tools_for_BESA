# Tools_for_BESA
Some code which can be used for BESA swf and bsa processing

excel_to_json: function to convert Excel-file to JSON file, optimized for Excel_table_template 

readASCII: function that reads your .swf files and returns a numpy array

plot_properly: functions for different plots (hp,irn or rho)

sustained_new: processes your sustained field and compares between groups

psychoacoustics_processing: optimized for Excel_table_template after conversion to .json, analyses psychoacoustic data automatically, with statistics and boxplot, saved into a text file

simple_bca_bootstrap: Bootstraps all your npy. arrays

bootstrap_t: inner and outer bootstrap, works with npy-files and self-defined slices

bootstrap_ci_plot: plots bootstrap average and bootstrap-CI, works with bootstrap matrices (perform simple_bca_bootstrap first)
