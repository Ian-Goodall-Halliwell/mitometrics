This is all the code I used for mitometer analysis.

Here is how to use it:

1. Open Fiji and open all the videos you want to analyze.
2. Run the Fiji macro named "saveproc.ijm", save the outputs to the "newtf" folder.
3. In the MitometerApp folder, right click GUI2.mlapp and select edit.
4. Go to code view, locate line 1996 and change "C:/Users/Ian/Desktop/old_desktop/mito/mitometrics/out" to the location of your "out" folder.
5. Save and open GUI2.mlapp, select "Start 2D", and select all the files you want to analyze.
6. After that is done, the raw results will be in the "out" folder.
7. Open the mitometrics directory and run the python script "dataprocessing.py", this will generate human readable results in a results folder.