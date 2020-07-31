# Stopwatch.py

The stopwatch class is used to report times for different operations on EDGE client devices. The stopwatch can record
average wall and process times for pre-processing, inference, and post-processing. 

#### Stopwatch

     Stopwatch(
         self,
         report_interval=10
     )
     
Creates a stopwatch object that allows a report every `report_interval` seconds. 

#### start

    start(self)

Starts the stopwatch (for any mode).

#### stop

     stop(
         self,
         mode    {MODE_PREPROCESS, MODE_INFER, MODE_POSTPROCESS)
     ) 

Stops the watch and marks the duration as the specified mode. 

#### report

    report(
        self,
        force=False
    )
    
Prints a report to `stdout` only if the report interval has been reached. Returns True if a report is produced, False
otherwise. Set `force=True` to force a report to be produced. 