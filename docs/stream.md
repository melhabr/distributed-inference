# Stream.py

The stream file provides a class for distributing a video stream to multiple inference endpoints, as well as a class for
recording stream metrics. 

## DistributedStream class

#### DistributedStream

    DistributedStream(
        self,
        port,
        labels,
        verbose=False
    )
    
Initializes a new DistributedStream object, on the given port. Accepts a path to a labels file, which is optional. Set
verbose to True to enable verbose logging. This function creates a Host object, which requires user input to indicate 
when the host is done accepting connections. 

#### stream_video

    stream_video(
        self,
        input
    )
    
Performs an inference stream with input video file path `input`. The streamer will send a frame to all ready devices 
(where a ready device is one that is not currently processing a frame), retrieve the results, and simultaneously draw
the results onto a video file at `out.avi`. 

### Unit Test

The `main()` function for this file provides a basic test of the interface, where the respective arguments can be
specified using `--input`, `--port`, `--labels`, and `--verbose`. An example usage could be

`python3 stream.py --input video.mp4 --port 8080 --labels labels.txt --verbose`