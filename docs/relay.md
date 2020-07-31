# relay.py

The relay class provides the network interface for any EDGE client. 

#### Relay

    Relay(
        ip, 
        verbose=False
    )

Creates a new Relay object using the specified `ip`. The `ip` should contain both the ip and the port (`ip:port`), default
port is 8080.  If `verbose` is true, the Relay will log information to `stdout`. The Relay class continuously accepts
images from the host and stores them in a buffer until the connection is closed.  
#### get_image

    get_image(self)
    
If there is an image in the Relay buffer, retrieves it and removes it from the buffer. If there are not any images to be
retrieved, this call blocks until an image is received. 

#### send_results

    get_infer_result(
        self,
        results
    )
    
Sends a set of inference results back to the host. The results should be in the format `[bbox, class, confidence]`,
where `bbox` is a list of four integers representing the bounding box, class is an integer, and confidence is a floating
point number. 

#### close

    close(self)
    
Closes the connection. 

## Usage of unit test (main())

The unit test accepts an IP (using argument `-ip`) and writes a received image to `out.jpg`.