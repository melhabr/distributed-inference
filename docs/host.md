# host.py

The host class provides the behavior for an inference host, as well as a basic test for distributed image inference. 

#### Host

    Host(
        port=8080, 
        verbose=False
    )

Creates a new host object using the specified `port`. If `verbose` is true, the Host will log information to `stdout`.
When a new Host is created, the host will accept TCP connections on the specified port, and collect connections until
the user presses the enter key. After the enter key is pressed, the Host will still accept new connections, but not add
them to the connection pool (i.e. the client connection will be accepted but the host will not service it). 

#### send_image

    send_image(
        self,
        conn,
        img
    )
    
Sends the image `img` to the specified connection `conn`. 

#### get_infer_result

    get_infer_result(
        self,
        conn
    )
    
Blocking function that returns an inference result for the specified connection, if one is received. Returns `None` if
the connection is closed at any point during the retrieval. If a result is successfully retrieved, the function returns
an array of proposals (which may be any size, including empty). The proposals are each of size 6, containing 
`[x1, y1, x2, y2, class, confidence]`, where the first four entries are the coordinates of the bounding box, `class` is 
the image class, and `confidence` is the inference of the proposal. 

#### get_connnections

    get_connections(self)
    
Returns an interator through the active connections held by the Host. While you can directly access the connections
through `host.conns`, using this method is safer, as it will automatically detect and skip over deleted connections. 

## Usage of image test (main())

The main function in the `host.py` file performs a basic inference test with a single image on any number of EDGE devices.
The arguments are as follows:

`--input (-i)` - Path to input image. This is required.

`--port (-p)` - Port to use for host application. Default is 8080

`--labels (-l)` - Path to a labels file. This is not required.

An example usage would be as follows: Start the host, and start an arbitrary number of clients. Once all the clients
are connected, press enter, and the program will sequentially perform the inference on each device. For each device, the
host will send the image, retrieve the result, and draw the result on the image in a folder named `out`.